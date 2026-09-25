"""Tool registration and MCP-compatible execution results.

Text blocks always include ``truncated`` and ``full_size``. ``full_size`` is
the original UTF-8 byte length before a character cap is applied. Fetch pages
also include ``full_size_chars`` for their readable-text character length.
"""

from __future__ import annotations

import asyncio
import copy
import os
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.abort import AbortGenerationRegistry
from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRequest,
)
from ..core.approval import canceled_result as _canceled_result
from ..core.safety import SafetyTier
from ..core.store import ConversationStore
from ..protocol.types import (
    StructuredToolResult,
    ToolCall,
    ToolResult,
    ToolSchema,
    flatten_tool_content,
)
from ..runtime.execution import (
    ToolExecutionContext,
    ToolHandler,
    ToolHandlerResult,
    ToolLifecycleSink,
    ToolStream,  # noqa: F401 - preserve the public registry import
    ToolStreamPublisher,  # noqa: F401 - preserve the public registry import
    ToolStreamSink,
    _signal_is_set,
    _ToolCallStreamPublisher,
    _ToolCanceled,  # noqa: F401 - preserve the exec tool's import
    _yield_for_abort,  # noqa: F401 - preserve the read tool's import
    bind_execution_context,
    build_execution_arguments,
    run_handler_with_abort,
)
from ..skills import SkillCatalog
from ._discovery import _register_discovered_tools
from ._results import (
    BoundedText as _BoundedText,  # noqa: F401 - preserve the public registry import
)
from ._results import (
    _apply_error_governance,
    _error_result,
    _legacy_result,
    _normalize_result,
    _success_result,
    text_block,
    validate_tool_result,
)
from ._schemas import (
    _coerce_arguments,
    _normalize_schema,
    _validate_arguments,
)
from ._shared.process import BackgroundTaskRegistry
from ._shared.sandbox import SandboxPolicy

if TYPE_CHECKING:
    from ..skills.agent_catalog import AgentCatalog

AbortSignal = ToolAbortSignal
ToolHook = Callable[[str, dict[str, Any]], bool | str | Awaitable[bool | str] | None]
ToolHandlerFactory = Callable[["ToolRegistry"], ToolHandler]
RouterToolsSink = Callable[[list[str] | None], None]
RouterBrowserCatalogSink = Callable[[Any | None], None]
MemoryStoreSink = Callable[[str], None]

def _bind_handler(handler: ToolHandler, registry: ToolRegistry) -> ToolHandler:
    return partial(handler, registry)


def _validate_unique_tool_call_ids(tool_calls: Sequence[ToolCall]) -> None:
    call_ids = [tool_call.id for tool_call in tool_calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("duplicate tool call id in one execution batch")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    handler_factory: ToolHandlerFactory | None = None
    parallel_safe: bool = False
    validate_arguments: bool = True
    requires_approval: bool = True
    approval_denial_is_cancellation: bool = False
    # Argument that ``tool(pattern)`` approval rules match against (ZETA-86).
    approval_subject: str | None = None

    def schema(self) -> ToolSchema:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }


def _copy_definition(definition: ToolDefinition, registry: ToolRegistry | None = None) -> ToolDefinition:
    handler = (
        definition.handler_factory(registry)
        if registry is not None and definition.handler_factory is not None
        else definition.handler
    )
    return replace(definition, parameters=copy.deepcopy(definition.parameters), handler=handler)


class ToolRegistry:
    """One provider-neutral registry for built-in and custom tools."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        pre_execute_hook: ToolHook | None = None,
        hook: ToolHook | None = None,
        abort_signal: ToolAbortSignal | None = None,
        approval_policy: ApprovalPolicy | None = None,
        approval_store: ConversationStore | None = None,
        session_store: ConversationStore | None = None,
        max_output_chars: int = 10_000,
        memory_config: str | None = None,
        register_builtin: bool = True,
        enforce_approvals: bool = False,
        safety_tier: SafetyTier | None = None,
        browser_enabled: bool = False,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog | None = None,
    ) -> None:
        if enforce_approvals and approval_policy is None:
            raise ValueError("enforced approvals require a policy")
        self.enforce_approvals = enforce_approvals
        # Deliberately shared by session clones so child denials reach the run record.
        self.denied_tools: list[str] = []
        self.cwd = Path(os.path.abspath(os.fspath(Path(cwd).expanduser())))
        cwd_fd = -1
        try:
            cwd_fd = os.open(
                self.cwd,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            cwd_stat = os.fstat(cwd_fd)
        except OSError as exc:
            if cwd_fd >= 0:
                os.close(cwd_fd)
            raise ValueError(
                f"tool cwd is not a directory or is a symlink: {self.cwd}"
            ) from exc
        self._cwd_fd = cwd_fd
        self._cwd_identity = (cwd_stat.st_dev, cwd_stat.st_ino)
        self._cwd_finalizer = weakref.finalize(self, os.close, cwd_fd)
        self.policy = SandboxPolicy(self.cwd)
        if type(max_output_chars) is not int or max_output_chars < 1:
            raise ValueError("max_output_chars must be a positive integer")
        if pre_execute_hook is not None and hook is not None:
            raise ValueError("pass only one pre-execution hook")
        self.pre_execute_hook = pre_execute_hook or hook
        if abort_signal is None:
            self._abort_registry = AbortGenerationRegistry()
            self.abort_signal = self._abort_registry.new_generation()
        else:
            self.abort_signal = abort_signal
            self._abort_registry = abort_signal.registry
        self.approval_policy = approval_policy
        self.safety_tier = safety_tier
        self.browser_enabled = browser_enabled
        self._approval_gate = ApprovalGate(
            self.approval_policy, self.pre_execute_hook, self.safety_tier
        )
        if self.approval_policy is not None and approval_store is not None:
            self.approval_policy.bind_store(approval_store)
        self.max_output_chars = max_output_chars
        self.memory_config = memory_config
        self._session_store = session_store
        self._todo_store = session_store
        self._agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None = None
        self.router_tools_sink: RouterToolsSink | None = None
        self.router_recent_steps: list[str] | None = None
        self.router_browser_catalog_sink: RouterBrowserCatalogSink | None = None
        self.memory_store_sink: MemoryStoreSink | None = None
        self.background_tasks = BackgroundTaskRegistry(
            session_dir=session_store.session_dir if session_store is not None else None,
            directory_fd=session_store.directory_fd if session_store is not None else None,
        )
        self.bash_cwd = (
            session_store.bash_cwd if session_store is not None else str(self.cwd)
        )
        self._tools: dict[str, ToolDefinition] = {}
        self._browser_session: Any | None = None
        self._browser_session_factory: Callable[[ToolRegistry], Any] | None = None
        self._browser_session_closed = False
        self.browser_catalog: Any | None = None
        self.router_browser_catalog: Any | None = None
        self.browser_adapter_factory: Callable[[], Any] | None = None
        self.browser_goal: str | None = None
        self.skill_catalog = skill_catalog
        if agent_catalog is None:
            from ..skills.agent_catalog import discover_packaged_agents

            agent_catalog = discover_packaged_agents()
        self.agent_catalog = agent_catalog
        self._register_builtin = register_builtin
        if register_builtin:
            _register_discovered_tools(self)

    @property
    def schemas(self) -> list[ToolSchema]:
        return [definition.schema() for definition in self._tools.values()]

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return self.schemas

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(_copy_definition(definition) for definition in self._tools.values())

    @property
    def definitions_by_name(self) -> Mapping[str, ToolDefinition]:
        return {
            name: _copy_definition(definition)
            for name, definition in self._tools.items()
        }

    @property
    def registered_names(self) -> frozenset[str]:
        """Return the current tool names without copying definitions."""

        return frozenset(self._tools)

    @property
    def browser_session(self) -> Any | None:
        """Return the lazy browser session owned by this registry."""

        if (
            self._browser_session is None
            and self._browser_session_factory is not None
            and not self._browser_session_closed
        ):
            self._browser_session = self._browser_session_factory(self)
        return self._browser_session

    @property
    def browser_session_closed(self) -> bool:
        return self._browser_session_closed

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
        input_schema: Mapping[str, Any] | None = None,
        schema: Mapping[str, Any] | None = None,
        parallel_safe: bool = False,
        validate_arguments: bool = True,
        requires_approval: bool = True,
        approval_denial_is_cancellation: bool = False,
        handler_factory: ToolHandlerFactory | None = None,
        approval_subject: str | None = None,
    ) -> ToolDefinition:
        if type(name) is not str or not name:
            raise ValueError("tool name must be a nonempty string")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        supplied_schemas = [
            candidate
            for candidate in (parameters, input_schema, schema)
            if candidate is not None
        ]
        if len(supplied_schemas) > 1:
            raise ValueError("pass only one tool parameter schema")
        normalized = _normalize_schema(
            supplied_schemas[0] if supplied_schemas else None,
            validate_definition=validate_arguments,
        )
        properties = normalized.get("properties")
        if approval_subject is not None and (
            type(approval_subject) is not str
            or not approval_subject
            or (isinstance(properties, Mapping) and properties and approval_subject not in properties)
        ):
            raise ValueError(
                f"approval_subject {approval_subject!r} must name a parameter of tool {name!r}"
            )
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=normalized,
            handler=handler,
            handler_factory=handler_factory,
            parallel_safe=parallel_safe,
            validate_arguments=validate_arguments,
            requires_approval=requires_approval,
            approval_denial_is_cancellation=approval_denial_is_cancellation,
            approval_subject=approval_subject,
        )
        self._tools[name] = definition
        if self.approval_policy is not None:
            self.approval_policy.declare_subjects({name: approval_subject})
        return _copy_definition(definition)

    register_tool = register

    def register_session_tool(self, name: str, handler: ToolHandler, **kwargs: Any) -> ToolDefinition:
        return self.register(name, _bind_handler(handler, self), handler_factory=partial(_bind_handler, handler), **kwargs)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    @property
    def agent_runner(self) -> Callable[..., Awaitable[ToolHandlerResult]] | None:
        return self._agent_runner

    def set_agent_runner(
        self,
        runner: Callable[..., Awaitable[ToolHandlerResult]] | None,
    ) -> None:
        self._agent_runner = runner

    def clone_for_session(
        self,
        store: ConversationStore,
        *,
        exclude_names: set[str] | frozenset[str] = frozenset(),
    ) -> ToolRegistry:
        clone = copy.copy(self)
        clone._cwd_fd = os.dup(self._cwd_fd)
        clone._cwd_finalizer = weakref.finalize(clone, os.close, clone._cwd_fd)
        clone._tools = {
            name: _copy_definition(definition, clone)
            for name, definition in self._tools.items()
            if name not in exclude_names
        }
        clone._browser_session = None
        clone._browser_session_closed = False
        clone.browser_catalog = None
        clone.router_browser_catalog = None
        clone.router_browser_catalog_sink = None
        clone.browser_goal = None
        clone._session_store = store
        clone._todo_store = self._todo_store or store
        clone.background_tasks = BackgroundTaskRegistry(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
        )
        clone.bash_cwd = store.bash_cwd
        clone.abort_signal = clone._abort_registry.new_generation()
        if self.safety_tier is not None:
            clone.safety_tier = SafetyTier(
                cwd=clone.cwd,
                headless=self.safety_tier.headless,
                task_excerpt=self.safety_tier.task_excerpt,
                telemetry=self.safety_tier.telemetry,
            )
        clone._approval_gate = ApprovalGate(
            clone.approval_policy,
            clone.pre_execute_hook,
            clone.safety_tier,
        )
        clone._agent_runner = None
        clone.agent_catalog = self.agent_catalog
        return clone
    def abort(self) -> None:
        self.abort_signal.abort()

    def start_batch(self) -> None:
        """Rotate the active signal before a new tool batch."""
        self.abort_signal = self._abort_registry.new_generation()

    def start_user_turn(self, goal: str) -> None:
        """Reset browser-local routing state at a user-turn boundary."""

        self.browser_goal = goal
        if self._browser_session is not None:
            self._browser_session.reset_turn_state()

    def set_browser_catalog(self, catalog: Any | None) -> None:
        """Publish the current browser catalog to browser tools and the router."""

        self.browser_catalog = catalog
        self.router_browser_catalog = catalog
        if self.router_browser_catalog_sink is not None:
            self.router_browser_catalog_sink(catalog)

    def bind_approval_store(self, store: ConversationStore) -> None:
        if self.approval_policy is not None:
            self.approval_policy.bind_store(store)

    def bind_session_store(self, store: ConversationStore) -> None:
        self._session_store = store
        if self._todo_store is None:
            self._todo_store = store
        self.background_tasks.bind_session_dir(store.session_dir, store.directory_fd)
        self.bash_cwd = store.bash_cwd

    @property
    def session_store(self) -> ConversationStore:
        if self._session_store is None:
            raise ValueError("tool requires a bound session store")
        return self._session_store

    @property
    def todo_store(self) -> ConversationStore:
        if self._todo_store is None:
            raise ValueError("todo tool requires a bound session store")
        return self._todo_store

    async def close(self) -> None:
        """Stop session-owned browser and background processes."""

        background_error: BaseException | None = None
        try:
            await self.background_tasks.close()
        except BaseException as exc:  # noqa: BLE001 - close must continue cleanup
            background_error = exc
        browser_session = self._browser_session
        self._browser_session_closed = True
        self._browser_session = None
        if browser_session is not None:
            await browser_session.close()
        if background_error is not None:
            raise background_error

    def set_pre_execute_hook(self, hook: ToolHook | None) -> None:
        self.pre_execute_hook = hook
        self._approval_gate.hook = hook

    def set_router_tools_sink(self, sink: RouterToolsSink | None) -> None:
        self.router_tools_sink = sink

    def set_router_browser_catalog_sink(
        self, sink: RouterBrowserCatalogSink | None
    ) -> None:
        self.router_browser_catalog_sink = sink

    def set_router_recent_steps(self, steps: list[str] | None) -> None:
        self.router_recent_steps = steps

    def set_memory_store_sink(self, sink: MemoryStoreSink | None) -> None:
        self.memory_store_sink = sink

    def govern_tool_result(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        """Apply registry error governance to a legacy tool result."""

        governed = _apply_error_governance(_legacy_result(result), tool_call.name)
        return ToolResult(
            result.tool_call_id,
            flatten_tool_content(governed["content"]),
            is_error=governed["isError"],
            content_blocks=governed["content"],
            structured_content=governed["structuredContent"],
            is_canceled=result.is_canceled,
        )

    def update_bash_cwd(self, cwd: str) -> None:
        if self._session_store is not None:
            self._session_store.set_bash_cwd(cwd)
        self.bash_cwd = cwd

    def set_approval_policy(self, policy: ApprovalPolicy | None) -> None:
        if self.enforce_approvals and policy is None:
            raise ValueError("cannot remove an enforced approval policy")
        self.approval_policy = policy
        self._approval_gate.policy = policy
        if policy is not None:  # tell the policy which argument scopes each tool
            policy.declare_subjects(
                {name: tool.approval_subject for name, tool in self._tools.items()}
            )

    def set_safety_task_excerpt(self, task_excerpt: str) -> None:
        if self.safety_tier is not None:
            self.safety_tier.set_task_excerpt(task_excerpt)

    def set_safety_headless(self, headless: bool) -> None:
        if self.safety_tier is not None:
            self.safety_tier.set_headless(headless)

    def _safety_cwd(self, tool_name: str, arguments: dict[str, object]) -> str:
        if self.safety_tier is None:
            return str(self.cwd)
        if tool_name == "bash" and not arguments.get("cwd"):
            base_cwd = self.bash_cwd
        else:
            base_cwd = str(self.cwd)
        return self.safety_tier.command_cwd(arguments, base_cwd=base_cwd)

    def prepare_approval(self, tool_call: ToolCall) -> ApprovalRequest | None:
        if self.approval_policy is None:
            return None
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return None
        if not definition.requires_approval and not self.enforce_approvals:
            return None
        if definition.validate_arguments:
            try:
                _validate_arguments(tool_call.arguments, definition.parameters)
            except (AttributeError, KeyError, TypeError, ValueError):
                self._abort_approval(tool_call)
                return None
        return self.approval_policy.prepare(tool_call)

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        abort_signal: ToolAbortSignal | None = None,
        _scope_signal: ToolAbortSignal | None = None,
        _boundary_signal: ToolAbortSignal | None = None,
        _stream_sink: ToolStreamSink | None = None,
        _lifecycle_sink: ToolLifecycleSink | None = None,
        _persist_approval: bool = True,
        _log_path: str | Path | None = None,
        _background: bool = False,
        _capture_output: bool = False,
        _skip_approval: bool = False,
    ) -> StructuredToolResult:
        signal_state = abort_signal or self.abort_signal

        def finalize(result: StructuredToolResult) -> StructuredToolResult:
            normalized = _normalize_result(result, self.max_output_chars)
            return _apply_error_governance(normalized, tool_call.name)

        if _boundary_signal is not None and _signal_is_set(_boundary_signal):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, _boundary_signal, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        definition = self._tools.get(tool_call.name)
        if definition is None:
            self._abort_approval(tool_call)
            return finalize(
                _error_result(
                    f"unknown tool: {tool_call.name}",
                    kind="unknown_tool",
                )
            )
        try:
            arguments = (
                _validate_arguments(tool_call.arguments, definition.parameters)
                if definition.validate_arguments
                else _coerce_arguments(tool_call.arguments)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._abort_approval(tool_call)
            return finalize(
                _error_result(
                    f"invalid arguments: {exc}",
                    kind="invalid_arguments",
                )
            )
        if _signal_is_set(signal_state):
            signal_state, abort_result = self._arbitrate_abort(
                tool_call, signal_state, _scope_signal
            )
            if abort_result is not None:
                return finalize(abort_result)
        safety_cwd = (
            self._safety_cwd(tool_call.name, arguments)
            if self.safety_tier is not None
            and self.safety_tier.applies(tool_call.name)
            else None
        )
        gate_result, execution_signal = await self._approval_gate.run(
            tool_call,
            arguments,
            signal_state,
            lambda current: self._next_abort_generation(current, _scope_signal),
            _lifecycle_sink,
            skip_approval=(
                not self.enforce_approvals
                and (_skip_approval or not definition.requires_approval)
            ),
            persist_request=_persist_approval,
            safety_cwd=safety_cwd,
        )
        if gate_result is not None:
            denied = gate_result.content == "tool execution denied"
            if (
                self.enforce_approvals
                and denied
            ):
                self.denied_tools.append(tool_call.name)
            if definition.approval_denial_is_cancellation and denied:
                return finalize(_legacy_result(_canceled_result(tool_call.id)))
            return finalize(_legacy_result(gate_result))
        if _scope_signal is not None:
            execution_signal = _scope_signal
            if execution_signal.is_set():
                return finalize(_legacy_result(_canceled_result(tool_call.id)))
        if _lifecycle_sink is not None:
            _lifecycle_sink("execution_start")
        stream_publisher = (
            _ToolCallStreamPublisher(tool_call, execution_signal, _stream_sink)
            if _stream_sink is not None
            else None
        )
        execution_context = ToolExecutionContext(
            tool_call,
            self._agent_runner,
            _lifecycle_sink,
            self.router_tools_sink,
            self.router_recent_steps,
            self.memory_store_sink,
        )
        handler = bind_execution_context(definition.handler, execution_context)
        execution_arguments = build_execution_arguments(
            arguments,
            log_path=_log_path,
            background=_background,
            capture_output=_capture_output,
        )
        result = await run_handler_with_abort(
            handler,
            execution_arguments,
            execution_signal,
            stream_publisher,
            tool_call.id,
        )
        if isinstance(result, ToolResult):
            if result.tool_call_id != tool_call.id:
                normalized_result = _error_result(
                    f"tool result id mismatch: expected {tool_call.id}, "
                    f"got {result.tool_call_id}",
                    kind="invalid_result",
                )
            else:
                normalized_result = _legacy_result(result)
        elif isinstance(result, Mapping):
            try:
                normalized_result = validate_tool_result(result)
            except ValueError as exc:
                normalized_result = _error_result(
                    f"invalid tool handler result: {exc}",
                    kind="invalid_result",
                )
        elif isinstance(result, str):
            normalized_result = _success_result(text_block(result))
        else:
            normalized_result = _error_result(
                "invalid tool handler result: expected str or structured tool result",
                kind="invalid_result",
            )
        return finalize(normalized_result)

    def _abort_approval(self, tool_call: ToolCall) -> ApprovalDecision | None:
        if self.approval_policy is None:
            return None
        try:
            return self.approval_policy.abort_or_winner(tool_call.id)
        except RuntimeError:
            return None

    def _arbitrate_abort(
        self,
        tool_call: ToolCall,
        signal_state: ToolAbortSignal,
        scope_signal: ToolAbortSignal | None = None,
    ) -> tuple[ToolAbortSignal, StructuredToolResult | None]:
        winner = self._abort_approval(tool_call)
        if winner is ApprovalDecision.ALLOW:
            signal_state = self._next_abort_generation(signal_state, scope_signal)
            if _signal_is_set(signal_state):
                return signal_state, _legacy_result(_canceled_result(tool_call.id))
            return signal_state, None
        if winner is ApprovalDecision.DENY:
            return signal_state, _error_result(
                "tool execution denied", kind="denied"
            )
        return signal_state, _legacy_result(_canceled_result(tool_call.id))

    def _next_abort_generation(
        self,
        signal_state: ToolAbortSignal,
        scope_signal: ToolAbortSignal | None = None,
    ) -> ToolAbortSignal:
        if scope_signal is not None:
            return scope_signal
        if self.abort_signal is signal_state:
            self.abort_signal = self._abort_registry.new_generation()
        return self.abort_signal

    async def execute_many(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        abort_signal: ToolAbortSignal | None = None,
    ) -> list[StructuredToolResult]:
        """Execute calls with safe contiguous groups in parallel, preserving order."""

        _validate_unique_tool_call_ids(tool_calls)
        parent_signal = abort_signal or self.abort_signal
        boundary_signal = parent_signal if _signal_is_set(parent_signal) else None
        scope_signal = parent_signal
        if abort_signal is None:
            scope_signal = self._abort_registry.new_generation()
            self.abort_signal = scope_signal
        results: list[StructuredToolResult | None] = [None] * len(tool_calls)
        index = 0
        while index < len(tool_calls):
            definition = self._tools.get(tool_calls[index].name)
            if definition is None or not definition.parallel_safe:
                results[index] = await self.execute(
                    tool_calls[index],
                    abort_signal=scope_signal,
                    _scope_signal=scope_signal,
                    _boundary_signal=boundary_signal,
                )
                index += 1
                continue
            end = index + 1
            while end < len(tool_calls):
                next_definition = self._tools.get(tool_calls[end].name)
                if next_definition is None or not next_definition.parallel_safe:
                    break
                end += 1
            group = await asyncio.gather(
                *(
                    self.execute(
                        call,
                        abort_signal=scope_signal,
                        _scope_signal=scope_signal,
                        _boundary_signal=boundary_signal,
                    )
                    for call in tool_calls[index:end]
                )
            )
            results[index:end] = group
            index = end
        return [result for result in results if result is not None]

    def _path(self, raw_path: object) -> Path:
        return self.policy.resolve(raw_path).absolute

    def _open_cwd(self) -> int:
        try:
            cwd_fd = os.open(
                self.cwd,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise ValueError("session cwd was replaced") from exc
        try:
            cwd_stat = os.fstat(cwd_fd)
        except OSError as exc:
            os.close(cwd_fd)
            raise ValueError("session cwd was replaced") from exc
        if (cwd_stat.st_dev, cwd_stat.st_ino) != self._cwd_identity:
            os.close(cwd_fd)
            raise ValueError("session cwd was replaced")
        return cwd_fd
