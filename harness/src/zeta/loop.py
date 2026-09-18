"""The provider-neutral agent loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

import httpx

from .agent_background import (
    BackgroundAgentOwner,
    recover_agent_children,
)
from .agent_budget import (
    MAX_AGENT_DEPTH,
    AgentTree,
    consume_turn,
)
from .agent_receipt import (
    TerminalState,
    finalize_agent_results,
    terminal_state,
)
from .agent_runner import run_agent_tool
from .core.abort import AbortSignal as ToolAbortSignal
from .core.approval import ApprovalPolicy
from .core.context import ContextAssembler, MEMORY_INJECTION_PREFIX
from .core.hooks import HookManager
from .core.store import ConversationStore
from .core.tool_dispatch import dispatch_tool_calls
from .mcp import (
    MCPConfigError,
    MCPMount,
    home_config_path,
    load_mcp_config_overlay,
    mount_mcp_servers,
    project_config_path,
    tool_prefix,
)
from .mcp.commands import (
    MCP_USAGE,
    MCPCommandError,
    add_and_mount,
    parse_add_command,
    remove_and_unshadow,
    render_mcp_status,
    run_mcp_auth,
    run_mcp_resource_attach,
    run_mcp_resources_list,
)
from .mcp.prompt_commands import SlashModelInput
from .prompts import load_identity
from .providers.jev import auto_route, memory_gate
from .skills import SkillCatalog
from .skills.agent_catalog import AgentCatalog
from .tools import ToolHandler, ToolRegistry, ToolStreamPublisher
from .tools.agent import MAX_AGENT_RESULT_BYTES, agent_result
from .tools.agent_presets import (
    compose_system_prompt,
)
from .tools.loop_setup import select_tool_registry
from .tools.memory import _memory_search
from .tools.plan_mode import (
    PLAN_MODE_PREAMBLE,
    PLAN_MODE_TOOLS,
)
from .tools.registry import (
    ToolExecutionContext,
    _validate_unique_tool_call_ids,
    validate_tool_result,
)
from .tools.route import ROUTE_TOPK_CONFIDENCE, build_catalog
from .types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    flatten_tool_content,
)

TaskResult = TypeVar("TaskResult")
MAX_ERROR_MESSAGE = 400
NEEDS_TOOL_GATE = 0.35
"""Auto-route Noul gate; thresholds do not transfer (Jev jaggedness section 8)."""
# TODO: calibrate this Noul threshold with needs-tool data.
MEMORY_INJECTION_GATE = 0.6
"""Memory injection Noul cutoff; calibrate with injection relevance data."""
# TODO: calibrate this Noul threshold with memory-help data.
MEMORY_INJECTION_TOP_K = 2
MEMORY_INJECTION_EXCERPT_CHARS = 600
MEMORY_INJECTION_TOTAL_CHARS = 1500
_logger = logging.getLogger(__name__)


async def _close_completion(
    completion: AsyncIterator[StreamEvent] | None,
) -> BaseException | None:
    if completion is None:
        return None
    close = getattr(completion, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except BaseException as exc:  # noqa: BLE001 - preserve close errors
        return exc
    return None


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _error_info(error: BaseException, *, provider_error: bool = False) -> ErrorInfo:
    """Normalize provider and transport failures for the transcript."""

    code = getattr(error, "code", None)
    if type(code) is not str or not code:
        if isinstance(error, TimeoutError):
            code = "timeout"
        elif isinstance(error, httpx.TransportError):
            code = "transport_error"
        else:
            cause = error.__cause__
            while cause is not None:
                if isinstance(cause, httpx.TransportError):
                    code = "transport_error"
                    break
                cause = cause.__cause__
            else:
                code = "backend_error"
    try:
        message = str(error).strip()
    except Exception:  # noqa: BLE001 - malformed exception text is recoverable
        message = ""
    if not message:
        message = type(error).__name__
    if len(message) > MAX_ERROR_MESSAGE:
        message = f"{message[: MAX_ERROR_MESSAGE - 3]}..."
    return ErrorInfo(
        code, message, status_code=getattr(error, "status_code", None),
        provider_error=provider_error,
    )


def _validated_tool_result(result: object, expected_id: str) -> ToolResult:
    if isinstance(result, ToolResult):
        if type(result.tool_call_id) is not str or not result.tool_call_id:
            return ToolResult(expected_id, "invalid tool result: call id", True)
        if type(result.content) is not str:
            return ToolResult(expected_id, "invalid tool result: content", True)
        if type(result.is_error) is not bool:
            return ToolResult(expected_id, "invalid tool result: is_error", True)
        if result.tool_call_id != expected_id:
            return ToolResult(
                expected_id,
                f"tool result id mismatch: expected {expected_id}, got {result.tool_call_id}",
                is_error=True,
            )
        return result
    if not isinstance(result, Mapping):
        return ToolResult(
            expected_id,
            "invalid tool result: expected structured result",
            True,
        )
    try:
        structured_result = validate_tool_result(result)
    except ValueError as exc:
        return ToolResult(expected_id, f"invalid tool result: {exc}", True)
    return ToolResult(
            expected_id,
            flatten_tool_content(structured_result["content"]),
            structured_result["isError"],
            content_blocks=structured_result["content"],
            structured_content=structured_result["structuredContent"],
            is_canceled=structured_result.get("isCanceled", False),
    )


class AgentLoop:
    def __init__(
        self,
        backend: CompletionBackend,
        store: ConversationStore,
        *,
        tools: Mapping[str, ToolHandler] | ToolRegistry | None = None,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog | None = None,
        registry: ToolRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
        tool_schemas: Sequence[ToolSchema] | None = None,
        max_turns: int = 150,
        context_assembler: ContextAssembler | None = None,
        system_prompt: str | Message | None = None,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        on_completion_success: Callable[[], None] | None = None,
        on_plan_mode_change: Callable[[bool], None] | None = None,
        hooks: HookManager | None = None,
        skip_mcp_mount: bool = False,
        agent_depth: int = 0,
        agent_instance_id: str | None = None,
        agent_turn_budget: int | None = None,
        agent_tree: AgentTree | None = None,
        background_owner: BackgroundAgentOwner | None = None,
        router_mode: bool = True,
        router_style: str = "auto",
        jev_compaction: bool = True,
        memory_injection: bool = False,
    ) -> None:
        if type(agent_depth) is not int or not 0 <= agent_depth <= MAX_AGENT_DEPTH:
            raise ValueError(f"agent depth must be between 0 and {MAX_AGENT_DEPTH}")
        self.backend = backend
        self.store = store
        self.agent_depth = agent_depth
        self.agent_instance_id = agent_instance_id
        if agent_turn_budget is not None and agent_tree is not None:
            raise ValueError("pass only one agent turn budget")
        if agent_turn_budget is not None and (
            type(agent_turn_budget) is not int or agent_turn_budget < 1
        ):
            raise ValueError("agent turn budget must be a positive integer")
        self._agent_turn_budget = agent_turn_budget
        self._agent_tree = agent_tree
        self._background_owner = background_owner or BackgroundAgentOwner(store)
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_child_stores: dict[str, ConversationStore] = {}
        self._agent_child_turns: dict[str, int] = {}
        self._agent_child_types: dict[str, str] = {}
        self._background_child_cancellers: dict[str, Callable[[], None]] = {}
        self._background_child_watchers: dict[str, asyncio.Task[Any]] = {}
        self._background_event_sink: Callable[[StreamEvent], None] | None = None
        self.router_mode = router_mode
        if router_style not in {"tool", "auto"}:
            raise ValueError("router style must be 'tool' or 'auto'")
        self.router_style = router_style
        self.memory_injection = memory_injection
        self._routed_tools: list[str] = []
        self._router_fail_open = False
        self._router_recent_steps: list[str] = []
        self._router_batch_has_route = False
        self._router_batch_allowed_tools: set[str] | None = None
        self._router_auto_allowed_tools: set[str] = set()
        self._router_auto_fail_open = False
        self._auto_invoke_schema: ToolSchema = {
            "name": "invoke",
            "description": "Invoke the routed tool with the supplied arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["tool", "args"],
                "additionalProperties": False,
            },
        }
        self._auto_tool_surface = [self._auto_invoke_schema]
        self.unrouted_attempts = 0
        self._mcp_notice_sink: Callable[[str], None] | None = None
        self._mcp_prompt_refresh: Callable[[MCPMount], None] | None = None
        self._activated = False
        recover_agent_children(self)
        self.tool_registry = select_tool_registry(
            store,
            tools=tools,
            registry=registry,
            skill_catalog=skill_catalog,
            agent_catalog=agent_catalog,
            tool_schemas=tool_schemas,
        )
        self._mcp_mount: MCPMount | None = None
        self._mcp_mount_attempted = skip_mcp_mount
        self._mcp_mount_task: asyncio.Task[None] | None = None
        self._mcp_home_hint: str | None = None
        self._mcp_project_dir_value: Path | None = (
            Path(self.store.cwd).expanduser().resolve()
        )
        self._mcp_config_error: str | None = None
        self._mcp_schema_names: set[str] = set()
        self._provided_tool_schemas = tool_schemas is not None
        self.tool_registry.bind_session_store(store)
        self.tool_registry.set_router_tools_sink(self._record_routed_tools)
        self.tool_registry.set_router_recent_steps(self._router_recent_steps)
        self.agent_catalog = self.tool_registry.agent_catalog
        if (
            approval_policy is not None
            and self.tool_registry.approval_policy is not None
            and self.tool_registry.approval_policy is not approval_policy
        ):
            raise ValueError("pass only one approval policy")
        if approval_policy is not None:
            self.tool_registry.set_approval_policy(approval_policy)
        if self.tool_registry.approval_policy is not None:
            self.tool_registry.bind_approval_store(store)
        self.tool_schemas = list(
            tool_schemas if tool_schemas is not None else self.tool_registry.schemas
        )
        self.max_turns = max_turns
        if system_prompt is None:
            system_prompt = load_identity(catalog=self.tool_registry.skill_catalog)
        self.context_assembler = context_assembler or ContextAssembler(
            store,
            token_budget=token_budget,
            retained_tail=retained_tail,
            system_prompt=system_prompt,
            backend=backend,
            on_completion_success=on_completion_success,
            jev_compaction=jev_compaction,
        )
        self.on_completion_success = on_completion_success
        self._on_plan_mode_change = on_plan_mode_change
        self.hooks = hooks
        if self.hooks is not None:
            self.hooks.bind_session(store.session_id)
            if self.tool_registry.pre_execute_hook is None:
                self.tool_registry.set_pre_execute_hook(self.hooks.pre_tool)
        self._plan_mode = False
        self._plan_mode_prior_prompt: Message | None = None
        self._steering_queue: deque[Message] = deque()
        if "agent" in self.tool_registry.definitions_by_name:
            self.tool_registry.set_agent_runner(self._run_agent_tool)

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        """Restrict the assistant to read-only tools, or lift the restriction.

        Both edges rewrite the system prompt and the advertised tools, which
        Anthropic caches as one prefix, so each toggle costs a cache miss. That
        is fine for an occasional mode change and is why nothing flips this
        per turn.
        """

        if enabled == self._plan_mode:
            return
        assembler = self.context_assembler
        if enabled:
            self._plan_mode_prior_prompt = assembler.system_prompt
            composed = compose_system_prompt(
                assembler.system_prompt, PLAN_MODE_PREAMBLE
            )
            assert isinstance(composed, Message)
            assembler.system_prompt = composed
        elif self._plan_mode_prior_prompt is not None:
            assembler.system_prompt = self._plan_mode_prior_prompt
            self._plan_mode_prior_prompt = None
        self._plan_mode = enabled
        if self._on_plan_mode_change is not None:
            self._on_plan_mode_change(enabled)

    def plan_mode_allows(self, tool_name: str) -> bool:
        """Check the current plan-mode allowlist at dispatch time."""

        return not self._plan_mode or tool_name in PLAN_MODE_TOOLS | {"agent", "route"}

    @property
    def background_work_descriptions(self) -> tuple[str, ...]:
        process_work = tuple(
            record.command
            for record in self.tool_registry.background_tasks.records
            if record.running
        )
        return self._background_owner.active_descriptions + process_work

    def _active_tool_schemas(self) -> list[ToolSchema]:
        """Return the schemas this turn advertises, honoring router and plan modes."""

        if not self.router_mode:
            active = [
                schema for schema in self.tool_schemas if schema.get("name") != "route"
            ]
        elif self.router_style == "auto":
            active = self._auto_tool_surface
        elif self._router_fail_open:
            active = list(self.tool_schemas)
        else:
            allowed_names = {"route", *self._routed_tools}
            active = [
                schema
                for schema in self.tool_schemas
                if schema.get("name") in allowed_names
            ]
        if self.router_style == "auto" and self.router_mode:
            return active
        if not self._plan_mode:
            return active
        allowed = PLAN_MODE_TOOLS | {"agent"}
        return [
            schema
            for schema in active
            if schema.get("name") in allowed | {"route"}
        ]

    def _record_routed_tools(self, tools: list[str] | None) -> None:
        if tools is None:
            self._routed_tools = []
            self._router_fail_open = True
            return
        self._routed_tools = list(dict.fromkeys(tools))
        self._router_fail_open = False

    def _router_start_batch(self, calls: Sequence[ToolCall]) -> None:
        if not self.router_mode:
            return
        if self.router_style == "auto":
            self._router_batch_has_route = False
            if self._router_auto_fail_open:
                self._router_batch_allowed_tools = {
                    name
                    for name in self.tool_registry.registered_names
                    if name not in {"route", "invoke"}
                }
            else:
                self._router_batch_allowed_tools = set(
                    self._router_auto_allowed_tools
                )
            return
        self._router_batch_has_route = any(call.name == "route" for call in calls)
        self._router_batch_allowed_tools = {
            schema["name"]
            for schema in self._active_tool_schemas()
            if isinstance(schema.get("name"), str)
        }
        if not self._router_batch_has_route:
            self._routed_tools = []
            self._router_fail_open = False

    def _router_end_batch(self) -> None:
        self._router_batch_has_route = False
        self._router_batch_allowed_tools = None

    def _router_before_tool_execution(self, tool_name: str) -> None:
        if (
            self.router_mode
            and tool_name != "route"
            and not self._router_batch_has_route
        ):
            self._routed_tools = []

    def _router_allows_tool(self, tool_name: str) -> bool:
        if not self.router_mode:
            return True
        if self.router_style == "auto":
            allowed = self._router_batch_allowed_tools
            if allowed is None:
                allowed = self._router_auto_allowed_tools
            return tool_name in allowed
        if tool_name == "route" or self._router_fail_open:
            return True
        allowed = self._router_batch_allowed_tools
        if allowed is None:
            allowed = {
                schema["name"]
                for schema in self._active_tool_schemas()
                if isinstance(schema.get("name"), str)
            }
        return tool_name in allowed

    def _router_rejection(self, tool_call: ToolCall) -> ToolResult | None:
        if self.router_mode and self.router_style == "auto" and tool_call.name == "invoke":
            result = ToolResult(
                tool_call.id,
                "invoke requires a routed tool name and args object",
                is_error=True,
                structured_content={"error_kind": "invalid_invoke"},
            )
            return self.tool_registry.govern_tool_result(tool_call, result)
        if self.router_mode and self.router_style == "auto":
            if self._router_allows_tool(tool_call.name):
                return None
            self.unrouted_attempts += 1
            _logger.warning("unrouted tool attempt: %s", tool_call.name)
            result = ToolResult(
                tool_call.id,
                "not available this turn — state what you need in text for the next turn: "
                + tool_call.name,
                is_error=True,
                structured_content={"error_kind": "unrouted_tool"},
            )
            return self.tool_registry.govern_tool_result(tool_call, result)
        if (
            not self.router_mode
            or self._router_allows_tool(tool_call.name)
            or tool_call.name not in self.tool_registry.registered_names
        ):
            return None
        self.unrouted_attempts += 1
        _logger.warning("unrouted tool attempt: %s", tool_call.name)
        message = (
            (
                "not available this turn — state what you need in text for the "
                "next turn: "
            )
            if self.router_style == "auto"
            else "not available this turn — describe your step to route first: "
        ) + tool_call.name
        result = ToolResult(
            tool_call.id,
            message,
            is_error=True,
            structured_content={"error_kind": "unrouted_tool"},
        )
        return self.tool_registry.govern_tool_result(tool_call, result)

    def _router_result(self, tool_name: str, result: ToolResult) -> None:
        if (
            self.router_mode
            and self.router_style == "tool"
            and tool_name == "route"
            and result.is_error
        ):
            self._record_routed_tools(None)

    def _auto_route_inputs(
        self, user_text: str
    ) -> tuple[str, str, list[dict[str, str]]]:
        messages = self.store.messages()
        tool_names = {
            block.tool_call.id: block.tool_call.name
            for message in messages
            for block in message.content
            if isinstance(block, ToolUseContent)
        }
        assistant = ""
        results: list[dict[str, str]] = []
        for message in reversed(messages):
            if not assistant and message.role is MessageRole.ASSISTANT:
                assistant = "".join(
                    block.text
                    for block in message.content
                    if isinstance(block, TextContent)
                )[:300]
            if message.tool_result is not None and len(results) < 2:
                results.append(
                    {
                        "tool": tool_names.get(message.tool_result.tool_call_id, ""),
                        "excerpt": message.tool_result.content[:200],
                    }
                )
            if assistant and len(results) >= 2:
                break
        results.reverse()
        return user_text[:500], assistant, results

    def _memory_query(self, user_text: str) -> str:
        task, last_assistant, _last_results = self._auto_route_inputs(user_text)
        return "\n".join(part for part in (task, last_assistant) if part)

    @staticmethod
    def _memory_key(value: Mapping[str, object]) -> tuple[str, tuple[str, ...]] | None:
        path = value.get("path")
        heading = value.get("heading", [])
        if not isinstance(path, str):
            return None
        if isinstance(heading, str):
            heading = [heading]
        if heading is None:
            heading = []
        if not isinstance(heading, list) or not all(
            isinstance(part, str) for part in heading
        ):
            return None
        return path, tuple(heading)

    def _known_memory_keys(self) -> set[tuple[str, tuple[str, ...]]]:
        keys: set[tuple[str, tuple[str, ...]]] = set()
        for message in self.store.messages():
            injected = message.metadata.get("memory_injection_items")
            if isinstance(injected, list):
                for item in injected:
                    if isinstance(item, Mapping):
                        key = self._memory_key(item)
                        if key is not None:
                            keys.add(key)
            result = message.tool_result
            structured = result.structured_content if result is not None else None
            if not isinstance(structured, Mapping):
                continue
            items = structured.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, Mapping):
                        key = self._memory_key(item)
                        if key is not None:
                            keys.add(key)
            key = self._memory_key(structured)
            if key is not None:
                keys.add(key)
        return keys

    async def _inject_memory(
        self, query: str, gate_score: float | None
    ) -> dict[str, object]:
        decision: dict[str, object] = {
            "gate_score": gate_score,
            "injected_count": 0,
            "chars": 0,
        }
        if (
            self.tool_registry.memory_config is None
            or gate_score is None
            or gate_score < MEMORY_INJECTION_GATE
        ):
            return decision
        try:
            result = await _memory_search(
                self.tool_registry,
                {"query": query},
                self.tool_registry.abort_signal,
            )
            if result.get("isError") is True:
                _logger.warning("memory injection search failed")
                return decision
            structured = result.get("structuredContent")
            items = structured.get("items") if isinstance(structured, Mapping) else None
            if not isinstance(items, list):
                return decision
            known = self._known_memory_keys()
            blocks: list[TextContent] = []
            injected_items: list[dict[str, object]] = []
            remaining = MEMORY_INJECTION_TOTAL_CHARS
            for raw in items:
                if len(blocks) >= MEMORY_INJECTION_TOP_K or remaining <= 0:
                    break
                if not isinstance(raw, Mapping):
                    continue
                key = self._memory_key(raw)
                excerpt = raw.get("excerpt")
                if key is None or not isinstance(excerpt, str) or not excerpt:
                    continue
                if key in known:
                    continue
                excerpt = excerpt[: min(MEMORY_INJECTION_EXCERPT_CHARS, remaining)]
                if not excerpt:
                    continue
                heading = " > ".join(key[1]) or "(document)"
                text = (
                    f"{MEMORY_INJECTION_PREFIX}\n"
                    f"path: {key[0]}\n"
                    f"heading: {heading}\n"
                    f"{excerpt}"
                )
                blocks.append(TextContent(text))
                injected_items.append({"path": key[0], "heading": list(key[1])})
                known.add(key)
                remaining -= len(excerpt)
            if not blocks:
                return decision
            if not self._persist_memory_blocks(blocks, injected_items):
                return decision
            decision["injected_count"] = len(blocks)
            decision["chars"] = MEMORY_INJECTION_TOTAL_CHARS - remaining
            return decision
        except Exception as exc:  # noqa: BLE001 - injection fails open
            _logger.warning("memory injection failed: %s", exc)
            return decision

    def _persist_memory_blocks(
        self, blocks: Sequence[TextContent], items: Sequence[Mapping[str, object]]
    ) -> bool:
        branch = self.store.replay()
        target_entry = next(
            (entry for entry in reversed(branch) if entry.type == "message"),
            None,
        )
        messages = self.store.messages()
        if target_entry is None or not messages:
            return False
        target = messages[-1]
        metadata = dict(target.metadata)
        previous = metadata.get("memory_injection_items", [])
        previous_items = list(previous) if isinstance(previous, list) else []
        metadata.update(
            {
                "memory_injection": True,
                "compaction_droppable": True,
                "memory_injection_items": [*previous_items, *items],
            }
        )
        self.store.append_message_revision(
            target_entry.id,
            replace(
                target,
                content=[*target.content, *blocks],
                metadata=metadata,
            ),
        )
        return True

    async def _prepare_user_memory(
        self, user_text: str
    ) -> tuple[dict[str, object], dict[str, int]]:
        if self.tool_registry.memory_config is None:
            return {"gate_score": None, "injected_count": 0, "chars": 0}, {}
        query = self._memory_query(user_text)
        try:
            result = await memory_gate(query)
        except Exception as exc:  # noqa: BLE001 - gate fails open
            _logger.warning("memory injection gate failed: %s", exc)
            return {"gate_score": None, "injected_count": 0, "chars": 0}, {}
        decision = await self._inject_memory(query, result.score)
        return decision, dict(result.usage)

    def _auto_catalog(self) -> dict[str, dict[str, object]]:
        return build_catalog(
            self.tool_registry.schemas,
            excluded_names={"route", "invoke"},
        )

    async def _prepare_auto_route(
        self, user_text: str
    ) -> tuple[list[ToolSchema], dict[str, object]]:
        task, last_assistant, last_results = self._auto_route_inputs(user_text)
        try:
            if self.memory_injection:
                result = await auto_route(
                    task,
                    last_assistant,
                    last_results,
                    self._auto_catalog(),
                    memory_injection=True,
                )
            else:
                result = await auto_route(
                    task,
                    last_assistant,
                    last_results,
                    self._auto_catalog(),
                )
        except Exception as exc:  # noqa: BLE001 - auto routing fails open
            full_catalog = [
                schema
                for schema in self.tool_registry.schemas
                if schema.get("name") not in {"route", "invoke"}
            ]
            self._router_auto_allowed_tools = {
                name
                for name in self.tool_registry.registered_names
                if name not in {"route", "invoke"}
            }
            self._router_auto_fail_open = True
            return full_catalog, {
                "error": str(exc),
                "advertised": sorted(self._router_auto_allowed_tools),
                "fail_open": True,
            }
        if result.needs_tool < NEEDS_TOOL_GATE:
            names: list[str] = []
        elif result.confidence >= ROUTE_TOPK_CONFIDENCE:
            names = [result.tool]
        else:
            names = [
                name
                for name, _probability in sorted(
                    result.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            ]
        available = {
            name
            for name in self.tool_registry.registered_names
            if name not in {"route", "invoke"}
        }
        names = [name for name in names if name in available]
        self._router_auto_allowed_tools = set(names)
        self._router_auto_fail_open = False
        schemas = [
            schema
            for schema in self.tool_registry.schemas
            if schema.get("name") in names
        ]
        routing_decision: dict[str, object] = {
            "tool": result.tool,
            "confidence": result.confidence,
            "needs_tool": result.needs_tool,
            "advertised": names,
            "direct_answer": not names and result.needs_tool < NEEDS_TOOL_GATE,
            "usage": dict(result.usage),
        }
        if result.call_confidence is not None:
            routing_decision["call_confidence"] = result.call_confidence
        if self.memory_injection:
            memory_decision = await self._inject_memory(
                "\n".join(part for part in (task, last_assistant) if part),
                result.memory_help,
            )
            routing_decision["memory_injection"] = memory_decision
        return schemas, routing_decision

    @staticmethod
    def _schema_text(schemas: Sequence[ToolSchema]) -> str:
        lines = ["routed tool schemas:"]
        for schema in schemas:
            name = schema.get("name", "")
            description = schema.get("description", "")
            parameters = schema.get("parameters", schema.get("input_schema", {}))
            lines.append(
                json.dumps(
                    {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return "\n".join(lines)

    def _persist_auto_schema_text(
        self,
        schemas: Sequence[ToolSchema],
        routing_decision: Mapping[str, object],
    ) -> None:
        branch = self.store.replay()
        target_entry = next(
            (entry for entry in reversed(branch) if entry.type == "message"),
            None,
        )
        if target_entry is None:
            return
        target = self.store.messages()[-1]
        text = (
            "no tool is needed this turn — answer directly"
            if routing_decision.get("direct_answer") is True
            else self._schema_text(schemas)
        )
        if any(
            isinstance(block, TextContent) and block.text == text
            for block in target.content
        ):
            return
        self.store.append_message_revision(
            target_entry.id,
            replace(target, content=[*target.content, TextContent(text)]),
        )

    def _expand_auto_invoke(self, call: ToolCall) -> ToolCall:
        if (
            not self.router_mode
            or self.router_style != "auto"
            or call.name != "invoke"
        ):
            return call
        tool = call.arguments.get("tool")
        args = call.arguments.get("args")
        if isinstance(tool, str) and isinstance(args, dict):
            return ToolCall(call.id, tool, args)
        return call

    def set_model(self, model: str) -> None:
        """Set the model used by subsequent provider completions."""

        if not model.strip():
            raise ValueError("model must be a nonempty name")
        if hasattr(self.backend, "model"):
            self.backend.model = model
        else:
            self._model = model

    def abort(self) -> None:
        """Signal the active tool batch before the caller cancels the turn."""

        self.tool_registry.abort()
        self._background_owner.cancel_all()
        self._steering_queue.clear()

    def steer(self, message: Message) -> None:
        """Queue a user message for injection at the next tool boundary.

        The running ``_run_turn`` drains this queue before the next provider
        call, so the message never lands between a tool_call and its
        tool_result. Callers must pass a durable USER-role message.
        """

        if message.role is not MessageRole.USER:
            raise ValueError("steering message must have the user role")
        self._steering_queue.append(message)

    @property
    def has_pending_steering(self) -> bool:
        return bool(self._steering_queue)

    def clear_pending_steering(self) -> None:
        self._steering_queue.clear()

    def set_background_event_sink(
        self, sink: Callable[[StreamEvent], None] | None
    ) -> None:
        """Set the sink for progress from children that outlive their turn."""

        self._background_event_sink = sink

    def set_mcp_notice_sink(self, sink: Callable[[str], None] | None) -> None:
        """Set the sink for MCP mount notices."""

        self._mcp_notice_sink = sink

    def set_mcp_prompt_refresh(
        self, callback: Callable[[MCPMount], None] | None
    ) -> None:
        """Set the owner callback for live MCP prompt commands."""

        self._mcp_prompt_refresh = callback
        if callback is not None and self._mcp_mount is not None:
            callback(self._mcp_mount)

    @property
    def mcp_summary(self) -> str:
        if self._mcp_mount is None:
            return "mcp: 0 mounted, 0 failed"
        return self._mcp_mount.summary

    async def slash_mcp(self, args: str) -> str | SlashModelInput:
        """Show MCP state, reconnect, add, remove, authorize, or attach."""

        await self._ensure_mcp_servers()
        mount = self._mcp_mount
        try:
            parts = shlex.split(args)
        except ValueError as exc:
            return f"mcp error: {exc}"
        if self._mcp_config_error is not None:
            return f"mcp error: {self._mcp_config_error}"
        if mount is None:
            return "mcp: no configured servers"
        if not parts:
            return render_mcp_status(mount, home=self._mcp_home_hint)
        verb = parts[0]
        try:
            if verb == "reconnect":
                if len(parts) != 2:
                    return MCP_USAGE
                await mount.reconnect(parts[1], notice_sink=self._mcp_notice_sink)
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "add":
                await add_and_mount(
                    mount,
                    parse_add_command(parts[1:]),
                    target=self._mcp_add_target(),
                    notice_sink=self._mcp_notice_sink,
                )
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "remove":
                if len(parts) != 2:
                    return MCP_USAGE
                await remove_and_unshadow(
                    mount,
                    parts[1],
                    home_path=self._mcp_home_path(),
                    load_home=lambda: load_mcp_config_overlay(
                        home=self._mcp_home_hint, project_dir=None
                    ).configured_servers,
                    notice_sink=self._mcp_notice_sink,
                )
                return render_mcp_status(mount, home=self._mcp_home_hint)
            if verb == "auth":
                if len(parts) != 2:
                    return MCP_USAGE
                return await run_mcp_auth(
                    mount,
                    parts[1],
                    home=self._mcp_home_hint,
                    notice_sink=self._mcp_notice_sink,
                )
            if verb == "resources":
                if len(parts) == 2:
                    return await run_mcp_resources_list(mount, parts[1])
                if len(parts) == 3:
                    return await run_mcp_resource_attach(
                        mount, parts[1], parts[2]
                    )
                return MCP_USAGE
        except (MCPCommandError, ValueError) as exc:
            return f"mcp error: {exc}"
        return MCP_USAGE

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        """Resolve one mounted MCP prompt for the next model turn."""

        await self._ensure_mcp_servers()
        if self._mcp_mount is None:
            raise RuntimeError("MCP mount is unavailable")
        return await self._mcp_mount.get_prompt(name, arguments)

    def _mcp_add_target(self) -> Path:
        project_dir = self._mcp_project_dir_value
        if project_dir is not None:
            return project_config_path(project_dir)
        return home_config_path(self._mcp_home_hint)

    def _mcp_home_path(self) -> Path:
        override = os.environ.get("ZETA_MCP_CONFIG")
        if override:
            return Path(override).expanduser().resolve()
        return home_config_path(self._mcp_home_hint).resolve()

    @property
    def background_children_running(self) -> bool:
        return self._background_owner.running or bool(self._background_child_cancellers)

    def _publish_background_event(self, event: StreamEvent) -> None:
        if self._background_event_sink is not None:
            self._background_event_sink(event)

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, TaskResult],
    ) -> asyncio.Task[TaskResult]:
        task = asyncio.create_task(coroutine)
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

    def _child_result_payload(
        self,
        tool_call_id: str,
        content: str,
        *,
        state: TerminalState | None = None,
        error: bool | None = None,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
        status: str | None = None,
        description: str | None = None,
        depth: int | None = None,
        budget_exhausted: bool = False,
        stats: dict[str, object] | None = None,
        include_stats: bool = True,
        canceled: bool = False,
    ) -> dict[str, object]:
        child_store = self._agent_child_stores.get(tool_call_id)
        path = (
            child_session_path
            if child_session_path is not None
            else str(child_store.session_dir)
            if child_store is not None
            else ""
        )
        turns = (
            turns_used
            if turns_used is not None
            else self._agent_child_turns.get(tool_call_id, 0)
        )
        if child_instance_id is None and child_store is not None:
            child_instance_id = child_store.agent_handle()
        result_status = (
            "running"
            if status == "running"
            else state
            or terminal_state(
                error=error is True,
                canceled=canceled,
                status=status,
            )
        )
        return agent_result(
            content,
            tool_call_id=tool_call_id,
            error=result_status == "failed",
            turns_used=turns,
            child_session_path=path,
            agent_type=agent_type if isinstance(agent_type, str) else None,
            status=status if status == "running" else None,
            child_instance_id=child_instance_id,
            description=description,
            depth=depth,
            budget_exhausted=budget_exhausted,
            stats=stats,
            include_stats=include_stats,
            canceled=result_status == "canceled",
            max_bytes=getattr(
                getattr(self, "tool_registry", None),
                "max_output_chars",
                MAX_AGENT_RESULT_BYTES,
            ),
        )

    def _canceled_agent_result(
        self,
        tool_call_id: str,
        *,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
    ) -> ToolResult:
        return _validated_tool_result(
            self._child_result_payload(
                tool_call_id,
                "tool execution canceled",
                state="canceled",
                child_session_path=child_session_path,
                turns_used=turns_used,
                agent_type=agent_type,
                child_instance_id=child_instance_id,
            ),
            tool_call_id,
        )

    async def _run_agent_tool(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any],
        abort_signal: ToolAbortSignal,
        publisher: ToolStreamPublisher | None,
        execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, object]:
        return await run_agent_tool(
            self,
            tool_call,
            arguments,
            abort_signal,
            publisher,
            validate_result=_validated_tool_result,
            error_message=lambda exc: _error_info(exc).message,
            execution_context=execution_context,
        )

    def prepare_resume_pending_tool(self, request_id: str) -> bool:
        """Reserve the abort generation before resuming an approved tool."""

        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return False
        if self._existing_tool_result(state[0].id) is not None:
            return False
        self.tool_registry.start_batch()
        return True

    def run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._run_turn(
            user_text,
            user_message=user_message,
            persist_user_message=persist_user_message,
            abort_signal=abort_signal,
        )

    async def close(self, *, cancel_background: bool = True) -> None:
        """Close session-owned transports and background processes."""

        try:
            if cancel_background and self.agent_depth == 0:
                self._background_owner.cancel_all()
            elif cancel_background:
                for cancel in tuple(self._background_child_cancellers.values()):
                    cancel()
            watchers = tuple(self._background_child_watchers.values())
            if cancel_background and watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
            if cancel_background and self.agent_depth == 0:
                await self._background_owner.wait()
            tracked_tasks = tuple(
                task
                for task in self._tracked_tasks
                if cancel_background or task not in self._background_child_watchers.values()
            )
            for task in tracked_tasks:
                task.cancel()
            await asyncio.gather(*tracked_tasks, return_exceptions=True)
            if self.hooks is not None:
                self.hooks.stop()
                await self.hooks.close()
            if self._mcp_mount is not None:
                await self._mcp_mount.close()
                self._mcp_mount = None
            if self._mcp_mount_task is not None and not self._mcp_mount_task.done():
                self._mcp_mount_task.cancel()
                await asyncio.gather(self._mcp_mount_task, return_exceptions=True)
        finally:
            try:
                await self.tool_registry.background_tasks.close()
            finally:
                if self.agent_depth == 0:
                    self._background_owner.store_leases.close()

    def session_start(self) -> None:
        if self.hooks is not None:
            self.hooks.session_start()

    async def activate(self) -> None:
        """Run frontend startup hooks after the frontend installs its sinks."""

        if self._activated:
            return
        self._activated = True
        self.session_start()

    async def _ensure_mcp_servers(self) -> None:
        if self._mcp_mount_attempted:
            return
        if self._mcp_mount_task is None:
            self._mcp_mount_task = asyncio.create_task(self._mount_mcp_servers())
        await asyncio.shield(self._mcp_mount_task)

    async def _mount_mcp_servers(self) -> None:
        try:
            config = load_mcp_config_overlay(
                home=self._mcp_home_hint,
                project_dir=self._mcp_project_dir_value,
            )
            self._mcp_mount = await mount_mcp_servers(
                self.tool_registry,
                config,
                notice_sink=self._mcp_notice_sink,
                home=self._mcp_home_hint,
            )
        except MCPConfigError as exc:
            self._mcp_config_error = str(exc)
            self._mcp_mount = MCPMount(
                self.tool_registry, {}, {}, home=self._mcp_home_hint
            )
        self._mcp_mount.set_schema_refresh(self._refresh_mcp_tool_schemas)
        if self._mcp_prompt_refresh is not None:
            self._mcp_mount.set_prompt_refresh(self._mcp_prompt_refresh)
        self._mcp_mount_attempted = True

    def attach_mcp_mount(self, mount: MCPMount) -> None:
        """Adopt an explicitly selected mount without loading project configuration."""

        if self._mcp_mount is not None:
            raise ValueError("MCP mount already attached")
        self._mcp_mount = mount
        self._mcp_mount_attempted = True
        mount.set_schema_refresh(self._refresh_mcp_tool_schemas)

    def set_mcp_scope(
        self,
        *,
        home: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        """Set the home + project scope this loop uses for MCP config files."""

        self._mcp_home_hint = None if home is None else str(home)
        self._mcp_project_dir_value = (
            None if project_dir is None else Path(project_dir).expanduser().resolve()
        )

    def _refresh_mcp_tool_schemas(self, mount: MCPMount | None = None) -> None:
        mount = mount or self._mcp_mount
        if mount is None:
            return
        mcp_prefixes = tuple(tool_prefix(name) for name in mount.configs)
        current_mcp = [
            schema
            for schema in self.tool_registry.schemas
            if isinstance(schema.get("name"), str)
            and schema["name"].startswith(mcp_prefixes)
        ]
        current_names = {
            schema["name"]
            for schema in current_mcp
            if isinstance(schema.get("name"), str)
        }
        if not self._provided_tool_schemas:
            self.tool_schemas = list(self.tool_registry.schemas)
            self._mcp_schema_names = current_names
            return
        names_to_replace = self._mcp_schema_names | current_names
        self.tool_schemas = [
            schema
            for schema in self.tool_schemas
            if not (
                isinstance(schema.get("name"), str)
                and schema["name"] in names_to_replace
            )
        ] + current_mcp
        self._mcp_schema_names = current_names

    async def ensure_mcp_servers(self) -> None:
        """Connect MCP servers before a direct tool resume."""

        await self._ensure_mcp_servers()

    async def resume_pending_tool(
        self,
        request_id: str,
        *,
        prepared: bool = False,
        event_sink: Callable[[StreamEvent], None] | None = None,
    ) -> ToolResult | None:
        """Finish a durable approval request before starting another turn."""

        await self._ensure_mcp_servers()
        state = self.store.approval_states().get(request_id)
        if state is None or state[1] is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        if not self.plan_mode_allows(tool_call.name):
            result = ToolResult(
                tool_call.id,
                f"tool execution denied in plan mode: {tool_call.name} is not allowed",
                is_error=True,
            )
            return self._finalize_tool_results([tool_call], [result])[0]
        if not prepared:
            self.tool_registry.start_batch()
        abort_signal = self.tool_registry.abort_signal

        def lifecycle(kind: str) -> None:
            if event_sink is None:
                return
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is not None:
                event_sink(StreamEvent(event_type, tool_call=tool_call))

        try:
            result = await self.tool_registry.execute(
                tool_call,
                abort_signal=abort_signal,
                _scope_signal=abort_signal,
                _lifecycle_sink=lifecycle,
            )
        except asyncio.CancelledError:
            result = self.finalize_canceled(request_id)
            if event_sink is not None and result is not None:
                event_sink(
                    StreamEvent(
                        StreamEventType.TOOL_EXECUTION_END,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                )
            raise
        except Exception as exc:  # noqa: BLE001 - report execution failures
            result = ToolResult(tool_call.id, str(exc), is_error=True)
        result = _validated_tool_result(result, tool_call.id)
        if result.is_canceled:
            result = self.finalize_canceled(request_id)
        else:
            result = self._finalize_tool_results([tool_call], [result])[0]
        if event_sink is not None and result is not None:
            event_sink(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=tool_call,
                    tool_result=result,
                )
            )
        return result

    def finalize_canceled(self, request_id: str) -> ToolResult | None:
        """Persist one canceled result for a durable approval request."""

        state = self.store.approval_states().get(request_id)
        if state is None:
            return None
        tool_call = state[0]
        existing = self._existing_tool_result(tool_call.id)
        if existing is not None:
            return existing
        return self._finalize_tool_results([tool_call], [None])[0]

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None:
        for message in reversed(self.store.messages()):
            result = message.tool_result
            if result is not None and result.tool_call_id == tool_call_id:
                return result
        return None

    async def _run_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        abort_signal: ToolAbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._routed_tools = []
        self._router_fail_open = False
        self._router_recent_steps.clear()
        self._router_auto_allowed_tools = set()
        self._router_auto_fail_open = False
        if self.hooks is not None:
            self.hooks.user_prompt_submit(user_text)
        if user_message is None:
            user_message = Message(MessageRole.USER, [TextContent(user_text)])
        elif user_message.role is not MessageRole.USER:
            raise ValueError("user_message must have the user role")
        if persist_user_message:
            self.store.append_message(user_message)
        elif user_message not in self.store.messages():
            raise ValueError("cannot reuse a user message that is not persisted")
        setup_error: ErrorInfo | None = None
        try:
            await self._ensure_mcp_servers()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report setup failures
            setup_error = _error_info(exc)

        for notification in self.store.agent_notifications():
            yield StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"notification_id": notification.id, **notification.data},
            )
            self.store.acknowledge_agent_notification(notification.id)

        if setup_error is not None:
            self._persist_partial_with_cancelled_tools([], None, failure=setup_error)
            yield StreamEvent(StreamEventType.AGENT_START)
            yield StreamEvent(StreamEventType.ERROR, error=setup_error)
            yield StreamEvent(StreamEventType.AGENT_END)
            return
        yield StreamEvent(StreamEventType.AGENT_START)
        if self.memory_injection and not (
            self.router_mode and self.router_style == "auto"
        ):
            memory_decision, memory_usage = await self._prepare_user_memory(user_text)
            yield StreamEvent(
                StreamEventType.USAGE,
                data={
                    "service": "jev",
                    "usage": memory_usage,
                    "memory_injection": memory_decision,
                },
            )

        for turn_number in range(1, self.max_turns + 1):
            # Mid-turn steering drains here — after the previous iteration's
            # dispatch_tool_calls persisted every tool_result, and before the
            # next provider call. The invariant "never split a tool_call from
            # its tool_result" holds because this point is strictly between
            # complete tool batches.
            while self._steering_queue:
                steering = self._steering_queue.popleft()
                self.store.append_message(steering)
            if (
                self.agent_depth
                and (
                    error := consume_turn(
                        self._agent_tree.budget
                        if self._agent_tree is not None
                        else None
                    )
                )
                is not None
            ):
                yield StreamEvent(StreamEventType.ERROR, error=error)
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            self.tool_registry.start_batch()
            turn_abort_signal = abort_signal or self.tool_registry.abort_signal
            yield StreamEvent(
                StreamEventType.TURN_START,
                data={"turn": turn_number},
            )
            partial_blocks: list[ContentBlock] = []
            assistant_message: Message | None = None
            completion: AsyncIterator[StreamEvent] | None = None
            completion_succeeded = False
            provider_error: ErrorInfo | None = None
            try:
                routing_decision: dict[str, object] = {}
                if self.router_mode and self.router_style == "auto":
                    routed_schemas, routing_decision = await self._prepare_auto_route(
                        user_text
                    )
                    self._persist_auto_schema_text(
                        routed_schemas, routing_decision
                    )
                    usage = routing_decision.get("usage")
                    yield StreamEvent(
                        StreamEventType.USAGE,
                        data={
                            "service": "jev",
                            "usage": dict(usage) if isinstance(usage, Mapping) else {},
                            "routing_decision": routing_decision,
                        },
                    )
                if self.context_assembler.needs_compaction():
                    yield StreamEvent(
                        StreamEventType.COMPACTION_START,
                        data={"turn": turn_number},
                    )
                context_messages = await self.context_assembler.assemble(
                    backend=self.backend
                )
                context = self.context_assembler.last_context
                if context is not None and context.compacted:
                    compaction_data = dict(
                        self.context_assembler.last_compaction_data
                    )
                    triage_data = compaction_data.get("jev_triage")
                    if isinstance(triage_data, Mapping):
                        triage_data = dict(triage_data)
                        usage = triage_data.pop("usage", None)
                        compaction_data["jev_triage"] = triage_data
                        if isinstance(usage, Mapping) and usage:
                            yield StreamEvent(
                                StreamEventType.USAGE,
                                data={"service": "jev", "usage": dict(usage)},
                            )
                    yield StreamEvent(
                        StreamEventType.COMPACTION_END,
                        data={
                            "turn": turn_number,
                            "token_count": context.token_count,
                            **compaction_data,
                        },
                    )
                completion = self.backend.complete(
                    context_messages, self._active_tool_schemas()
                )
                async for event in completion:
                    self.context_assembler.observe_event(event)
                    if event.type is StreamEventType.ERROR:
                        provider_error = (
                            replace(event.error, provider_error=True)
                            if isinstance(event.error, ErrorInfo)
                            else ErrorInfo(
                                "backend_error",
                                "provider emitted an invalid error event",
                            )
                        )
                        yield StreamEvent(
                            StreamEventType.ERROR,
                            error=provider_error,
                            data=dict(event.data),
                        )
                        break
                    if (
                        event.type is StreamEventType.RETRY
                        and event.data.get("is_stall")
                        and not completion_succeeded
                    ):
                        partial_blocks = []
                        assistant_message = None
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial_blocks.append(event.content)
                        if event.delta is not None:
                            partial_blocks.append(TextContent(event.delta))
                    if (
                        event.message is not None
                        and event.type is StreamEventType.MESSAGE_END
                    ):
                        assistant_message = event.message
                    if (
                        event.type is StreamEventType.MESSAGE_END
                        and not event.data.get("truncated")
                    ):
                        completion_succeeded = True
                    yield event
                    if provider_error is not None:
                        yield StreamEvent(
                            StreamEventType.ERROR,
                            error=provider_error,
                        )
                        break
                if provider_error is None and not completion_succeeded:
                    provider_error = ErrorInfo(
                        "stream_error",
                        "provider stream ended before completion",
                    )
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        error=provider_error,
                    )
            except asyncio.CancelledError:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except GeneratorExit:
                await _close_completion(completion)
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise
            except Exception as exc:
                await _close_completion(completion)
                if _task_is_cancelling():
                    self._persist_partial_for_control(partial_blocks, assistant_message)
                    raise asyncio.CancelledError() from exc
                error = _error_info(exc, provider_error=True)
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message, failure=error
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=error,
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            cleanup_error = await _close_completion(completion)
            if _task_is_cancelling():
                self._persist_partial_for_control(partial_blocks, assistant_message)
                raise asyncio.CancelledError()
            if cleanup_error is not None:
                self._persist_partial_with_cancelled_tools(
                    partial_blocks,
                    assistant_message,
                    failure=_error_info(cleanup_error),
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=_error_info(cleanup_error),
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            if provider_error is not None:
                self._persist_partial_with_cancelled_tools(
                    partial_blocks, assistant_message, failure=provider_error
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return
            if completion_succeeded and self.on_completion_success is not None:
                self.on_completion_success()

            if assistant_message is None and partial_blocks:
                assistant_message = Message(MessageRole.ASSISTANT, partial_blocks)
            if assistant_message is None:
                assistant_message = Message(MessageRole.ASSISTANT)
            calls = [
                block.tool_call
                for block in assistant_message.content
                if isinstance(block, ToolUseContent)
            ]
            expanded_calls = [self._expand_auto_invoke(call) for call in calls]
            if expanded_calls != calls:
                expanded_by_id = {
                    call.id: call for call in expanded_calls
                }
                assistant_message = replace(
                    assistant_message,
                    content=[
                        replace(
                            block,
                            tool_call=expanded_by_id[block.tool_call.id],
                        )
                        if isinstance(block, ToolUseContent)
                        else block
                        for block in assistant_message.content
                    ],
                )
                calls = expanded_calls
            _validate_unique_tool_call_ids(calls)
            approval_requests: list[tuple[str, ToolCall]] = []
            for tool_call in calls:
                if not self.plan_mode_allows(tool_call.name):
                    continue
                request = self.tool_registry.prepare_approval(tool_call)
                if request is not None:
                    approval_requests.append((request.request_id, request.tool_call))
            self.store.append_message_with_approval_requests(
                _durable_message(assistant_message),
                approval_requests,
            )
            if not calls:
                yield StreamEvent(
                    StreamEventType.TURN_END,
                    message=assistant_message,
                    data={"turn": turn_number, "tool_calls": 0},
                )
                yield StreamEvent(StreamEventType.AGENT_END)
                return

            dispatch = dispatch_tool_calls(
                self,
                calls,
                _validated_tool_result,
                abort_signal=turn_abort_signal,
            )
            try:
                async for event in dispatch:
                    yield event
            finally:
                await dispatch.aclose()
            yield StreamEvent(
                StreamEventType.TURN_END,
                message=assistant_message,
                data={"turn": turn_number, "tool_calls": len(calls)},
            )

        yield StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("max_turns", f"maximum turns reached: {self.max_turns}"),
        )
        yield StreamEvent(StreamEventType.AGENT_END)

    def _finalize_tool_results(
        self,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]:
        return finalize_agent_results(self, calls, slots)

    def _persist_partial(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        if assistant_message is None:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent) or block.signature
            ]
            if not durable_blocks and failure is None:
                return
            assistant_message = Message(MessageRole.ASSISTANT, durable_blocks)
        if failure is not None:
            metadata = dict(assistant_message.metadata)
            metadata[FAILED_TURN_MARKER] = True
            metadata[FAILED_TURN_ERROR] = failure.to_dict()
            assistant_message = Message(
                assistant_message.role,
                assistant_message.content,
                tool_result=assistant_message.tool_result,
                metadata=metadata,
            )
        self.store.append_message(_durable_message(assistant_message))

    def _persist_partial_for_control(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
    ) -> None:
        try:
            self._persist_partial_with_cancelled_tools(
                partial_blocks, assistant_message
            )
        except Exception as exc:  # noqa: BLE001 - warn when persistence fails
            try:
                warnings.warn(
                    f"failed to persist partial state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except BaseException:  # noqa: BLE001, S110 - warning failure is ignored
                pass

    def _persist_partial_with_cancelled_tools(
        self,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message, failure=failure)
        blocks = (
            assistant_message.content
            if assistant_message is not None
            else partial_blocks
        )
        calls = [
            block.tool_call for block in blocks if isinstance(block, ToolUseContent)
        ]
        self._finalize_tool_results(calls, [None] * len(calls))


def _durable_message(message: Message) -> Message:
    content = [
        block
        for block in message.content
        if not isinstance(block, ThinkingContent) or block.signature
    ]
    if len(content) == len(message.content):
        return message
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=dict(message.metadata),
    )
