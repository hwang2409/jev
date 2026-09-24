"""Child-agent helpers for the provider-neutral agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from ...agent.receipt import MAX_AGENT_RESULT_BYTES, TerminalState, terminal_state
from ...agent.runner import run_agent_tool
from ...core.abort import AbortSignal as ToolAbortSignal
from ...protocol.types import StreamEvent, ToolCall, ToolResult
from ...tools import ToolStreamPublisher
from ...tools.agent import agent_result
from ...tools.registry import ToolExecutionContext
from . import TaskResult, _error_info, _validated_tool_result


class AgentChildMixin:
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
