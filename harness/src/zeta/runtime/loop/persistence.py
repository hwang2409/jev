"""Persistence helpers for the provider-neutral agent loop."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from ...agent.receipt import finalize_agent_results
from ...protocol.types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


class PersistenceMixin:
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
