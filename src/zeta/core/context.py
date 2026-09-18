"""Context assembly and durable conversation compaction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from .store import ConversationEntry, ConversationStore
from ..providers import jev
from ..types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    FAILED_TURN_MARKER,
    flatten_tool_content,
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolUseContent,
)


class BudgetExceeded(RuntimeError):
    """The committed context cannot fit in the configured budget."""


class SummaryCompletionError(RuntimeError):
    """The no-tools completion did not produce a usable summary."""


class SummaryInputTooLarge(SummaryCompletionError):
    """The source range is too large for a safe summary request."""


class StaleBranchError(RuntimeError):
    """The active branch changed while a summary was in flight."""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    messages: list[Message]
    token_count: int
    digest: str
    compacted: bool = False


@dataclass(frozen=True, slots=True)
class _ContextItem:
    entry: ConversationEntry | None
    message: Message
    fixed: bool = False
    source_id: str | None = None


IMAGE_TOKEN_ESTIMATE = 1024
JEV_TRIAGE_SIZE_FLOOR = 200
JEV_TRIAGE_DROP_THRESHOLD = 0.35


def _message_token_count(message: Message) -> int:
    """Estimate text tokens and charge a small fixed amount per image.

    Base64 is transport data, not text. Without image dimensions, use a fixed
    estimate that keeps images near the 4 MiB transport cap usable.
    """

    value = message.to_dict()
    image_count = 0
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image":
                content[index] = {
                    key: item for key, item in block.items() if key != "data"
                }
                image_count += 1
    tool_result = value.get("tool_result")
    if isinstance(tool_result, dict):
        blocks = tool_result.get("content_blocks")
        if isinstance(blocks, list):
            tool_result["content_blocks"] = [
                {
                    key: item for key, item in block.items() if key != "data"
                }
                if isinstance(block, dict) and block.get("type") == "image"
                else block
                for block in blocks
            ]
            image_count += sum(
                isinstance(block, dict) and block.get("type") == "image"
                for block in blocks
            )
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return max(1, ceil(len(encoded) / 4) + image_count * IMAGE_TOKEN_ESTIMATE)


def _digest(messages: Sequence[Message]) -> str:
    encoded = json.dumps(
        [message.to_dict() for message in messages],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text_from_message(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "".join(parts)


def _strip_thinking(message: Message) -> Message:
    content = [
        block
        for block in message.content
        if isinstance(block, (ImageContent, TextContent, ToolUseContent))
    ]
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=dict(message.metadata),
    )


def _summary_message(message: Message) -> dict[str, Any]:
    """Serialize a message without copying image base64 into a summary prompt."""

    value = message.to_dict()
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(message.content):
            if not isinstance(block, ImageContent):
                continue
            filename = Path(block.path).name if block.path else "clipboard image"
            size = block.size if block.size is not None else "unknown"
            content[index] = {
                "type": "text",
                "text": (
                    f"[image attachment] filename={filename} "
                    f"media_type={block.mime_type} bytes={size}"
                ),
            }
    tool_result = value.get("tool_result")
    if not isinstance(tool_result, dict):
        return value
    blocks = tool_result.get("content_blocks")
    if not isinstance(blocks, list) or not any(
        isinstance(block, dict) and block.get("type") == "image" for block in blocks
    ):
        return value
    tool_result["content"] = flatten_tool_content(blocks, detailed_images=True)
    tool_result.pop("content_blocks", None)
    return value


def _has_tool_call(message: Message) -> bool:
    return any(isinstance(block, ToolUseContent) for block in message.content)


class CompactionPolicy:
    """Summarize a selected range with a completion that has no tools."""

    def __init__(
        self,
        backend: CompletionBackend | None = None,
        *,
        summary_prompt: str = (
            "Summarize the conversation range below. Preserve decisions, facts, "
            "open work, and tool results. Return only the summary."
        ),
    ) -> None:
        self.backend = backend
        self.summary_prompt = summary_prompt

    async def summarize(
        self,
        messages: Sequence[Message],
        *,
        backend: CompletionBackend | None = None,
        system_prompt: Message | None = None,
        max_source_tokens: int | None = None,
        on_success: Callable[[], None] | None = None,
        on_usage: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> str:
        completion_backend = backend or self.backend
        if completion_backend is None:
            raise SummaryCompletionError("compaction requires a completion backend")
        sanitized_messages = [_strip_thinking(message) for message in messages]
        source = json.dumps(
            [_summary_message(message) for message in sanitized_messages],
            sort_keys=True,
            separators=(",", ":"),
        )
        source_tokens = max(1, ceil(len(source) / 4))
        if max_source_tokens is not None and source_tokens > max_source_tokens:
            raise SummaryInputTooLarge(
                f"summary source is too large: {source_tokens} tokens "
                f"exceeds {max_source_tokens}"
            )
        prompt = Message(
            MessageRole.USER,
            [TextContent(f"{self.summary_prompt}\n\n{source}")],
        )
        summary_messages: list[Message] = []
        if system_prompt is not None:
            summary_messages.append(_strip_thinking(system_prompt))
        summary_messages.append(prompt)

        partial: list[ContentBlock] = []
        completed: Message | None = None
        summary_usage: dict[str, Any] = {}
        try:
            completion = completion_backend.complete(summary_messages, [])
            async for event in completion:
                usage = event.data.get("usage")
                if isinstance(usage, Mapping):
                    summary_usage.update(usage)
                if event.type is StreamEventType.ERROR:
                    info = (
                        event.error
                        if isinstance(event.error, ErrorInfo)
                        else ErrorInfo(
                            "backend_error",
                            "provider emitted an invalid error event",
                        )
                    )
                    failure = SummaryCompletionError(info.message)
                    failure.code = info.code
                    failure.status_code = info.status_code
                    raise failure
                if event.type is StreamEventType.MESSAGE_UPDATE:
                    if event.content is not None:
                        partial.append(event.content)
                    if event.delta is not None:
                        partial.append(TextContent(event.delta))
                if event.type is StreamEventType.MESSAGE_END and event.message is not None:
                    completed = event.message
        except Exception as exc:
            if type(getattr(exc, "code", None)) is str:
                raise
            raise SummaryCompletionError("summary completion failed") from exc

        if on_usage is not None and summary_usage:
            on_usage(summary_usage)
        result = completed or Message(MessageRole.ASSISTANT, partial)
        if _has_tool_call(result):
            raise SummaryCompletionError("summary completion returned a tool call")
        summary = _text_from_message(result).strip()
        if not summary or not any(character.isalnum() for character in summary):
            raise SummaryCompletionError("summary completion returned an empty summary")
        if on_success is not None:
            on_success()
        return summary

    @staticmethod
    def should_compact(total_tokens: int, budget: int) -> bool:
        return total_tokens > budget


class ContextAssembler:
    """Build provider messages from the active branch and compact old entries."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        system_prompt: str | Message = "",
        backend: CompletionBackend | None = None,
        compaction_policy: CompactionPolicy | None = None,
        token_counter: Callable[[Message], int] | None = None,
        on_completion_success: Callable[[], None] | None = None,
        jev_compaction: bool = True,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        if retained_tail < 1:
            raise ValueError("retained tail must be at least one")
        self.store = store
        self.token_budget = token_budget
        self.retained_tail = retained_tail
        self.backend = backend
        self.compaction_policy = compaction_policy or CompactionPolicy(backend)
        self.token_counter = token_counter or _message_token_count
        self.on_completion_success = on_completion_success
        self.jev_compaction = jev_compaction
        self.last_compaction_data: dict[str, Any] = {}
        self.system_prompt = (
            system_prompt
            if isinstance(system_prompt, Message)
            else Message(MessageRole.SYSTEM, [TextContent(system_prompt)])
        )
        self.last_context: AssembledContext | None = None
        self.last_usage: dict[str, Any] = {}
        self._provider_token_total: int | None = None
        self._tokens_used_this_session = 0
        self._cache_read_input_tokens_this_session = 0
        self._cache_creation_input_tokens_this_session = 0
        self._uncached_input_tokens_this_session = 0
        self._output_tokens_this_session = 0

    @property
    def digest(self) -> str | None:
        return self.last_context.digest if self.last_context is not None else None

    def needs_compaction(self, *, force: bool = False) -> bool:
        """Return whether the next assembly must compact the active branch."""

        if force:
            return True

        branch = self.store.replay()
        items = self._visible_items(branch)
        system_prompt = self._system_prompt_message()
        messages = [
            *([] if system_prompt is None else [system_prompt]),
            *(item.message for item in items),
        ]
        return self.compaction_policy.should_compact(
            self._total_tokens(messages), self.token_budget
        )

    @property
    def token_count(self) -> int | None:
        return self.last_context.token_count if self.last_context is not None else None

    @property
    def tokens_used_this_session(self) -> int:
        return self._tokens_used_this_session

    @property
    def cache_read_input_tokens_this_session(self) -> int:
        return self._cache_read_input_tokens_this_session

    @property
    def cache_creation_input_tokens_this_session(self) -> int:
        return self._cache_creation_input_tokens_this_session

    @property
    def uncached_input_tokens_this_session(self) -> int:
        return self._uncached_input_tokens_this_session

    @property
    def output_tokens_this_session(self) -> int:
        return self._output_tokens_this_session

    def record_usage(self, usage: Mapping[str, Any]) -> None:
        self.last_usage = dict(usage)
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        if type(input_tokens) is int and input_tokens >= 0:
            self._uncached_input_tokens_this_session += input_tokens
        if type(output_tokens) is int and output_tokens >= 0:
            self._output_tokens_this_session += output_tokens
        if type(cache_read_tokens) is int and cache_read_tokens >= 0:
            self._cache_read_input_tokens_this_session += cache_read_tokens
        if type(cache_creation_tokens) is int and cache_creation_tokens >= 0:
            self._cache_creation_input_tokens_this_session += cache_creation_tokens
        total = usage.get("total_tokens")
        if type(total) is not int:
            known_tokens = [
                value
                for value in (
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if type(value) is int and value >= 0
            ]
            if known_tokens:
                total = sum(known_tokens)
        if type(total) is int and total >= 0:
            self._provider_token_total = total
            self._tokens_used_this_session += total

    def observe_event(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, Mapping):
            self.record_usage(usage)

    async def assemble(
        self,
        *,
        backend: CompletionBackend | None = None,
        force: bool = False,
    ) -> list[Message]:
        context = await self.assemble_context(backend=backend, force=force)
        return list(context.messages)

    async def assemble_context(
        self,
        *,
        backend: CompletionBackend | None = None,
        force: bool = False,
    ) -> AssembledContext:
        self.last_compaction_data = {}
        branch = self.store.replay()
        branch_id = self._branch_id(branch)
        items = self._visible_items(branch)
        boundary = self._tail_boundary(items)
        system_prompt = self._system_prompt_message()
        system_messages = [] if system_prompt is None else [system_prompt]
        committed = [item for item in items if item.fixed]
        committed.extend(item for item in items[boundary:] if not item.fixed)
        committed_messages = [*system_messages, *(item.message for item in committed)]
        committed_tokens = self._count(committed_messages)
        all_messages = [*system_messages, *(item.message for item in items)]
        total_tokens = self._total_tokens(all_messages)
        should_compact = force or self.compaction_policy.should_compact(
            total_tokens, self.token_budget
        )
        if not should_compact:
            return self._save(
                all_messages,
                False,
            )
        if committed_tokens > self.token_budget:
            raise BudgetExceeded(
                "system prompt and retained tail "
                f"({committed_tokens} tokens) exceed the token budget "
                f"({self.token_budget}); raise it with --token-budget"
            )

        candidates = list(items[:boundary])
        if not candidates:
            if force:
                return self._save(all_messages, False)
            raise BudgetExceeded("context exceeds budget and has no compactible range")
        source_entries = {
            item.entry.id: item.entry
            for item in candidates
            if item.entry is not None
        }
        source_ranges = [
            (
                entry.data["source_seq_start"],
                entry.data["source_seq_end"],
            )
            if entry.type == "compaction"
            else (entry.seq, entry.seq)
            for entry in source_entries.values()
        ]
        source_start = min(start for start, _ in source_ranges)
        source_end = max(end for _, end in source_ranges)
        replaces = [
            entry.id
            for entry in source_entries.values()
            if entry.type == "compaction"
        ]
        triaged_messages = [item.message for item in candidates]
        jev_triage: dict[str, Any] | None = None
        if self.jev_compaction:
            triage = await self._try_jev_triage(items, candidates)
            if triage is not None:
                triaged_messages, jev_triage = triage
                triage_context = self._context(
                    [
                        *system_messages,
                        *triaged_messages,
                        *(item.message for item in items[boundary:]),
                    ],
                    True,
                )
                tokens_recovered = jev_triage.get("tokens_recovered", 0)
                if type(tokens_recovered) is not int:
                    tokens_recovered = 0
                triage_token_count = max(
                    0,
                    self._total_tokens(all_messages) - tokens_recovered,
                )
                if triage_token_count <= self.token_budget:
                    if self._branch_id(self.store.replay()) != branch_id:
                        raise StaleBranchError("active branch changed during compaction")
                    persisted_triage_messages = list(triaged_messages)
                    persisted_triage_source_ids: list[str] = []
                    for item in candidates:
                        source_id = self._source_id(item)
                        if source_id is None:
                            raise StaleBranchError(
                                "triage message has no source id"
                            )
                        persisted_triage_source_ids.append(source_id)
                    for item in items[boundary:]:
                        if (
                            item.entry is None
                            or item.entry.type != "compaction"
                            or item.entry.id not in replaces
                        ):
                            continue
                        source_id = self._source_id(item)
                        if source_id is None:
                            raise StaleBranchError(
                                "replayed triage message has no source id"
                            )
                        persisted_triage_messages.append(item.message)
                        persisted_triage_source_ids.append(source_id)
                    try:
                        self.store.append_compaction_marker(
                            "jev triage compaction",
                            source_start,
                            source_end,
                            replaces=replaces,
                            expected_parent_id=branch_id,
                            triage_messages=persisted_triage_messages,
                            triage_source_ids=persisted_triage_source_ids,
                            jev_triage=jev_triage,
                        )
                    except ValueError as exc:
                        raise StaleBranchError(
                            "active branch changed during compaction"
                        ) from exc
                    self.last_compaction_data = {"jev_triage": jev_triage}
                    self._provider_token_total = None
                    self.last_context = triage_context
                    return triage_context
                jev_triage["skipped_summarize"] = False
        summary = await self.compaction_policy.summarize(
            triaged_messages,
            backend=backend or self.backend,
            system_prompt=system_prompt,
            # Cap the source at the budget minus a small overhead for the
            # summary prompt and response, floored at half-budget so tiny
            # budgets used in tests still get a workable cap. Old behaviour
            # was `budget // 2` unconditionally, which dead-ended real
            # sessions with a single large tool_result in the compactible
            # range even though the provider window had room for it.
            max_source_tokens=max(1, self.token_budget // 2, self.token_budget - 8_000),
            on_success=self.on_completion_success,
            on_usage=self.record_usage,
        )
        if self._branch_id(self.store.replay()) != branch_id:
            raise StaleBranchError("active branch changed during compaction")

        marker_messages = self._marker_messages(source_start, source_end, summary)
        proposed_messages = [
            *system_messages,
            *marker_messages,
            *(item.message for item in items[boundary:]),
        ]
        proposed = self._context(proposed_messages, True)
        if proposed.token_count > self.token_budget:
            raise BudgetExceeded("compacted context exceeds the token budget")

        try:
            self.store.append_compaction_marker(
                summary,
                source_start,
                source_end,
                replaces=replaces,
                expected_parent_id=branch_id,
                jev_triage=jev_triage,
            )
        except ValueError as exc:
            raise StaleBranchError("active branch changed during compaction") from exc
        self._provider_token_total = None
        self.last_context = proposed
        if jev_triage is not None:
            self.last_compaction_data = {"jev_triage": jev_triage}
        return proposed

    async def _try_jev_triage(
        self,
        items: Sequence[_ContextItem],
        candidates: Sequence[_ContextItem],
    ) -> tuple[list[Message], dict[str, Any]] | None:
        request_items: list[dict[str, str]] = []
        tools_by_id: dict[str, str] = {}
        for index, item in enumerate(candidates):
            message = item.message
            if (
                item.entry is None
                or message.role is not MessageRole.TOOL_RESULT
                or message.tool_result is None
                or len(message.tool_result.content) <= JEV_TRIAGE_SIZE_FLOOR
            ):
                continue
            source_id = self._source_id(item)
            assert source_id is not None
            tool = self._tool_name(candidates, index)
            request_items.append(
                {
                    "id": source_id,
                    "kind": "tool_result",
                    "tool": tool,
                    "excerpt": message.tool_result.content[:200],
                }
            )
            tools_by_id[source_id] = tool
        if not request_items:
            return None

        task = next(
            (
                _text_from_message(item.message)
                for item in reversed(items)
                if item.message.role is MessageRole.USER
            ),
            "",
        )
        try:
            result = await jev.triage(task, request_items)
            dropped_ids = {
                item_id
                for item_id, probability in result.keep_probabilities.items()
                if probability < JEV_TRIAGE_DROP_THRESHOLD
            }
            dropped_items: list[dict[str, Any]] = []
            triaged_messages: list[Message] = []
            tokens_recovered = 0
            for item in candidates:
                entry = item.entry
                source_id = self._source_id(item)
                if (
                    entry is None
                    or source_id is None
                    or source_id not in dropped_ids
                ):
                    triaged_messages.append(item.message)
                    continue
                estimate = self.token_counter(item.message)
                tombstone = self._tombstone(
                    item.message,
                    tools_by_id.get(source_id, "unknown"),
                    estimate,
                )
                tokens_recovered += max(0, estimate - self.token_counter(tombstone))
                dropped_items.append({"id": source_id, "tokens": estimate})
                triaged_messages.append(tombstone)

            stats: dict[str, Any] = {
                "candidates": len(request_items),
                "dropped": len(dropped_items),
                "tokens_recovered": tokens_recovered,
                "skipped_summarize": True,
                "dropped_items": dropped_items,
            }
            if result.usage:
                stats["usage"] = dict(result.usage)
            return triaged_messages, stats
        except Exception:  # noqa: BLE001 - triage must fail open
            return None

    @staticmethod
    def _tool_name(items: Sequence[_ContextItem], index: int) -> str:
        result = items[index].message.tool_result
        if result is None:
            return "unknown"
        call_index = ContextAssembler._find_tool_call(items, result.tool_call_id, index)
        if call_index is None:
            return "unknown"
        for block in items[call_index].message.content:
            if isinstance(block, ToolUseContent):
                return block.tool_call.name
        return "unknown"

    @staticmethod
    def _source_id(item: _ContextItem) -> str | None:
        if item.source_id is not None:
            return item.source_id
        if item.entry is None:
            return None
        if item.entry.type == "compaction" and item.message.tool_result is not None:
            return f"{item.entry.id}:{item.message.tool_result.tool_call_id}"
        return item.entry.id

    @staticmethod
    def _tombstone(message: Message, tool: str, estimate: int) -> Message:
        result = message.tool_result
        if result is None:
            return message
        text = (
            f"[dropped by jev-compaction: {tool} result, "
            f"~{max(1, estimate)} tokens]"
        )
        block = {
            "type": "text",
            "text": text,
            "truncated": False,
            "full_size": len(text.encode("utf-8")),
        }
        return Message(
            MessageRole.TOOL_RESULT,
            [TextContent(text)],
            tool_result=type(result)(
                result.tool_call_id,
                text,
                result.is_error,
                content_blocks=[block],
                structured_content=result.structured_content,
                is_canceled=result.is_canceled,
            ),
            metadata=dict(message.metadata),
        )

    def _context(self, messages: list[Message], compacted: bool) -> AssembledContext:
        return AssembledContext(
            messages=messages,
            token_count=self._count(messages),
            digest=_digest(messages),
            compacted=compacted,
        )

    def _save(self, messages: list[Message], compacted: bool) -> AssembledContext:
        context = self._context(messages, compacted)
        self.last_context = context
        return context

    def _count(self, messages: Sequence[Message]) -> int:
        return sum(self.token_counter(message) for message in messages)

    def _system_prompt_message(self) -> Message | None:
        if not _text_from_message(self.system_prompt).strip():
            return None
        return self.system_prompt

    def _total_tokens(self, messages: Sequence[Message]) -> int:
        estimated = self._count(messages)
        if self._provider_token_total is None:
            return estimated
        return max(estimated, self._provider_token_total)

    def _visible_items(self, entries: Sequence[ConversationEntry]) -> list[_ContextItem]:
        all_markers = [entry for entry in entries if entry.type == "compaction"]
        superseded_ids: set[str] = set()
        for marker in all_markers:
            superseded_ids.update(marker.data.get("replaces", []))
        markers = [entry for entry in all_markers if entry.id not in superseded_ids]
        compacted_ranges = [
            (entry.data["source_seq_start"], entry.data["source_seq_end"])
            for entry in markers
        ]
        markers_by_start = {
            entry.data["source_seq_start"]: entry for entry in markers
        }
        items: list[_ContextItem] = []
        emitted_marker_ids: set[str] = set()
        failed_tool_call_ids: set[str] = set()
        for entry in entries:
            marker = markers_by_start.get(entry.seq)
            if marker is not None:
                items.extend(self._marker_items(marker))
                emitted_marker_ids.add(marker.id)
            if entry.type == "compaction":
                continue
            if entry.type in {"warning", "checkpoint", "fork"}:
                continue
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            if message.metadata.get(FAILED_TURN_MARKER):
                failed_tool_call_ids.update(
                    block.tool_call.id
                    for block in message.content
                    if isinstance(block, ToolUseContent)
                )
                continue
            if any(start <= entry.seq <= end for start, end in compacted_ranges):
                continue
            if (
                message.role is MessageRole.TOOL_RESULT
                and message.tool_result is not None
                and message.tool_result.tool_call_id in failed_tool_call_ids
            ):
                continue
            items.append(_ContextItem(entry, message))
        for marker in markers:
            if marker.id not in emitted_marker_ids:
                items.extend(self._marker_items(marker))
        return items

    @staticmethod
    def _marker_items(entry: ConversationEntry) -> list[_ContextItem]:
        triage_messages = entry.data.get("triage_messages")
        if isinstance(triage_messages, list):
            source_ids = entry.data.get("triage_source_ids")
            if not isinstance(source_ids, list) or len(source_ids) != len(triage_messages):
                source_ids = [
                    f"{entry.id}:{index}" for index in range(len(triage_messages))
                ]
            return [
                _ContextItem(
                    entry,
                    Message.from_dict(message),
                    source_id=(source_ids[index] if isinstance(source_ids[index], str) else None),
                )
                for index, message in enumerate(triage_messages)
            ]
        messages = ContextAssembler._marker_messages(
            entry.data["source_seq_start"],
            entry.data["source_seq_end"],
            entry.data["summary"],
        )
        return [_ContextItem(entry, message, fixed=True) for message in messages]

    @staticmethod
    def _marker_messages(
        source_start: int,
        source_end: int,
        summary: str,
    ) -> list[Message]:
        marker_text = f"[compaction marker: entries {source_start}–{source_end}]"
        metadata = {
            "source_seq_start": source_start,
            "source_seq_end": source_end,
        }
        return [
            Message(
                MessageRole.COMPACTION,
                [TextContent(marker_text)],
                metadata=metadata,
            ),
            Message(
                MessageRole.ASSISTANT,
                [TextContent(summary)],
                metadata={"compaction_summary": True, **metadata},
            ),
        ]

    @staticmethod
    def _branch_id(entries: Sequence[ConversationEntry]) -> str | None:
        return entries[-1].id if entries else None

    def _tail_boundary(self, items: Sequence[_ContextItem]) -> int:
        tail_start = max(0, len(items) - self.retained_tail)
        while tail_start > 0 and self._is_tool_result(items[tail_start]):
            result = items[tail_start].message.tool_result
            if result is None:
                break
            call_id = result.tool_call_id
            call_index = self._find_tool_call(items, call_id, tail_start)
            if call_index is None:
                break
            tail_start = call_index
        return tail_start

    @staticmethod
    def _is_tool_result(item: _ContextItem) -> bool:
        return (
            item.message.role is MessageRole.TOOL_RESULT
            and item.message.tool_result is not None
        )

    @staticmethod
    def _find_tool_call(
        items: Sequence[_ContextItem], call_id: str, before: int
    ) -> int | None:
        for index in range(before - 1, -1, -1):
            message = items[index].message
            if any(
                isinstance(block, ToolUseContent) and block.tool_call.id == call_id
                for block in message.content
            ):
                return index
        return None
