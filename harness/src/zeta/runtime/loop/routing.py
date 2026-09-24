"""Routing and memory helpers for the provider-neutral agent loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ...agent.plan_mode import PLAN_MODE_TOOLS
from ...core.context import MEMORY_INJECTION_PREFIX
from ...protocol.types import (
    MessageRole,
    RoutingSchemaContent,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)
from ...tools.route import ROUTE_TOPK_CONFIDENCE, build_catalog
from . import (
    _BROWSER_ELEMENT_TOOLS,
    MEMORY_INJECTION_EXCERPT_CHARS,
    MEMORY_INJECTION_TOP_K,
    MEMORY_INJECTION_TOTAL_CHARS,
    MEMORY_RELEVANCE_GATE,
    NEEDS_TOOL_GATE,
    MemoryInjectionSkipReason,
    _logger,
    _memory_content_hash,
)

if TYPE_CHECKING:
    from ..tools.browser.catalog import BrowserCatalog


class RoutingMixin:
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
            schema for schema in active if schema.get("name") in allowed | {"route"}
        ]

    def _record_routed_tools(self, tools: list[str] | None) -> None:
        if tools is None:
            self._routed_tools = []
            self._router_fail_open = True
            return
        self._routed_tools = list(dict.fromkeys(tools))
        self._router_fail_open = False

    def set_browser_catalog(self, catalog: BrowserCatalog | None) -> None:
        """Keep the current page catalog in the loop's router context."""

        self.tool_registry.browser_catalog = catalog
        self.tool_registry.router_browser_catalog = catalog

    def browser_catalog(self) -> BrowserCatalog | None:
        return self.tool_registry.router_browser_catalog

    def _browser_catalog_state(self) -> dict[str, object]:
        catalog = self.browser_catalog()
        if catalog is None:
            return {"snapshot_id": None, "entries": []}
        return {
            "snapshot_id": catalog.snapshot_id,
            "generation": catalog.generation,
            "entries": [asdict(entry) for entry in catalog.entries],
        }

    def _browser_element_rejection(self, tool_call: ToolCall) -> ToolResult | None:
        if not self.router_mode or tool_call.name not in _BROWSER_ELEMENT_TOOLS:
            return None
        element_id = tool_call.arguments.get("element_id")
        if tool_call.name == "browser_extract" and element_id is None:
            return None
        snapshot_id = tool_call.arguments.get("snapshot_id")
        catalog = self.browser_catalog()
        entry = None
        if (
            catalog is not None
            and type(snapshot_id) is int
            and isinstance(element_id, str)
            and snapshot_id == catalog.snapshot_id
        ):
            entry = next(
                (
                    candidate
                    for candidate in catalog.entries
                    if candidate.element_id == element_id
                ),
                None,
            )
        if entry is not None:
            role = tool_call.arguments.get("role")
            affordance = tool_call.arguments.get("affordance")
            if (
                role is None
                or affordance is None
                or (role == entry.role and affordance == entry.affordance)
            ):
                return None
        self.unrouted_attempts += 1
        result = ToolResult(
            tool_call.id,
            "browser element is not in the current catalog",
            is_error=True,
            structured_content={"error_kind": "unrouted_element"},
        )
        return self.tool_registry.govern_tool_result(tool_call, result)

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
                self._router_batch_allowed_tools = set(self._router_auto_allowed_tools)
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
        browser_rejection = self._browser_element_rejection(tool_call)
        if browser_rejection is not None:
            return browser_rejection
        if (
            self.router_mode
            and self.router_style == "auto"
            and tool_call.name == "invoke"
        ):
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
    def _memory_path_key(path: str) -> str:
        return str(Path(path).expanduser().resolve())

    def _record_memory_store(self, path: str) -> None:
        self._actively_modified_memory_paths.add(self._memory_path_key(path))

    @staticmethod
    def _memory_section_date(candidate: Mapping[str, object]) -> datetime | None:
        heading = candidate.get("heading")
        if not isinstance(heading, list) or len(heading) < 2:
            return None
        section = heading[-1]
        if not isinstance(section, str):
            return None
        try:
            parsed = datetime.fromisoformat(section)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def _memory_skip_reasons(
        self, candidates: Sequence[Mapping[str, object]]
    ) -> dict[str, str]:
        by_path: dict[str, list[Mapping[str, object]]] = {}
        for candidate in candidates:
            path = candidate.get("path")
            if isinstance(path, str):
                by_path.setdefault(self._memory_path_key(path), []).append(candidate)

        skipped: dict[str, str] = {}
        for path, path_candidates in by_path.items():
            dated = [
                date
                for candidate in path_candidates
                if (date := self._memory_section_date(candidate)) is not None
            ]
            newest = max(dated) if dated else None
            for candidate in path_candidates:
                candidate_id = candidate.get("id")
                if not isinstance(candidate_id, str):
                    continue
                if path in self._actively_modified_memory_paths:
                    skipped[candidate_id] = (
                        MemoryInjectionSkipReason.ACTIVELY_MODIFIED.value
                    )
                    continue
                if newest is None:
                    continue
                candidate_date = self._memory_section_date(candidate)
                if candidate_date is None or candidate_date < newest:
                    skipped[candidate_id] = MemoryInjectionSkipReason.SUPERSEDED.value
        return skipped

    @staticmethod
    def _memory_key(
        value: Mapping[str, object],
    ) -> tuple[str, tuple[str, ...], str] | None:
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
        content_hash = value.get("content_hash")
        excerpt = value.get("excerpt")
        if not isinstance(content_hash, str):
            content_hash = (
                _memory_content_hash(excerpt) if isinstance(excerpt, str) else ""
            )
        return path, tuple(heading), content_hash

    def _known_memory_keys(self) -> set[tuple[str, tuple[str, ...], str]]:
        keys: set[tuple[str, tuple[str, ...], str]] = set()
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

    async def _retrieve_memory(
        self, query: str
    ) -> tuple[list[dict[str, object]], str | None, bool]:
        self._memory_retrieval_skips = []
        if self.tool_registry.memory_config is None:
            return [], "memory_unconfigured", False
        try:
            from . import _memory_search as memory_search

            result = await memory_search(
                self.tool_registry,
                {"query": query},
                self.tool_registry.abort_signal,
            )
            if result.get("isError") is True:
                _logger.warning("memory injection search failed")
                return [], "memory_error", False
            structured = result.get("structuredContent")
            items = structured.get("items") if isinstance(structured, Mapping) else None
            if not isinstance(items, list):
                return [], "memory_error", False
            known = self._known_memory_keys()
            known_hashes = {key[2] for key in known if key[2]}
            raw_candidates: list[dict[str, object]] = []
            for raw in items:
                if not isinstance(raw, Mapping):
                    continue
                excerpt = raw.get("excerpt")
                if not isinstance(excerpt, str) or not excerpt:
                    continue
                content_hash = _memory_content_hash(excerpt)
                excerpt = excerpt[:MEMORY_INJECTION_EXCERPT_CHARS]
                key = self._memory_key({**raw, "content_hash": content_hash})
                if key is None:
                    continue
                raw_candidates.append(
                    {
                        "id": f"candidate-{len(raw_candidates)}",
                        "path": key[0],
                        "heading": list(key[1]),
                        "excerpt": excerpt,
                        "content_hash": content_hash,
                    }
                )
            skip_reasons = self._memory_skip_reasons(raw_candidates)
            self._memory_retrieval_skips = [
                {"id": candidate_id, "reason": reason}
                for candidate_id, reason in skip_reasons.items()
            ]
            candidates: list[dict[str, object]] = []
            deduped = False
            capped = False
            for candidate in raw_candidates:
                if len(candidates) >= MEMORY_INJECTION_TOP_K:
                    capped = True
                    break
                candidate_id = candidate["id"]
                if candidate_id in skip_reasons:
                    continue
                content_hash = candidate["content_hash"]
                key = self._memory_key(candidate)
                if key is None or not isinstance(content_hash, str):
                    continue
                if key in known or content_hash in known_hashes:
                    deduped = True
                    continue
                candidates.append(candidate)
                known.add(key)
                known_hashes.add(content_hash)
            if candidates:
                return candidates, None, capped
            if self._memory_retrieval_skips:
                reasons = {item["reason"] for item in self._memory_retrieval_skips}
                if "actively_modified" in reasons:
                    return [], MemoryInjectionSkipReason.ACTIVELY_MODIFIED.value, capped
                if "superseded" in reasons:
                    return [], MemoryInjectionSkipReason.SUPERSEDED.value, capped
            return [], "deduped" if deduped else "no_candidates", capped
        except Exception as exc:  # noqa: BLE001 - retrieval fails open
            _logger.warning("memory injection search failed: %s", exc)
            return [], "memory_error", False

    async def _inject_memory(
        self,
        candidates: list[dict[str, object]],
        relevance_scores: Mapping[str, float] | None,
        *,
        retrieval_reason: str | None = None,
        retrieval_capped: bool = False,
    ) -> dict[str, object]:
        retrieval_skips = list(self._memory_retrieval_skips)
        self._memory_retrieval_skips = []
        decision: dict[str, object] = {
            "candidate_scores": [],
            "injected_count": 0,
            "chars": 0,
            "reason": retrieval_reason,
            "skipped_candidates": retrieval_skips,
        }
        direct_skip_reasons = self._memory_skip_reasons(candidates)
        if direct_skip_reasons:
            skipped_ids = {entry["id"] for entry in retrieval_skips}
            retrieval_skips.extend(
                {"id": candidate_id, "reason": reason}
                for candidate_id, reason in direct_skip_reasons.items()
                if candidate_id not in skipped_ids
            )
            decision["skipped_candidates"] = retrieval_skips
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("id") not in direct_skip_reasons
            ]
            if not candidates and decision["reason"] is None:
                decision["reason"] = next(iter(direct_skip_reasons.values()))
        if retrieval_reason is not None:
            return decision
        if relevance_scores is None:
            decision["reason"] = "jev_error"
            return decision
        candidate_scores: list[dict[str, object]] = []
        eligible: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_id = candidate.get("id")
            score = (
                relevance_scores.get(candidate_id)
                if isinstance(candidate_id, str)
                else None
            )
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                decision["reason"] = "memory_error"
                return decision
            candidate_scores.append({"id": candidate_id, "score": score})
            if score > MEMORY_RELEVANCE_GATE:
                eligible.append(candidate)
        decision["candidate_scores"] = candidate_scores
        if not eligible:
            decision["reason"] = "below_relevance"
            return decision
        try:
            known = self._known_memory_keys()
            known_hashes = {key[2] for key in known if key[2]}
            blocks: list[TextContent] = []
            injected_items: list[dict[str, object]] = []
            total_chars = 0
            deduped = False
            capped = retrieval_capped
            for candidate in eligible:
                if len(blocks) >= MEMORY_INJECTION_TOP_K:
                    capped = True
                    break
                excerpt = candidate.get("excerpt")
                path = candidate.get("path")
                heading_value = candidate.get("heading", [])
                content_hash = candidate.get("content_hash")
                if (
                    not isinstance(excerpt, str)
                    or not isinstance(path, str)
                    or not isinstance(heading_value, list)
                    or not all(isinstance(part, str) for part in heading_value)
                    or not isinstance(content_hash, str)
                ):
                    decision["reason"] = "memory_error"
                    return decision
                key = (path, tuple(heading_value), content_hash)
                if key in known or content_hash in known_hashes:
                    deduped = True
                    continue
                heading = " > ".join(heading_value) or "(document)"
                text = (
                    f"{MEMORY_INJECTION_PREFIX}\n"
                    f"path: {path}\n"
                    f"heading: {heading}\n"
                    f"{excerpt}"
                )
                if total_chars + len(text) > MEMORY_INJECTION_TOTAL_CHARS:
                    capped = True
                    break
                blocks.append(TextContent(text))
                injected_items.append(
                    {
                        "path": path,
                        "heading": list(heading_value),
                        "content_hash": content_hash,
                    }
                )
                known.add(key)
                known_hashes.add(content_hash)
                total_chars += len(text)
            if not blocks:
                decision["reason"] = (
                    "capped" if capped else "deduped" if deduped else "memory_error"
                )
                return decision
            if not self._persist_memory_blocks(blocks, injected_items):
                decision["reason"] = "memory_error"
                return decision
            decision["injected_count"] = len(blocks)
            decision["chars"] = total_chars
            if capped:
                decision["reason"] = "capped"
            return decision
        except Exception as exc:  # noqa: BLE001 - injection fails open
            _logger.warning("memory injection failed: %s", exc)
            decision["reason"] = "memory_error"
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
        query = self._memory_query(user_text)
        candidates, retrieval_reason, retrieval_capped = await self._retrieve_memory(
            query
        )
        if not candidates:
            decision = await self._inject_memory(
                candidates,
                None,
                retrieval_reason=retrieval_reason,
                retrieval_capped=retrieval_capped,
            )
            return decision, {}
        try:
            from . import memory_relevance as relevance

            result = await relevance(query, candidates)
        except Exception as exc:  # noqa: BLE001 - relevance fails open
            _logger.warning("memory injection relevance failed: %s", exc)
            decision = await self._inject_memory(candidates, None)
            return decision, {}
        decision = await self._inject_memory(
            candidates,
            result.scores,
            retrieval_capped=retrieval_capped,
        )
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
        memory_candidates: list[dict[str, object]] = []
        memory_retrieval_reason: str | None = None
        memory_retrieval_capped = False
        if self.memory_injection:
            (
                memory_candidates,
                memory_retrieval_reason,
                memory_retrieval_capped,
            ) = await self._retrieve_memory(
                "\n".join(part for part in (task, last_assistant) if part)
            )
        try:
            route_kwargs = (
                {"memory_candidates": memory_candidates} if memory_candidates else {}
            )
            from . import auto_route as route

            result = await route(
                task,
                last_assistant,
                last_results,
                self._auto_catalog(),
                **route_kwargs,
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
            routing_decision = {
                "error": str(exc),
                "advertised": sorted(self._router_auto_allowed_tools),
                "fail_open": True,
            }
            if self.memory_injection:
                routing_decision["memory_injection"] = await self._inject_memory(
                    memory_candidates,
                    None,
                    retrieval_reason=memory_retrieval_reason,
                    retrieval_capped=memory_retrieval_capped,
                )
            return full_catalog, routing_decision
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
                memory_candidates,
                result.memory_relevance,
                retrieval_reason=memory_retrieval_reason,
                retrieval_capped=memory_retrieval_capped,
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
            isinstance(block, RoutingSchemaContent) and block.text == text
            for block in target.content
        ):
            return
        self.store.append_message_revision(
            target_entry.id,
            replace(target, content=[*target.content, RoutingSchemaContent(text)]),
        )

    def _expand_auto_invoke(self, call: ToolCall) -> ToolCall:
        if not self.router_mode or self.router_style != "auto" or call.name != "invoke":
            return call
        tool = call.arguments.get("tool")
        args = call.arguments.get("args")
        if isinstance(tool, str) and isinstance(args, dict):
            return ToolCall(call.id, tool, args)
        return call
