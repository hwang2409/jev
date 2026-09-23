import asyncio


import base64


import io


import json


from pathlib import Path


from types import SimpleNamespace


import pytest


from rich.console import Console


from zeta.core.context import ContextAssembler


from zeta.core.store import ConversationStore


from zeta.media.images import IMAGE_DEGRADATION_WARNING, detect_image_media_type


from zeta.runtime.loop import _validated_tool_result


from zeta.providers.anthropic import build_messages_payload


from zeta.providers.codex import build_responses_payload


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools.read import IMAGE_MAX_BYTES


from zeta.tui.checkpoints import CheckpointTranscriptMixin


from zeta.tui.render import render_event


from zeta.protocol.types import (
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


IMAGE_FIXTURES = (
    ("png", "image/png", PNG),
    ("jpeg", "image/jpeg", b"\xff\xd8\xff\xd9"),
    ("gif", "image/gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
    (
        "webp",
        "image/webp",
        b"RIFF"
        + (22).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10,
    ),
)


INVALID_IMAGE_FIXTURES = (
    ("image/png", PNG[:24]),
    ("image/jpeg", b"\xff\xd8\xff"),
    ("image/gif", b"GIF89a\x01\x00\x01\x00"),
    (
        "image/webp",
        b"RIFF"
        + (23).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10,
    ),
)


def _webp_data(chunk_type: bytes, chunk_data: bytes) -> bytes:
    chunk = (
        chunk_type
        + len(chunk_data).to_bytes(4, "little")
        + chunk_data
        + (b"\x00" if len(chunk_data) % 2 else b"")
    )
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


READ_WEBP_FIXTURES = (
    (
        "vp8",
        _webp_data(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00"),
    ),
    ("vp8l", _webp_data(b"VP8L", b"/\x00\x00\x00\x00")),
    ("animated-vp8x", _webp_data(b"VP8X", b"\x02" + b"\x00" * 9)),
)


def _oversized_webp() -> bytes:
    chunk_size = IMAGE_MAX_BYTES
    riff_size = chunk_size + 12
    payload = b"\x2f\x00\x00\x00\x00" + b"x" * (chunk_size - 5)
    return (
        b"RIFF"
        + riff_size.to_bytes(4, "little")
        + b"WEBPVP8L"
        + chunk_size.to_bytes(4, "little")
        + payload
    )


def _oversized_lookalike() -> bytes:
    prefix = b"RIFFxxxxWEBPthis is UTF-8 text\n"
    return prefix + b"x" * (IMAGE_MAX_BYTES + 1 - len(prefix))


def _oversized_invalid_webp() -> bytes:
    data = _oversized_webp()
    return data[:12] + b"NOPE" + data[16:]


def _no_eoi_progressive_jpeg() -> bytes:
    return b"\xff\xd8\xff\xc2\x00\x11" + b"progressive JPEG without EOI"


def _jpeg_eoi_in_app_payload() -> bytes:
    return b"\xff\xd8\xff\xe1\x00\x05ab\xff\xd9"


def _progressive_jpeg() -> bytes:
    return (
        b"\xff\xd8\xff\xc2\x00\x11"
        + b"\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x00"
        + b"\xff\xd9"
    )


def _oversized_truncated_png() -> bytes:
    data = PNG[:24]
    return data + b"x" * (IMAGE_MAX_BYTES + 1 - len(data))


def _malformed_webp_chunks() -> bytes:
    return (
        b"RIFF"
        + (26).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10
        + b"NOPE"
    )


def _leading_junk_webp() -> bytes:
    junk = b"JUNK" + (0).to_bytes(4, "little")
    codec = b"VP8X" + (10).to_bytes(4, "little") + b"\x00" * 10
    body = b"WEBP" + junk + codec
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _oversized(data: bytes) -> bytes:
    return data + b"x" * (IMAGE_MAX_BYTES + 1 - len(data))


DECISION_TABLE_CASES = [
    pytest.param("row-1-text", b"plain text\n", {}, "text", id="row-1-text"),
    pytest.param(
        "row-2-oversized-invalid-webp",
        _oversized_invalid_webp(),
        {},
        "size",
        id="row-2-oversized-invalid-webp",
    ),
    pytest.param(
        "row-2-oversized-truncated-png",
        _oversized_truncated_png(),
        {},
        "size",
        id="row-2-oversized-truncated-png",
    ),
    pytest.param(
        "row-2-oversized-lookalike",
        _oversized_lookalike(),
        {},
        "size",
        id="row-2-oversized-lookalike",
    ),
]


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-2-oversized-valid-{format_name}",
        _oversized(data),
        {},
        "size",
        id=f"row-2-oversized-valid-{format_name}",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-2-oversized-invalid-{format_name}",
        _oversized(invalid_data),
        {},
        "size",
        id=f"row-2-oversized-invalid-{format_name}",
    )
    for (format_name, _mime_type, _valid_data), (_invalid_mime, invalid_data) in zip(
        IMAGE_FIXTURES, INVALID_IMAGE_FIXTURES, strict=True
    )
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-3-oversized-valid-{format_name}-with-paging",
        _oversized(data),
        {"offset": 1},
        "paging",
        id=f"row-3-oversized-valid-{format_name}-with-paging",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-3-oversized-invalid-{format_name}-with-paging",
        _oversized(invalid_data),
        {"offset": 1},
        "paging",
        id=f"row-3-oversized-invalid-{format_name}-with-paging",
    )
    for (format_name, _mime_type, _valid_data), (_invalid_mime, invalid_data) in zip(
        IMAGE_FIXTURES, INVALID_IMAGE_FIXTURES, strict=True
    )
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-4-{format_name}-with-paging",
        data,
        {"limit": 1},
        "paging",
        id=f"row-4-{format_name}-with-paging",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-4-{format_name}-invalid-with-paging",
        data,
        {"limit": 1},
        "paging",
        id=f"row-4-{format_name}-invalid-with-paging",
    )
    for (format_name, _image_mime_type, _valid_data), (_mime_type, data) in zip(
        IMAGE_FIXTURES, INVALID_IMAGE_FIXTURES, strict=True
    )
)


DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-5-{format_name}-trailing-data",
        (_progressive_jpeg() if format_name == "jpeg" else data) + b"trailing metadata",
        {},
        "image",
        id=f"row-5-{format_name}-trailing-data",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)


DECISION_TABLE_CASES.extend(
    [
        pytest.param(
            "row-6-invalid-png",
            PNG[:24],
            {},
            (
                True,
                "'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte",
            ),
            id="row-6-invalid-png",
        ),
        pytest.param(
            "row-6-no-eoi-progressive-jpeg",
            _no_eoi_progressive_jpeg(),
            {},
            (
                True,
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
            ),
            id="row-6-no-eoi-progressive-jpeg",
        ),
        pytest.param(
            "row-6-jpeg-eoi-in-app-payload",
            _jpeg_eoi_in_app_payload(),
            {},
            (
                True,
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
            ),
            id="row-6-jpeg-eoi-in-app-payload",
        ),
        pytest.param(
            "row-6-invalid-gif",
            b"GIF89a\x01\x00\x01\x00\x00\x00;",
            {},
            (False, "GIF89a\x01\x00\x01\x00\x00\x00;"),
            id="row-6-invalid-gif",
        ),
        pytest.param(
            "row-6-invalid-webp-lookalike",
            b"RIFFxxxxWEBPthis is UTF-8 text\n",
            {},
            (False, "RIFFxxxxWEBPthis is UTF-8 text"),
            id="row-6-invalid-webp-lookalike",
        ),
        pytest.param(
            "row-6-malformed-webp-chunks",
            _malformed_webp_chunks(),
            {},
            (False, _malformed_webp_chunks().decode()),
            id="row-6-malformed-webp-chunks",
        ),
        pytest.param(
            "row-6-leading-junk-webp",
            _leading_junk_webp(),
            {},
            (False, _leading_junk_webp().decode()),
            id="row-6-leading-junk-webp",
        ),
    ]
)


def _image_tool_result(data: bytes = PNG) -> ToolResult:
    encoded = base64.b64encode(data).decode("ascii")
    return ToolResult(
        "read-call",
        "filename=screenshot.png bytes=70 format=png",
        content_blocks=[
            {
                "type": "text",
                "text": "filename=screenshot.png bytes=70 format=png",
                "truncated": False,
                "full_size": 43,
            },
            {
                "type": "image",
                "data": encoded,
                "mimeType": "image/png",
                "path": "/tmp/screenshot.png",
                "size": len(data),
            },
        ],
        structured_content={
            "filename": "screenshot.png",
            "bytes": len(data),
            "format": "png",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("format_name", "mime_type", "data"), IMAGE_FIXTURES)
async def test_read_detects_images_by_magic_bytes(
    tmp_path: Path, format_name: str, mime_type: str, data: bytes
) -> None:
    path = tmp_path / f"renamed.{format_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-image", "read", {"path": path.name})
    )

    assert result["isError"] is False
    image = result["content"][1]
    assert image["type"] == "image"
    assert image["mimeType"] == mime_type
    assert base64.b64decode(image["data"]) == data
    assert result["structuredContent"]["format"] == format_name


@pytest.mark.asyncio
async def test_read_keeps_text_behavior_for_non_images(tmp_path: Path) -> None:
    path = tmp_path / "note.bin"
    data = "one\r\ntwo\n三".encode()
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-text", "read", {"path": path.name})
    )

    assert result["isError"] is False
    block = result["content"][0]
    assert block["text"] == "one\ntwo\n三"
    assert block["full_size"] == len("one\ntwo\n三".encode())
    assert block["truncated"] is False
    assert result["structuredContent"]["sha256"]


@pytest.mark.asyncio
async def test_read_falls_back_for_webp_lookalike_text(tmp_path: Path) -> None:
    path = tmp_path / "note.bin"
    path.write_bytes(b"RIFFxxxxWEBPthis is UTF-8 text\n")

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-lookalike", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "RIFFxxxxWEBPthis is UTF-8 text"


def test_png_lookalike_fails_full_image_validation() -> None:
    data = b"\x89PNG\r\n\x1a\nthis is UTF-8 text"

    assert detect_image_media_type(data) == "image/png"
    assert detect_image_media_type(data, complete=True) is None


@pytest.mark.parametrize(("mime_type", "data"), INVALID_IMAGE_FIXTURES)
def test_complete_image_validation_requires_container_structure(
    mime_type: str, data: bytes
) -> None:
    assert detect_image_media_type(data) == mime_type
    assert detect_image_media_type(data, complete=True) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "data", "arguments", "expected"), DECISION_TABLE_CASES
)
async def test_image_read_decision_table(
    tmp_path: Path,
    case_name: str,
    data: bytes,
    arguments: dict[str, int],
    expected: str,
) -> None:
    path = tmp_path / f"{case_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(case_name, "read", {"path": path.name, **arguments})
    )

    if expected == "size":
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            f"image is {len(data)} bytes; cap is "
            f"{IMAGE_MAX_BYTES} bytes (4 MiB)"
        )
    elif expected == "paging":
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "offset and limit are not supported for image reads"
        )
    elif expected == "image":
        assert result["isError"] is False
        assert result["content"][1]["type"] == "image"
        assert base64.b64decode(result["content"][1]["data"]) == data
    elif expected == "text":
        assert result["isError"] is False
        assert result["content"][0]["text"] == "plain text"
    else:
        assert isinstance(expected, tuple)
        expected_error, expected_text = expected
        assert result["isError"] is expected_error
        assert result["content"][0]["text"] == expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_name", "data"), READ_WEBP_FIXTURES)
async def test_read_detects_webp_codecs(
    tmp_path: Path, case_name: str, data: bytes
) -> None:
    path = tmp_path / f"{case_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(f"read-{case_name}", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["mimeType"] == "image/webp"
    assert base64.b64decode(result["content"][1]["data"]) == data
    assert result["structuredContent"]["format"] == "webp"


@pytest.mark.asyncio
async def test_read_rejects_oversized_images_with_size_and_cap(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    path.write_bytes(PNG + b"x" * (4 * 1024 * 1024))

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-large-image", "read", {"path": path.name})
    )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "image is 4194374 bytes" in message
    assert "cap is 4194304 bytes (4 MiB)" in message


@pytest.mark.asyncio
async def test_read_rejects_oversized_webp_before_sample_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.webp"
    path.write_bytes(_oversized_webp())

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-large-webp", "read", {"path": path.name})
    )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "image is 4194324 bytes" in message
    assert "cap is 4194304 bytes (4 MiB)" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["offset", "limit"])
async def test_read_rejects_paging_arguments_for_images(
    tmp_path: Path, argument: str
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(PNG)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-paged-image", "read", {"path": path.name, argument: 1})
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "offset and limit are not supported for image reads"
    )


@pytest.mark.asyncio
async def test_read_image_near_cap_fits_default_context_budget(tmp_path: Path) -> None:
    path = tmp_path / "near-cap.png"
    path.write_bytes(PNG + b"x" * (IMAGE_MAX_BYTES - len(PNG)))

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-near-cap", "read", {"path": path.name})
    )
    assert result["isError"] is False

    blocks = result["content"]
    receipt = blocks[0]["text"]
    store = ConversationStore(tmp_path, session_id="near-cap-session")
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(
                "read-near-cap",
                receipt,
                content_blocks=blocks,
                structured_content=result["structuredContent"],
            ),
        )
    )

    assembled = await ContextAssembler(store).assemble()

    assert assembled[0].tool_result is not None
    assert assembled[0].tool_result.content_blocks is not None
    assert len(assembled[0].tool_result.content_blocks[1]["data"]) > 5_000_000


def test_codex_pasted_image_path_keeps_image_bytes_in_user_content() -> None:
    encoded = base64.b64encode(PNG).decode("ascii")
    payload = build_responses_payload(
        [
            Message(
                MessageRole.USER,
                [TextContent("inspect this"), ImageContent(encoded, "image/png")],
            )
        ],
        [],
        model="codex-test",
    )

    image = payload["input"][0]["content"][1]
    assert image == {
        "type": "input_image",
        "image_url": "data:image/png;base64," + encoded,
    }
