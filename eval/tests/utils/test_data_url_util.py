from __future__ import annotations

import base64
from urllib.parse import quote

import pytest

from agentrl.eval.utils import data_url as data_url_module, DataUrlUtil


def _to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_scrub_handles_all_supported_collections() -> None:
    data_txt = "data:text/plain;base64," + _to_base64(b"hello")
    data_png = "data:image/png;base64," + _to_base64(b"png")
    data_json = "data:application/json;base64," + _to_base64(b"{}")

    sentinel = object()
    payload = {
        "plain": "keep",
        "string": data_txt,
        "list": ["ok", data_png],
        "tuple": ("still", data_json),
        "set": {"anchor", data_png},
        "dict": {"inner": data_txt},
        "object": sentinel,
    }

    scrubbed = DataUrlUtil.scrub(payload)

    expected_marker = f"[{len(data_txt.encode())} bytes data url]"
    assert scrubbed["plain"] == "keep"
    assert scrubbed["string"] == expected_marker
    assert scrubbed["list"] == ["ok", f"[{len(data_png.encode())} bytes data url]"]
    assert scrubbed["tuple"] == ("still", f"[{len(data_json.encode())} bytes data url]")
    assert scrubbed["dict"] == {"inner": expected_marker}
    assert scrubbed["set"] == {"anchor", f"[{len(data_png.encode())} bytes data url]"}
    assert scrubbed["object"] is sentinel


def test_extract_writes_files_and_handles_edge_cases(tmp_path, monkeypatch):
    base_dir = tmp_path / "artifacts"

    jpeg_bytes = b"jpeg-bytes"
    png_bytes = b"png-bytes"
    bin_bytes = b"\x00\x01bin"
    text_payload = "Plain text!"

    jpeg_url = "data:image/jpeg;base64," + _to_base64(jpeg_bytes)
    png_url = "data:image/png;base64," + _to_base64(png_bytes)
    bin_url = "data:application/x-custom;base64," + _to_base64(bin_bytes)
    text_url = f"data:text/plain,{quote(text_payload)}"
    invalid_base64 = "data:text/plain;base64,a"
    missing_comma = "data:text/plain;base64"

    # Force mime_to_ext to rely on mimetypes for image/jpeg so the .jpe branch is exercised.
    monkeypatch.delitem(data_url_module._EXT, "image/jpeg", raising=False)

    payload = {
        "primary": jpeg_url,
        "collection": ["keep", jpeg_url],
        "tuple": ("still", png_url),
        "set": {"anchor", png_url},
        "bin": bin_url,
        "text": text_url,
        "invalid": invalid_base64,
        "missing_comma": missing_comma,
        "plain": "regular string",
        "number": 42,
    }

    transformed, next_idx = DataUrlUtil.extract(payload, base_path=base_dir, start_index=7)

    assert next_idx == 11  # 4 unique payloads -> names 007-010

    assert transformed["primary"] == "007.jpg"
    assert transformed["collection"] == ["keep", "007.jpg"]
    assert transformed["tuple"] == ("still", "008.png")
    assert transformed["set"] == {"anchor", "008.png"}
    assert transformed["bin"] == "009.bin"
    assert transformed["text"] == "010.txt"
    assert transformed["invalid"] == invalid_base64
    assert transformed["missing_comma"] == missing_comma
    assert transformed["plain"] == "regular string"
    assert transformed["number"] == 42

    written = sorted(p.name for p in base_dir.iterdir())
    assert written == ["007.jpg", "008.png", "009.bin", "010.txt"]
    assert (base_dir / "007.jpg").read_bytes() == jpeg_bytes
    assert (base_dir / "008.png").read_bytes() == png_bytes
    assert (base_dir / "009.bin").read_bytes() == bin_bytes
    assert (base_dir / "010.txt").read_bytes() == text_payload.encode()


def test_rebuild_reads_files_and_respects_strict_mode(tmp_path):
    base_dir = tmp_path / "files"
    png_bytes = b"png-data"
    png_url = "data:image/png;base64," + _to_base64(png_bytes)

    extracted, _ = DataUrlUtil.extract({"image": png_url}, base_path=base_dir)
    image_name = extracted["image"]

    (base_dir / "002.weird").write_bytes(b"binary")
    (base_dir / "003.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")

    payload = {
        "image": image_name,
        "list": [image_name, "002.weird", "003.csv", "999.png"],
        "tuple": (image_name,),
        "set": {image_name, "note.txt"},
        "missing": "999.png",
        "nonmatching": "note.txt",
        "number": 7,
    }

    rebuilt = DataUrlUtil.rebuild(payload, base_path=base_dir)

    assert rebuilt["image"].startswith("data:image/png;base64,")
    assert rebuilt["tuple"] == (rebuilt["image"],)
    assert rebuilt["set"] == {rebuilt["image"], "note.txt"}

    list_entry = rebuilt["list"]
    assert list_entry[0] == rebuilt["image"]
    assert list_entry[1].startswith("data:application/octet-stream;base64,")
    assert list_entry[2].startswith("data:text/csv;base64,")
    assert list_entry[3] == "999.png"
    assert rebuilt["missing"] == "999.png"
    assert rebuilt["nonmatching"] == "note.txt"
    assert rebuilt["number"] == 7

    with pytest.raises(KeyError):
        DataUrlUtil.rebuild("999.png", base_path=base_dir, strict=True)
