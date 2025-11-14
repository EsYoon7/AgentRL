from __future__ import annotations

import re
from base64 import b64decode, b64encode
from hashlib import sha256
from mimetypes import guess_extension, guess_file_type
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote_to_bytes

_FILENAME_REGEX = re.compile(r'^\d{3,}\.[A-Za-z0-9._+-]+$')
_EXT = {
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/svg+xml': '.svg',
    'application/json': '.json',
    'text/plain': '.txt',
    'application/pdf': '.pdf',
    'audio/mpeg': '.mp3',
    'audio/wav': '.wav',
    'video/mp4': '.mp4'
}


class DataUrlUtil:

    @staticmethod
    def scrub(x: Any) -> Any:
        if isinstance(x, str):
            return f'[{len(x.encode())} bytes data url]' if x.startswith('data:') else x
        if isinstance(x, dict):
            return {k: DataUrlUtil.scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [DataUrlUtil.scrub(i) for i in x]
        if isinstance(x, tuple):
            return tuple(DataUrlUtil.scrub(i) for i in x)
        if isinstance(x, set):
            return {DataUrlUtil.scrub(i) for i in x}
        return x

    @staticmethod
    def extract(
        obj: Any,
        base_path: Path,
        start_index: int = 1,
        _seen: Optional[dict[str, str]] = None,
    ) -> tuple[Any, int]:
        """
        Recursively replace data: URLs with filenames, writing payloads under `base_dir`.
        Returns (transformed_obj, next_index).
        """

        seen = _seen or {}
        idx_box = [start_index]

        def mime_to_ext(mime: str) -> str:
            ext = _EXT.get(mime) or guess_extension(mime) or '.bin'
            return '.jpg' if ext == '.jpe' else ext

        def parse_data_url(s: str) -> Optional[tuple[str, bytes]]:
            if not isinstance(s, str) or not s.startswith('data:'):
                return None
            try:
                meta, data = s[5:].split(',', 1)
            except ValueError:
                return None
            mime, is_b64 = 'text/plain', False
            if meta:
                parts = meta.split(';')
                if '/' in parts[0]:
                    mime, parts = parts[0].lower(), parts[1:]
                is_b64 = any(p.strip().lower() == 'base64' for p in parts)
            try:
                payload = b64decode(data, validate=False) if is_b64 else unquote_to_bytes(data)
            except Exception:
                return None
            return mime, payload

        def ensure_file(mime: str, payload: bytes) -> str:
            h = sha256(payload).hexdigest()
            name = seen.get(h)
            if not name:
                base_path.mkdir(parents=True, exist_ok=True)
                name = f'{idx_box[0]:03d}{mime_to_ext(mime)}'
                idx_box[0] += 1
                seen[h] = name
                path = base_path / name
                path.write_bytes(payload)
            return name

        def walk(o: Any) -> Any:
            if isinstance(o, str):
                parsed = parse_data_url(o)
                return ensure_file(*parsed) if parsed else o
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v) for v in o]
            if isinstance(o, tuple):
                return tuple(walk(v) for v in o)
            if isinstance(o, set):
                return {walk(v) for v in o}
            return o

        transformed = walk(obj)
        return transformed, idx_box[0]

    @staticmethod
    def rebuild(
        obj: Any,
        base_path: Path,
        strict: bool = False,
    ) -> Any:
        """
        Replace numbered filenames (e.g., "001.jpg") with base64 data URLs by reading `base_dir`.
        If `strict=True`, missing filenames raise KeyError; otherwise they are left unchanged.
        """

        def ext_to_mime(name: str) -> str:
            # Try known map first, else mimetypes; default to octet-stream
            ext = '.' + name.rsplit('.', 1)[-1].lower() if '.' in name else ''
            # reverse lookup from overrides
            for m, e in _EXT.items():
                if e == ext: return m
            mime, _ = guess_file_type('x' + ext)  # any name with that ext
            return (mime or 'application/octet-stream').lower()

        def to_data_url(name: str) -> str:
            payload_path = base_path / name
            if not payload_path.is_file():
                if strict: raise KeyError(f'Missing file bytes for "{name}"')
                return name
            mime = ext_to_mime(name)
            b64 = b64encode(payload_path.read_bytes()).decode('ascii')
            return f'data:{mime};base64,{b64}'

        def walk(o: Any) -> Any:
            if isinstance(o, str) and _FILENAME_REGEX.match(o):
                return to_data_url(o)
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v) for v in o]
            if isinstance(o, tuple):
                return tuple(walk(v) for v in o)
            if isinstance(o, set):
                return {walk(v) for v in o}
            return o

        return walk(obj)
