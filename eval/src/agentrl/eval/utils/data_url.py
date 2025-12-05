from __future__ import annotations

import re
from base64 import b64decode, b64encode
from hashlib import sha256
from mimetypes import guess_extension
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import unquote_to_bytes

try:
    from mimetypes import guess_file_type
except ImportError:  # pragma: no cover
    # python < 3.13 fallback
    from mimetypes import guess_type as guess_file_type


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
    def parse(value: Any) -> Optional[tuple[str, list[str], bytes, bool]]:
        """Parse a data URL into (mime, params, payload, is_base64)."""
        if not isinstance(value, str) or not value.startswith('data:'):
            return None
        try:
            header, data_part = value.split(',', 1)
        except ValueError:
            return None

        meta = header[5:]
        mime = 'text/plain'
        params: list[str] = []

        if meta:
            parts = [p.strip() for p in meta.split(';') if p.strip()]
            if parts and '/' in parts[0]:
                mime = parts[0].lower()
                params = parts[1:]
            else:
                params = parts

        is_base64 = any(p.strip().lower() == 'base64' for p in params)
        try:
            payload = b64decode(data_part, validate=False) if is_base64 else unquote_to_bytes(data_part)
        except Exception:
            return None

        return mime, params, payload, is_base64

    @staticmethod
    def build(mime: str, payload: bytes, params: Optional[Iterable[str]] = None) -> str:
        """Build a base64 data URL preserving non-base64 parameters."""
        extras = [p for p in (params or []) if p.strip().lower() != 'base64']
        extras.append('base64')
        param_section = ';'.join(extras)
        mime = mime.lower() if mime else ''
        if mime and param_section:
            header = f'{mime};{param_section}'
        elif mime:
            header = mime
        elif param_section:
            header = f';{param_section}'
        else:
            header = ''
        encoded = b64encode(payload).decode('ascii')
        return f'data:{header},{encoded}'

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
                parsed = DataUrlUtil.parse(o)
                return ensure_file(parsed[0], parsed[2]) if parsed else o
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
            return DataUrlUtil.build(mime, payload_path.read_bytes())

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
