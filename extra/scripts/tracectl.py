#!/usr/bin/env python3

import json
import shutil
import sys
import os
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Global lock to prevent interleaved prompts/prints across threads
_IO_LOCK = threading.Lock()

def log(msg: str) -> None:
    with _IO_LOCK:
        print(msg, flush=True)

def prompt_yes_no(question: str) -> bool:
    with _IO_LOCK:
        if not sys.stdin.isatty():
            return False
        resp = input(f"{question} [y/N] ").strip().lower()
        return resp == "y"

def ensure_clean(path: Path, assume_yes: bool = False):
    if path.exists():
        do_override = assume_yes or prompt_yes_no(f"Path {path} exists, override it?")
        if not do_override:
            log("Aborting.")
            sys.exit(1)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def describe_entry(path: Path) -> str:
    """Return 'dir', 'zip', 'file', or 'unknown' for the given path."""
    if path.is_dir():
        return 'dir'
    if path.is_file():
        if path.suffix.lower() == '.zip':
            return 'zip'
        return 'file'
    return 'unknown'


def copy_entry(src: Path, output: Path, assume_yes: bool) -> Optional[Path]:
    """Copy a trace payload to the output directory.

    Returns the destination path (file or directory) if copied, otherwise None.
    """
    entry_type = describe_entry(src)
    dest = output / src.name

    if entry_type == 'dir':
        ensure_clean(dest, assume_yes=assume_yes)
        shutil.copytree(src, dest, symlinks=True, ignore_dangling_symlinks=True)
        return dest

    if entry_type == 'file':
        ensure_clean(dest, assume_yes=assume_yes)
        shutil.copy2(src, dest)
        return dest

    if entry_type == 'zip':
        ensure_clean(dest, assume_yes=assume_yes)
        shutil.copy2(src, dest)
        extract_dir = output / src.stem
        ensure_clean(extract_dir, assume_yes=assume_yes)
        log(f'extracting {dest} -> {extract_dir}')
        shutil.unpack_archive(dest, extract_dir)
        dest.unlink()
        return extract_dir

    log(f"unsupported entry type for {src} ({entry_type})")
    return None


def parse_session_ids(input_args: List[str]) -> List[int]:
    result: List[int] = []
    for arg in input_args:
        p = Path(arg)
        if p.is_file():
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            for key in ('session_id', 'sid'):
                                if data.get(key) is not None:
                                    line = str(data[key])
                                    break
                    except ValueError:
                        pass
                    if line.isdigit():
                        result.append(int(line))
        elif arg.isdigit():
            result.append(int(arg))
    return result

def find_by_session(root: Path, session_id: int) -> Optional[Path]:
    a = f'{(session_id >> 8) & 0xFF:02x}'
    b = f'{(session_id >> 0) & 0xFF:02x}'
    path = root / a / b / f'session_{session_id}'
    if not path.exists():
        return None
    return path.resolve(strict=True)

def process_one(idx_total_root_output_overwrite: Tuple[int, int, Path, Path, bool, int]) -> None:
    """
    Worker function for a single session id.
    """
    idx, total, root, output, assume_yes, session_id = idx_total_root_output_overwrite
    label = f'[{idx}/{total}]'

    path = find_by_session(root, session_id)
    if path is None:
        log(f'{label} session {session_id} not found')
        return
    log(f'{label} copying from {path}')

    dest = copy_entry(path, output, assume_yes=assume_yes)
    if dest is None:
        log(f'{label} skipped {path}')
    elif dest.is_dir():
        log(f'{label} copied directory to {dest}')
    else:
        log(f'{label} copied file to {dest}')

def guess_default_jobs(n_items: int) -> int:
    # IO-bound: allow higher concurrency, but keep it reasonable.
    # Heuristic: up to 32, but not more than number of items.
    cpu = os.cpu_count() or 4
    default = min(max(4, cpu * 5), 32, max(1, n_items))
    return default

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--store', required=True, help='root of the traces store')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-j', '--jobs', type=int, help='number of concurrent workers (default: heuristic)')
    parser.add_argument('-y', '--yes', action='store_true', help='assume Yes for all prompts (non-interactive)')
    parser.add_argument('input', nargs='+', help='session id or result file to query')
    args = parser.parse_args()

    session_ids = parse_session_ids(args.input)
    if not session_ids:
        print(f'No valid session ids found in {args.input}')
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    root = Path(args.store) / 'by-session'
    total = len(session_ids)

    jobs = args.jobs if args.jobs and args.jobs > 0 else guess_default_jobs(total)
    log(f'Processing {total} session(s) with {jobs} worker(s)...')

    # Prepare task tuples so labels stay deterministic per task
    tasks: List[Tuple[int, int, Path, Path, bool, int]] = [
        (i + 1, total, root, output, args.yes, sid) for i, sid in enumerate(session_ids)
    ]

    # Execute concurrently; handle Ctrl-C gracefully
    futures: List[Future] = []
    try:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="copy") as ex:
            for t in tasks:
                futures.append(ex.submit(process_one, t))
            for fut in as_completed(futures):
                # Propagate exceptions promptly
                fut.result()
    except KeyboardInterrupt:
        log("Interrupted. Attempting to cancel pending work...")
        for fut in futures:
            fut.cancel()
        # Best-effort: some threads may already be running and will finish soon.
        raise
    log("Done.")

if __name__ == '__main__':
    main()
