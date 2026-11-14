"""One ingest per match directory: a lock file with the process id of the running ingest.

The lock is created atomically by the ingest process itself and removed when it ends. A lock
left behind by a process that no longer exists (killed, power cut) is taken over.
"""

import ctypes
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from aicoach.errors import IngestError

LOCK_FILE = "ingest.lock"
_STILL_ACTIVE = 259  # GetExitCodeProcess value of a running process (Windows)
_QUERY_LIMITED_INFORMATION = 0x1000

AliveCheck = Callable[[int], bool]


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running (the id may have been reused by now)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) would send CTRL_C_EVENT on Windows, so ask the kernel instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and (
                code.value == _STILL_ACTIVE
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, it is only not ours
    return True


def read_lock_pid(match_dir: Path) -> int | None:
    """The process id written in the lock file; None without a readable lock."""
    try:
        text = (match_dir / LOCK_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def running_pid(match_dir: Path, is_alive: AliveCheck = pid_alive) -> int | None:
    """The id of the ingest process now working on this directory, if there is one."""
    pid = read_lock_pid(match_dir)
    return pid if pid is not None and is_alive(pid) else None


def _create(path: Path, pid: int) -> bool:
    """Create the lock file with `pid`; False when it already exists."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(pid))
    return True


@contextmanager
def ingest_lock(
    match_dir: Path, pid: int | None = None, is_alive: AliveCheck = pid_alive
) -> Iterator[None]:
    """Hold the lock of `match_dir` for the duration of the block.

    Raises:
        IngestError: Another live process holds the lock, or the lock cannot be created.
    """
    own = os.getpid() if pid is None else pid
    path = match_dir / LOCK_FILE
    try:
        created = _create(path, own)
        if not created:
            other = running_pid(match_dir, is_alive)
            if other is not None and other != own:
                msg = (
                    f"this match is already being processed (process {other}); wait for it to "
                    f"finish, or delete {path} if no such process is running"
                )
                raise IngestError(msg)
            path.unlink(missing_ok=True)  # left behind by a dead process
            created = _create(path, own)
    except OSError as err:
        msg = f"cannot create the lock {path}: {err}"
        raise IngestError(msg) from err
    if not created:
        msg = f"another ingest took the lock {path} at the same moment"
        raise IngestError(msg)
    try:
        yield
    finally:
        if read_lock_pid(match_dir) == own:
            path.unlink(missing_ok=True)
