"""What the file system says changed while a command ran: the effect receipt for RUN_COMMAND.

`changed` (actions._tree_state) already diffs a stat snapshot of the work root before and after a
command. It is honest but blind in two ways: a file written and deleted inside the command leaves no
trace in it, and a tree above 5000 files is not watched at all. VERIFICATION-AUDIT asked for a
"command effect receipt" that comes from the volume rather than from DRock's own bookkeeping, and
research/deep-research-2026-09-22.md §F10 named the NTFS change journal as that receipt.

Measured on this machine (Windows 11 Home 26200, standard user, 2026-09-22):

    fsutil usn queryjournal C:            works unelevated: "Next Usn : 0x00000001dbebefa0"
    fsutil usn readjournal C: startusn=…  "Error 5: Access is denied."

So the journal can be STAMPED before a command by anyone, but READ only by an administrator, and
DRock never runs elevated (README §14: no escalation). Two paths therefore live here:

  * the journal path, used when `readjournal` works (an elevated shell, or a future build): the USN
    is recorded before the command and the records after it are parsed;
  * the watcher path, used everywhere else: a ReadDirectoryChangesW subscription on the work root,
    started before the command and drained after it, through ctypes and nothing else. It is the
    same kernel notification the journal is built from, delivered to a process instead of a log,
    and it sees a file that was created and deleted again — which the stat diff cannot.

Both produce the same shape, a list of {"path": <relative>, "reasons": [...]} entries under the work
root, so the receipt reads the same whichever path filled it. Neither replaces `changed`: the stat
diff stays, and this list is additional evidence next to it. An empty list is a fact ("the kernel
reported nothing under the work directory"); None means the receipt could not be taken at all (no
NTFS, a POSIX host, the notification buffer overflowed), which is "unknown", not "nothing".

The journal path has one limit worth knowing: a USN record carries a file NAME and a parent file id,
not a path, and resolving the id is another elevated call. Records are therefore kept only when a
file of that name exists under the work root after the command (or existed before it, per the stat
snapshot the caller hands in), and the reason flags say what happened to it.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import threading
import time

_WINDOWS = os.name == "nt"
MAX_ENTRIES = 50   # a receipt names files; a build tree's thousands of objects are a count, not a list

# ---------- the journal path (fsutil) ----------

_NEXT_USN = re.compile(r"Next Usn\s*:\s*(0x[0-9a-fA-F]+)")
# fsutil's record dump: one "Key : value" line per field, records separated by a blank line.
_FIELD = re.compile(r"^\s*([A-Za-z ]+?)\s*:\s*(.*?)\s*$")
# "Reason : 0x80000102: Data extend | File create | Close" -> ["data extend", "file create", "close"]
_REASON_WORDS = {
    "data overwrite": "modify", "data extend": "modify", "data truncation": "modify",
    "named data overwrite": "modify", "named data extend": "modify", "named data truncation": "modify",
    "file create": "create", "file delete": "delete", "ea change": "attributes",
    "security change": "attributes", "rename old name": "rename-from", "rename new name": "rename-to",
    "indexable change": "attributes", "basic info change": "attributes", "hard link change": "link",
    "compression change": "attributes", "encryption change": "attributes", "object id change": "attributes",
    "reparse point change": "attributes", "stream change": "modify", "close": "close",
}


def volume_of(path: str) -> str | None:
    """"C:" for a path on a drive letter, else None (a UNC share has no journal DRock can stamp)."""
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    return drive if re.fullmatch(r"[A-Za-z]:", drive or "") else None


def _fsutil(args: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(["fsutil", "usn", *args], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def next_usn(volume: str) -> int | None:
    """The volume's next USN, readable without elevation (measured above); None off NTFS."""
    if not _WINDOWS:
        return None
    rc, out = _fsutil(["queryjournal", volume])
    m = _NEXT_USN.search(out) if rc == 0 else None
    return int(m.group(1), 16) if m else None


def parse_readjournal(text: str) -> list[dict]:
    """The records in `fsutil usn readjournal` output: [{"name", "reasons", "usn"}], in order.

    Field names are matched case-insensitively and by their leading words, because fsutil's dump
    has changed its spacing between Windows releases and the record fields (Usn, File name, Reason)
    have not. Reasons are mapped to the short flags the watcher path uses, so the two receipts
    read the same; a flag fsutil prints that this table does not know is kept verbatim."""
    records, cur = [], {}
    for line in text.replace("\r", "").split("\n") + [""]:
        if not line.strip():
            if cur.get("name"):
                records.append(cur)
            cur = {}
            continue
        m = _FIELD.match(line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2)
        if key == "usn":
            try:
                cur["usn"] = int(val, 16)
            except ValueError:
                pass
        elif key == "file name":
            cur["name"] = val
        elif key == "reason":
            words = [w.strip().lower() for w in val.split(":", 1)[-1].split("|")]
            cur["reasons"] = sorted({_REASON_WORDS.get(w, w) for w in words if w} - {"close"}) or ["close"]
    return records


def read_records(volume: str, start_usn: int) -> list[dict] | None:
    """Records written since `start_usn`, or None when the journal cannot be read (Error 5 here)."""
    if not _WINDOWS:
        return None
    rc, out = _fsutil(["readjournal", volume, f"startusn={start_usn:#x}"], timeout=60)
    if rc != 0:
        return None
    return parse_readjournal(out)


_JOURNAL_READABLE: dict[str, bool] = {}


def journal_readable(volume: str) -> bool:
    """Can this process read the volume's journal? Probed once per volume per process: on a standard
    user's desktop the answer is no every time, and a command should not pay a failing fsutil twice."""
    if volume not in _JOURNAL_READABLE:
        usn = next_usn(volume)
        _JOURNAL_READABLE[volume] = usn is not None and read_records(volume, usn) is not None
    return _JOURNAL_READABLE[volume]


def entries_from_records(records: list[dict], root: str, before: dict | None = None) -> list[dict]:
    """Journal records as receipt entries, kept only for names that are (or were) under `root`."""
    names: dict[str, list[str]] = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")
            names.setdefault(os.path.normcase(f), []).append(rel)
    for rel in (before or {}):
        names.setdefault(os.path.normcase(rel.rsplit("/", 1)[-1]), []).append(rel)
    out: dict[str, set[str]] = {}
    for r in records:
        for rel in names.get(os.path.normcase(r.get("name", "")), ()):
            out.setdefault(rel, set()).update(r.get("reasons") or ())
    return _cap([{"path": p, "reasons": sorted(f)} for p, f in out.items()])


# ---------- the watcher path (ReadDirectoryChangesW) ----------

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".venv", "venv"}
_ACTIONS = {1: "create", 2: "delete", 3: "modify", 4: "rename-from", 5: "rename-to"}
_FILTER = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x40 | 0x100   # file/dir name, attributes, size, last write, creation, security
_BUF = 256 * 1024  # 256 KB: a notification is ~20 bytes + the name; ~6000 events before overflow
_FILE_LIST_DIRECTORY = 0x1
_SHARE_ALL = 0x1 | 0x2 | 0x4
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS_OVERLAPPED = 0x02000000 | 0x40000000
_INVALID = ctypes.c_void_p(-1).value
_ERROR_IO_PENDING, _ERROR_OPERATION_ABORTED, _ERROR_NOTIFY_ENUM_DIR = 997, 995, 1022


class _Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                ("Offset", ctypes.c_uint32), ("OffsetHigh", ctypes.c_uint32), ("hEvent", ctypes.c_void_p)]


class Watcher:
    """Every change the kernel reports under `root` between start() and stop().

    Overlapped I/O with an event, so stop() can cancel the pending read instead of waiting for a
    change that never comes. A read that returns zero bytes is the kernel saying its buffer
    overflowed (ERROR_NOTIFY_ENUM_DIR): the receipt is then unknown, never partial."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.events: list[tuple[int, str]] = []
        self.overflow = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._ready = threading.Event()

    def start(self) -> bool:
        if not _WINDOWS or not os.path.isdir(self.root):
            return False
        self._thread = threading.Thread(target=self._run, name="drock-journal", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self._handle is not None and self.error is None

    def _run(self) -> None:
        k32 = ctypes.windll.kernel32
        k32.CreateFileW.restype = ctypes.c_void_p
        k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        k32.CreateEventW.restype = ctypes.c_void_p
        k32.ReadDirectoryChangesW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                              ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.POINTER(_Overlapped), ctypes.c_void_p]
        k32.GetOverlappedResult.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Overlapped), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        k32.CancelIoEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Overlapped)]
        k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.CreateFileW(self.root, _FILE_LIST_DIRECTORY, _SHARE_ALL, None, _OPEN_EXISTING,
                            _BACKUP_SEMANTICS_OVERLAPPED, None)
        if h is None or h == _INVALID:
            self.error = f"CreateFileW failed ({ctypes.get_last_error() or k32.GetLastError()})"
            self._ready.set()
            return
        self._handle = h
        ev = k32.CreateEventW(None, True, False, None)
        buf = ctypes.create_string_buffer(_BUF)
        ov = _Overlapped()
        ov.hEvent = ev
        got = ctypes.c_uint32(0)
        first = True
        try:
            while not self._stop.is_set():
                k32.ResetEvent(ev)
                ok = k32.ReadDirectoryChangesW(h, buf, _BUF, True, _FILTER, ctypes.byref(got), ctypes.byref(ov), None)
                if not ok and k32.GetLastError() != _ERROR_IO_PENDING:
                    self.error = f"ReadDirectoryChangesW failed ({k32.GetLastError()})"
                    break
                if first:
                    first = False
                    self._ready.set()   # the subscription is live: the command may start now
                while not self._stop.is_set():
                    if k32.WaitForSingleObject(ev, 50) == 0:
                        break
                if self._stop.is_set():
                    k32.CancelIoEx(h, ctypes.byref(ov))
                    k32.WaitForSingleObject(ev, 1000)
                if not k32.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), False):
                    err = k32.GetLastError()
                    if err == _ERROR_NOTIFY_ENUM_DIR:
                        self.overflow = True
                    elif err != _ERROR_OPERATION_ABORTED:
                        self.error = f"GetOverlappedResult failed ({err})"
                    break
                n = got.value
                if n == 0:
                    self.overflow = True   # the documented overflow signal: too many changes to list
                    break
                self._parse(buf.raw[:n])
        finally:
            k32.CloseHandle(h)
            k32.CloseHandle(ev)
            self._handle = None
            self._ready.set()

    def _parse(self, raw: bytes) -> None:
        off = 0
        while True:
            nxt, action, nlen = (int.from_bytes(raw[off + i:off + i + 4], "little") for i in (0, 4, 8))
            name = raw[off + 12:off + 12 + nlen].decode("utf-16-le", "replace")
            self.events.append((action, name))
            if not nxt:
                break
            off += nxt

    def stop(self, grace: float = 0.15) -> list[dict] | None:
        """The receipt: entries under root, or None when the watcher could not vouch for the period.
        `grace` lets the last notifications of a command that just exited arrive before the read is
        cancelled; the kernel queues them, it does not drop them, but delivery is asynchronous."""
        if self._thread is None:
            return None
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and not self.overflow and not self.error:
            time.sleep(0.02)
        self._stop.set()
        self._thread.join(3.0)
        if self.overflow or self.error or self._thread.is_alive():
            return None
        return self.entries()

    def entries(self) -> list[dict]:
        out: dict[str, set[str]] = {}
        for action, name in self.events:
            rel = name.replace("\\", "/")
            parts = rel.split("/")
            if any(p in SKIP_DIRS for p in parts) or parts[-1].endswith((".pyc", ".pyo")):
                continue   # the same noise the stat diff leaves out, so the two receipts agree on scope
                           # (the __pycache__ directory being created is as much noise as the .pyc inside it)
            out.setdefault(rel, set()).add(_ACTIONS.get(action, f"action-{action}"))
        return _cap([{"path": p, "reasons": sorted(f)} for p, f in out.items()])


def _cap(entries: list[dict]) -> list[dict]:
    entries.sort(key=lambda e: e["path"].lower())
    if len(entries) > MAX_ENTRIES:
        more = len(entries) - MAX_ENTRIES
        entries = entries[:MAX_ENTRIES] + [{"path": f"…{more} more", "reasons": []}]
    return entries


# ---------- the receipt: one object, whichever path fills it ----------

class Receipt:
    """Start before the command, stop after it; `entries` is the list or None, `source` says which
    path produced it ("journal", "watcher") or why none did."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.source = "none"
        self.entries: list[dict] | None = None
        self._usn: int | None = None
        self._volume = volume_of(self.root) if _WINDOWS else None
        self._watcher: Watcher | None = None

    def start(self) -> "Receipt":
        if not _WINDOWS:
            self.source = "not windows"
            return self
        if self._volume and journal_readable(self._volume):
            self._usn = next_usn(self._volume)
            if self._usn is not None:
                self.source = "journal"
                return self
        self._watcher = Watcher(self.root)
        if self._watcher.start():
            self.source = "watcher"
        else:
            self.source = f"watcher unavailable ({self._watcher.error or 'no directory'})"
            self._watcher = None
        return self

    def stop(self, before: dict | None = None) -> list[dict] | None:
        if self.source == "journal" and self._usn is not None and self._volume:
            recs = read_records(self._volume, self._usn)
            self.entries = None if recs is None else entries_from_records(recs, self.root, before)
        elif self._watcher is not None:
            self.entries = self._watcher.stop()
            if self.entries is None:
                self.source = "watcher overflowed" if self._watcher.overflow else f"watcher failed ({self._watcher.error})"
        return self.entries


def describe(entries: list[dict] | None, source: str = "") -> str:
    """The verification clause: what the kernel reported, counted honestly, credited to the observer that saw it.
    Unelevated, `fsutil usn readjournal` fails with Error 5 and the watcher stands in (Receipt.start), so the
    clause said "the volume journal shows" for changes the volume journal never reported (2026-10-13). An
    unnamed source makes no claim: the count is the receipt's, and the receipt knows which path produced it."""
    if entries is None:
        return ""
    real = [e for e in entries if not e["path"].startswith("…")]
    more = sum(int(m.group(1)) for e in entries if (m := re.match(r"…(\d+) more", e["path"])))
    n = len(real) + more
    if source == "journal":
        return f"; the volume journal shows {n} file(s) changed under the work directory"
    if source == "watcher":
        return f"; the directory watcher saw {n} file(s) change under the work directory"
    return f"; {n} file(s) changed under the work directory"
