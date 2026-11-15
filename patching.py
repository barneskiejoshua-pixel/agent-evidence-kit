"""Line edits and unified diffs that keep the file's form.

"Forgiveness for form, never for substance": every edit here preserves what the file already is —
its line ending (CRLF or LF, by majority; a new line takes the file's ending; an existing line keeps
its own), whether it ends with a newline, and its encoding (utf-8, utf-8-sig with the BOM kept,
cp1252 / latin-1 when utf-8 fails) — and writes atomically (temp file in the same folder, then
os.replace). Line numbers are 1-based, as READ_FILE shows them.

    insert_lines(path, after_line, text)   -> {"inserted", "at", "lines_before", "lines_after", "diff"}
    delete_lines(path, start, end=None)    -> {"deleted", "removed", "lines_before", "lines_after", "diff"}
    apply_patch(patch_text, root, dry_run) -> {"ok", "files", "failed", "summary"}   (never raises)
    unified(before, after, path, n=3)      -> str

Pure Python, no git. apply_patch matches a hunk at its stated line first, then within +/-50 lines
(the offset is reported), tolerating whitespace-only differences in context ("fuzz"). A file whose
hunks do not all apply is left untouched. Patch paths are relative to root and may not escape it.
"""
from __future__ import annotations

import difflib
import os
import re
import tempfile
from pathlib import Path

MAX_OFFSET = 50
REMOVED_CAP = 2000


class StaleFile(Exception):
    """The file changed between the read this edit was computed from and the write.

    Every edit here is a read-modify-write of the whole file, so anything else that wrote that file
    in between loses its change -- and the edit still verifies, because the read-back matches what
    the edit meant to write. Verifying the writer's own bytes is not the same as verifying that
    nothing was destroyed to place them. Measured 2026-11-07: EDIT_FILE over a file rewritten
    mid-edit returned ok=True verified=True error=None with the other writer's content gone."""


def token(path) -> tuple | None:
    """The freshness token of a path: (size, mtime_ns), or None when it does not exist.

    Size and nanosecond mtime, not a hash: a hash costs a full extra read of every file DRock edits,
    and two different contents written in the same nanosecond at the same length is not a case this
    is defending against. The hash lives in actions._stat, for the receipts that want one."""
    try:
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)
    except (OSError, ValueError):
        return None

_BOM = b"\xef\xbb\xbf"
_LINE_RE = re.compile(r"(.*?)(\r\n|\n|\r|\Z)", re.S)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# --------------------------------------------------------------------------- form-preserving file model
class TextFile:
    """A file as (content, ending) line pairs plus the encoding and majority ending it was read with."""

    def __init__(self, path: Path, lines: list[list[str]], encoding: str, ending: str, read_at=None):
        self.path = path
        self.lines = lines          # [content, ending]; ending "" only on the final line
        self.encoding = encoding
        self.ending = ending        # the file's majority ending; new lines take it
        # The file as it was when these lines were read (patching.token). write() refuses if it moved.
        self.read_at = read_at

    # -- read / write ------------------------------------------------------------------------------
    @classmethod
    def read(cls, path: Path) -> "TextFile":
        at = token(path)
        raw = path.read_bytes() if path.exists() else b""
        text, encoding = decode(raw)
        lines = split_lines(text)
        return cls(path, lines, encoding, majority_ending(lines), read_at=at)

    @classmethod
    def empty(cls, path: Path, ending: str = "\n") -> "TextFile":
        return cls(path, [], "utf-8", ending)

    @property
    def trailing_newline(self) -> bool:
        return not self.lines or self.lines[-1][1] != ""

    @property
    def contents(self) -> list[str]:
        return [c for c, _ in self.lines]

    def text(self) -> str:
        return "".join(c + e for c, e in self.lines)

    def normalise(self, trailing: bool) -> None:
        """Every line but the last gets an ending; the last one gets the file's ending or none."""
        for i, pair in enumerate(self.lines):
            if i < len(self.lines) - 1:
                if not pair[1]:
                    pair[1] = self.ending
            else:
                pair[1] = self.ending if trailing else ""

    def fill_endings(self) -> None:
        """Every line but the last gets an ending if it lost one; the last line keeps its own (a "+"
        line carries the patch's say, a context line carries the file's)."""
        for pair in self.lines[:-1]:
            if not pair[1]:
                pair[1] = self.ending

    def write(self) -> None:
        atomic_write(self.path, self.text().encode(self.encoding), if_unchanged=self.read_at)


def decode(raw: bytes) -> tuple[str, str]:
    """(text, encoding): utf-8-sig when the BOM is there, utf-8, then cp1252, then latin-1."""
    if raw.startswith(_BOM):
        return raw[len(_BOM):].decode("utf-8", errors="replace"), "utf-8-sig"
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def split_lines(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for m in _LINE_RE.finditer(text):
        content, ending = m.group(1), m.group(2)
        if content == "" and ending == "":
            break
        out.append([content, ending])
    return out


def majority_ending(lines: list[list[str]]) -> str:
    counts = {"\r\n": 0, "\n": 0, "\r": 0}
    for _, e in lines:
        if e:
            counts[e] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else "\n"


def atomic_write(path: Path, data: bytes, if_unchanged: tuple | None = None) -> None:
    """Write `data` to `path` atomically. With `if_unchanged` (a token() from when the data this
    write is based on was read) the file is re-checked in the last moment before os.replace and a
    StaleFile is raised instead of overwriting a change that arrived in between. The check is as
    late as it can be and the window it leaves is the rename itself; nothing short of a lock closes
    that, and a lock on every edit is a worse trade on a single-machine agent than a refusal that
    names the problem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".drock-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if if_unchanged is not None and token(path) != if_unchanged:
            raise StaleFile(f"{path} changed while this edit was being prepared (it was "
                            f"{if_unchanged[0]} bytes at mtime {if_unchanged[1]}, it is now "
                            f"{(token(path) or ('gone', 0))[0]}); READ_FILE it again and redo the edit")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # The read-back INSERT_LINES, DELETE_LINES and APPLY_PATCH report as their verification: until the tribunal audit
    # (2026-10-12) they said "re-read after the write" and returned verified=True without reading anything.
    if path.read_bytes() != data:
        raise OSError(f"{path}: the file read back after the write does not hold what was written")


def unified(before: str, after: str, path: str, n: int = 3) -> str:
    """A unified diff of two texts, endings ignored, one line per row, LF-joined."""
    a = [c for c, _ in split_lines(before)]
    b = [c for c, _ in split_lines(after)]
    return "\n".join(difflib.unified_diff(a, b, "a/" + path, "b/" + path, n=n, lineterm=""))


def _new_lines(text: str) -> list[str]:
    """Split inserted text into contents; any \\r\\n or \\n inside is a line break; a final break
    does not add an empty line."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if t.endswith("\n"):
        t = t[:-1]
    return t.split("\n")


# --------------------------------------------------------------------------- line edits
def insert_lines(path, after_line: int, text: str) -> dict:
    """Insert text after line N (0 = top; N = line count = append). Returns the count, the first new
    line number, the before/after line counts and a short unified diff."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    tf = TextFile.read(p)
    n = len(tf.lines)
    if not isinstance(after_line, int) or after_line < 0 or after_line > n:
        raise ValueError(f"after_line must be between 0 and {n} (the file has {n} lines), not {after_line!r}")
    before = tf.text()
    trailing = tf.trailing_newline
    new = _new_lines(text)
    tf.lines[after_line:after_line] = [[c, tf.ending] for c in new]
    tf.normalise(trailing)
    after = tf.text()
    tf.write()
    return {"path": str(p), "inserted": len(new), "at": after_line + 1, "lines_before": n,
            "lines_after": len(tf.lines), "diff": unified(before, after, p.name, n=1)}


def delete_lines(path, start: int, end: int | None = None) -> dict:
    """Delete lines start..end inclusive (1-based; end defaults to start)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    tf = TextFile.read(p)
    n = len(tf.lines)
    if end is None:
        end = start
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end > n or end < start:
        raise ValueError(f"lines {start}..{end} are out of range: the file has {n} lines")
    before = tf.text()
    trailing = tf.trailing_newline
    removed = tf.lines[start - 1:end]
    removed_text = "".join(c + (e or "") for c, e in removed)
    del tf.lines[start - 1:end]
    tf.normalise(trailing)
    after = tf.text()
    tf.write()
    if len(removed_text) > REMOVED_CAP:
        removed_text = removed_text[:REMOVED_CAP] + f"... [{len(removed_text) - REMOVED_CAP} more chars]"
    return {"path": str(p), "deleted": len(removed), "removed": removed_text, "lines_before": n,
            "lines_after": len(tf.lines), "diff": unified(before, after, p.name, n=1)}


# --------------------------------------------------------------------------- unified diff parsing
class Hunk:
    def __init__(self, header: str, old_start: int, old_len: int, new_start: int, new_len: int):
        self.header = header
        self.old_start, self.old_len, self.new_start, self.new_len = old_start, old_len, new_start, new_len
        self.ops: list[list] = []     # [tag, content, no_newline]  tag in " -+"

    def old_lines(self) -> list[str]:
        return [c for t, c, _ in self.ops if t in " -"]


class FilePatch:
    def __init__(self):
        self.old_path: str | None = None
        self.new_path: str | None = None
        self.git_path: str | None = None
        self.new_file = False
        self.deleted_file = False
        self.hunks: list[Hunk] = []
        self.crlf = 0
        self.lf = 0

    @property
    def path(self) -> str | None:
        for cand in (self.new_path, self.old_path, self.git_path):
            if cand and cand != "/dev/null":
                return cand
        return None


def _strip_path(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) > 1:
        s = s[1:-1]
    s = s.split("\t")[0].strip()
    if s == "/dev/null":
        return s
    if s.startswith(("a/", "b/")):
        s = s[2:]
    return s


def parse_patch(patch_text: str) -> list[FilePatch]:
    """Unified diff text -> [FilePatch]. Raises ValueError when it is not a unified diff."""
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    lines = patch_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for raw in lines:
        crlf = raw.endswith("\r")
        line = raw[:-1] if crlf else raw
        if hunk is not None and (remaining_old > 0 or remaining_new > 0):
            if line.startswith("\\"):
                if hunk.ops:
                    hunk.ops[-1][2] = True
                continue
            tag = line[:1] if line else " "
            content = line[1:] if line else ""
            if tag == " ":
                remaining_old -= 1
                remaining_new -= 1
            elif tag == "-":
                remaining_old -= 1
            elif tag == "+":
                remaining_new -= 1
            else:
                raise ValueError(f"unexpected line inside hunk {hunk.header}: {line[:60]!r}")
            hunk.ops.append([tag, content, False])
            if cur is not None:
                cur.crlf += crlf
                cur.lf += not crlf
            continue
        if line.startswith("\\") and hunk is not None:
            if hunk.ops:
                hunk.ops[-1][2] = True
            continue
        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            hunk = None
            m = re.match(r'diff --git (?:a/)?(\S+|"[^"]+") (?:b/)?(\S+|"[^"]+")', line)
            if m:
                cur.git_path = _strip_path(m.group(2))
            continue
        if line.startswith("--- "):
            if cur is None or cur.old_path is not None or cur.hunks:
                cur = FilePatch()
                files.append(cur)
            hunk = None
            cur.old_path = _strip_path(line[4:])
            continue
        if line.startswith("+++ "):
            if cur is None:
                cur = FilePatch()
                files.append(cur)
            cur.new_path = _strip_path(line[4:])
            continue
        if cur is not None and line.startswith("new file mode"):
            cur.new_file = True
            continue
        if cur is not None and line.startswith("deleted file mode"):
            cur.deleted_file = True
            continue
        m = _HUNK_RE.match(line)
        if m:
            if cur is None:
                raise ValueError("hunk before any --- / +++ header")
            os_, ol, ns, nl = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
            hunk = Hunk(line.strip(), os_, 1 if ol is None else int(ol), ns, 1 if nl is None else int(nl))
            remaining_old, remaining_new = hunk.old_len, hunk.new_len
            cur.hunks.append(hunk)
            continue
        # anything else (index lines, commit text, blank separators) is ignored
    files = [f for f in files if f.hunks or f.new_file or f.deleted_file or f.old_path or f.new_path]
    if not files or not any(f.hunks for f in files):
        raise ValueError("not a unified diff (no --- / +++ headers with @@ hunks)")
    for f in files:
        if f.old_path == "/dev/null" or f.new_file:
            f.new_file = True
        if f.new_path == "/dev/null" or f.deleted_file:
            f.deleted_file = True
        if f.path is None:
            raise ValueError("a file section has no path")
    return files


# --------------------------------------------------------------------------- hunk application
def _match_at(have: list[str], want: list[str], pos: int, fuzzy: bool) -> bool:
    if pos < 0 or pos + len(want) > len(have):
        return False
    for i, w in enumerate(want):
        h = have[pos + i]
        if h == w:
            continue
        if fuzzy and h.strip() == w.strip():
            continue
        return False
    return True


def _find(have: list[str], want: list[str], expected: int) -> tuple[int | None, bool]:
    """(position, fuzz): the exact match nearest to expected within MAX_OFFSET, else the nearest
    whitespace-tolerant one."""
    for fuzzy in (False, True):
        for delta in range(0, MAX_OFFSET + 1):
            for pos in ((expected + delta, expected - delta) if delta else (expected,)):
                if _match_at(have, want, pos, fuzzy):
                    return pos, fuzzy
    return None, False


def _first_mismatch(have: list[str], want: list[str], pos: int) -> str:
    for i, w in enumerate(want):
        if pos + i >= len(have):
            return f"file ends before expected line {pos + i + 1}: {w!r}"
        if have[pos + i] != w:
            return f"line {pos + i + 1}: expected {w!r}, found {have[pos + i]!r}"
    return "no match"


def _apply_file(fp: FilePatch, target: Path) -> tuple[dict | None, dict | None, TextFile | None]:
    """(file report, failure, new TextFile) — the TextFile is None when the file is to be deleted."""
    rel = fp.path
    if fp.new_file:
        if target.exists() and target.stat().st_size:
            return None, {"path": rel, "hunk": fp.hunks[0].header if fp.hunks else "", "reason": "file already exists"}, None
        ending = "\r\n" if fp.crlf > fp.lf else "\n"
        tf = TextFile.empty(target, ending)
    else:
        if not target.is_file():
            return None, {"path": rel, "hunk": fp.hunks[0].header if fp.hunks else "", "reason": "file not found"}, None
        tf = TextFile.read(target)
    work = [list(pair) for pair in tf.lines]
    max_offset = 0
    fuzz_lines = 0
    for h in fp.hunks:
        have = [c for c, _ in work]
        want = h.old_lines()
        expected = h.new_start - 1 if h.new_len > 0 else h.new_start
        expected = max(0, min(expected, len(have)))
        pos, fuzzy = _find(have, want, expected)
        if pos is None:
            guess = max(0, min(h.old_start - 1, len(have)))
            return None, {"path": rel, "hunk": h.header, "reason": _first_mismatch(have, want, guess)}, None
        off = pos - expected
        if abs(off) > abs(max_offset):
            max_offset = off
        if fuzzy:
            fuzz_lines += sum(1 for i, w in enumerate(want) if have[pos + i] != w)
        replacement: list[list[str]] = []
        old_i = pos
        for tag, content, nonl in h.ops:
            if tag == " ":
                replacement.append([work[old_i][0], work[old_i][1]])   # a context line keeps its own ending
                old_i += 1
            elif tag == "-":
                old_i += 1
            else:
                replacement.append([content, "" if nonl else tf.ending])
        work[pos:pos + len(want)] = replacement
    if fp.deleted_file:
        if any(c for c, _ in work):
            return None, {"path": rel, "hunk": fp.hunks[-1].header, "reason": "patch deletes the file but leaves content"}, None
        report = {"path": rel, "hunks": len(fp.hunks), "applied": len(fp.hunks), "offset": max_offset,
                  "fuzz": fuzz_lines, "created": False, "deleted": True}
        return report, None, None
    new = TextFile(target, work, tf.encoding, tf.ending)
    new.fill_endings()
    report = {"path": rel, "hunks": len(fp.hunks), "applied": len(fp.hunks), "offset": max_offset,
              "fuzz": fuzz_lines, "created": fp.new_file, "deleted": False}
    return report, None, new


def apply_patch(patch_text: str, root, dry_run: bool = False) -> dict:
    """Apply a unified diff under root. Never raises for a bad patch."""
    root_p = Path(root).resolve()
    out: dict = {"ok": False, "files": [], "failed": [], "summary": "", "dry_run": dry_run}
    try:
        patches = parse_patch(patch_text)
    except ValueError as e:
        out["failed"].append({"path": "", "hunk": "", "reason": str(e)})
        out["summary"] = f"nothing patched: {e}"
        return out
    pending: list[tuple[FilePatch, Path, TextFile | None, dict]] = []
    for fp in patches:
        rel = fp.path or ""
        target = (root_p / rel)
        try:
            resolved = target.resolve()
            inside = resolved == root_p or root_p in resolved.parents
        except OSError:
            inside = False
        if not rel or not inside or Path(rel).is_absolute():
            out["failed"].append({"path": rel, "hunk": fp.hunks[0].header if fp.hunks else "",
                                  "reason": f"path escapes root: {rel!r}"})
            continue
        try:
            report, failure, new = _apply_file(fp, target)
        except (OSError, UnicodeError) as e:
            report, failure, new = None, {"path": rel, "hunk": fp.hunks[0].header if fp.hunks else "", "reason": f"{type(e).__name__}: {e}"}, None
        if failure:
            out["failed"].append(failure)
            continue
        pending.append((fp, target, new, report))
    if dry_run:
        out["files"] = [r for _, _, _, r in pending]
    elif not out["failed"]:
        # All files or none (audit 2026-10-10: a.txt was written while b.txt's hunk failed). A write that fails half
        # way puts back what was already written: the old bytes, or no file for one this patch created.
        done: list[tuple[Path, bytes | None]] = []
        for fp, target, new, report in pending:
            before = target.read_bytes() if target.exists() else None
            try:
                if report["deleted"]:
                    if target.exists():
                        target.unlink()
                else:
                    new.write()
            except OSError as e:
                out["failed"].append({"path": report["path"], "hunk": "", "reason": f"write failed: {e}"})
                for path, old in reversed(done):
                    try:
                        if old is None:
                            path.unlink(missing_ok=True)
                        else:
                            atomic_write(path, old)
                    except OSError as e2:
                        out["failed"].append({"path": str(path), "hunk": "", "reason": f"could not put back: {e2}"})
                out["files"] = []
                break
            done.append((target, before))
            out["files"].append(report)
    out["ok"] = not out["failed"] and bool(out["files"])
    parts = [f"{r['path']}: {r['applied']} hunk{'s' if r['applied'] != 1 else ''}"
             + (" (new)" if r["created"] else " (deleted)" if r["deleted"] else "")
             + (f", offset {r['offset']:+d}" if r["offset"] else "") + (f", fuzz {r['fuzz']}" if r["fuzz"] else "")
             for r in out["files"]]
    n = len(out["files"])
    summary = f"{n} file{'s' if n != 1 else ''} {'would be ' if dry_run else ''}patched"
    if not n and out["failed"] and not dry_run and pending:
        summary = "nothing patched (a patch changes all its files or none)"
    if parts:
        summary += " (" + "; ".join(parts) + ")"
    if out["failed"]:
        summary += f"; {len(out['failed'])} failed (" + "; ".join(
            f"{f['path'] or 'patch'}: {f['hunk'] + ': ' if f['hunk'] else ''}{f['reason']}" for f in out["failed"]) + ")"
    out["summary"] = summary
    return out
