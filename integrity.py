"""Verification integrity: a green run counts only when it ran the program the task designated.

The invariant (BUILD-PLAN, Verifier Integrity v1, 2026-09-26):

    A task may only be marked successful when the verification run executes the program/artifact the
    task designated, or a declared dependency needed to establish correctness (the project's own tests).
    A file created during the task, a copy, a stub, a shell builtin or an unrelated executable cannot
    stand in for it. When the executed artifact cannot be identified, the run does not verify anything.

Found in benchmark run abc2: arm A ran a one-line test file the model had just created
(`print('test')`, exit 0) and arm B wrote `echo import csv > nested/report.py`; the gate accepted both as
the run that proved the fix. Nothing in the gate knew WHICH program a green run had executed.

Three pieces, all deterministic:

1. `Snapshot` - the workspace's files at task start (the executor takes it per task), so "existed before
   the task" and "created during the task" are facts, not guesses.
2. `executed(...)` - for a RUN_COMMAND / RUN_TESTS, the artifacts the command actually executed: a script
   given to an interpreter, a binary, a test discovery over a folder, a build (sources -> output), or
   nothing (a builtin such as echo). Every record carries its path and the two flags above.
3. `designated(goal, root)` and `verdict(goal, trace, root)` - what the task names as the thing to run,
   and whether the trace holds a green run of it after the last code change. The gate (expect.py) and the
   receipt (agent.py) both read the verdict; the model never defines its own success condition.
"""
from __future__ import annotations

import os
import re
import shlex
import time
from pathlib import Path

CODE_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".java", ".go",
            ".rs", ".rb", ".php", ".ps1", ".sh", ".bat", ".cmd", ".cs", ".kt", ".swift", ".lua", ".pl", ".r")
RUNNABLE_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".ps1", ".sh", ".bat", ".cmd", ".exe", ".rb", ".php", ".pl", ".lua",
                ".jar", ".r")
SOURCE_EXT = (".cpp", ".cc", ".cxx", ".c", ".java", ".go", ".rs", ".cs", ".kt", ".swift", ".ts")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}
SNAPSHOT_LIMIT = 20000

_BUILTINS = {"echo", "type", "cat", "set", "cd", "chdir", "dir", "ls", "copy", "cp", "del", "rm", "move", "mv", "mkdir",
             "md", "rmdir", "rd", "rem", "write-host", "write-output", "out-file", "set-content", "add-content",
             "get-content", "new-item", "remove-item", "copy-item", "move-item", "printf", "touch", "exit", "cls",
             "clear", "pwd", "start", "timeout", "sleep", "true", "false", "call", "pushd", "popd", "findstr", "grep",
             "where", "which", "more", "sort", "tee", "head", "tail"}
_INTERPRETERS = {"python", "python3", "py", "pythonw", "node", "deno", "ts-node", "tsx", "bun", "ruby", "perl", "php",
                 "lua", "rscript", "bash", "sh", "zsh", "pwsh", "powershell", "cscript", "wscript", "java", "go"}
_TEST_RUNNERS = re.compile(r"^(?:pytest|py\.test|jest|mocha|vitest|ctest)$|^(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b|"
                           r"^cargo\s+test\b|^go\s+test\b|^deno\s+test\b|^dotnet\s+test\b|^mvn\s+test\b|^gradle\s+test\b|"
                           r"^make\s+(?:test|check)\b", re.I)
_ONE_WORD_RUNNERS = {"pytest", "py.test", "jest", "mocha", "vitest", "ctest"}
_COMPILERS = {"g++", "gcc", "clang", "clang++", "cc", "c++", "cl", "javac", "tsc", "rustc", "csc", "kotlinc", "swiftc"}
_TEST_FILE = re.compile(r"(?:^|[\\/])(?:test_[\w-]+\.py|[\w-]+_test\.(?:py|go)|[\w.-]+\.(?:test|spec)\.(?:js|ts|mjs|cjs|tsx)|"
                        r"Test[\w]+\.java|[\w-]+Tests?\.cs)$", re.I)


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


# ---------------------------------------------------------------------------------------------- snapshot
class Snapshot:
    """The workspace's files when the task started: normalised paths, plus when the snapshot was taken."""

    def __init__(self, root: str, limit: int = SNAPSHOT_LIMIT, budget: float = 1.5):
        # A budget in seconds as well as a file cap: from the home folder the walk took 13.7 s on this
        # machine's HDD (2026-09-26). Cut short, the snapshot says so and creation times decide instead.
        self.root = _norm(root)
        self.at = time.time()
        self.files: set[str] = set()
        self.truncated = False
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                if time.time() - self.at > budget:
                    self.truncated = True
                    return
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
                for f in filenames:
                    self.files.add(_norm(os.path.join(dirpath, f)))
                    if len(self.files) >= limit:
                        self.truncated = True
                        return
        except OSError:
            self.truncated = True

    def covers(self, path: str) -> bool:
        p = _norm(path)
        return p == self.root or p.startswith(self.root + os.sep)

    def existed(self, path: str) -> bool | None:
        """True / False when the snapshot can tell; None when the path is outside it or it was cut short
        and the file's creation time cannot decide either."""
        p = _norm(path)
        if p in self.files:
            return True
        if self.covers(p) and not self.truncated:
            return False
        try:   # outside the snapshot, or a truncated one: the file's own creation time (Windows st_ctime)
            if os.name == "nt":
                return os.stat(p).st_ctime < self.at - 1
        except OSError:
            return False
        return None


def existed_before(executor, path: str) -> bool | None:
    snap = getattr(executor, "snapshot_files", None)
    if snap is None:
        return None
    return snap.existed(path)


# ---------------------------------------------------------------------------------------------- executed
def _split_segments(cmd: str) -> list[str]:
    """Top-level command segments (&&, ||, ;, |), quotes respected."""
    out, cur, q, i = [], [], None, 0
    while i < len(cmd):
        ch = cmd[i]
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            cur.append(ch)
        elif cmd.startswith("&&", i) or cmd.startswith("||", i):
            out.append("".join(cur)); cur = []; i += 2; continue
        elif ch in ";|&":
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def _tokens(seg: str) -> list[str]:
    try:
        toks = shlex.split(seg, posix=False)
    except ValueError:
        toks = seg.split()
    return [t.strip("\"'") for t in toks]


def _stem(tok: str) -> str:
    base = os.path.basename(tok.replace("\\", "/")).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


def _resolve(tok: str, cwd: str, exts=("", ".exe", ".bat", ".cmd", ".ps1")) -> str | None:
    t = os.path.expanduser(tok)
    for ext in exts:
        p = t + ext if not t.lower().endswith(ext) or not ext else t
        full = p if os.path.isabs(p) else os.path.join(cwd, p)
        if os.path.isfile(full):
            return _norm(full)
    return None


def _record(executor, path: str, kind: str, **extra) -> dict:
    ex = existed_before(executor, path)
    built = (getattr(executor, "builds", {}) or {}).get(_norm(path))
    rec = {"path": path, "kind": kind, "existed_before": ex,
           "created_in_task": (ex is False) and os.path.exists(path), **extra}
    if built:
        rec["built_from"] = built["sources"]
        rec["built_at_step"] = built.get("step")
    return rec


def _test_files(folder: str, limit: int = 200) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if _TEST_FILE.search(f):
                out.append(_norm(os.path.join(dirpath, f)))
                if len(out) >= limit:
                    return out
    return out


def executed(executor, action: dict, ok: bool) -> list[dict]:
    """What a RUN_COMMAND / RUN_TESTS actually executed. Builds on success are remembered on the executor
    (output -> sources), so a later run of that output is known to be the program built from them."""
    name = action.get("action")
    cwd = str(executor._p(action["cwd"])) if isinstance(action.get("cwd"), str) else str(executor.cwd)
    if name == "RUN_TESTS":
        root = executor._p(action.get("path") or ".")
        folder = str(root.parent if root.is_file() else root)
        cmd = str(action.get("command") or "")
        if cmd:
            recs = _from_command(executor, cmd, folder, ok)
            if recs:
                return recs
        return [{"path": _norm(folder), "kind": "tests", "test_files": [
            {"path": f, "existed_before": existed_before(executor, f)} for f in _test_files(folder)]}]
    return _from_command(executor, str(action.get("command") or ""), cwd, ok)


def _from_command(executor, cmd: str, cwd: str, ok: bool) -> list[dict]:
    recs: list[dict] = []
    for seg in _split_segments(cmd):
        toks = [t for t in _tokens(seg) if t]
        if not toks:
            continue
        # redirections are not programs
        toks = [t for t in toks if not re.match(r"^\d?>>?|^<", t)]
        if not toks:
            continue
        head = _stem(toks[0])
        if head in ("cd", "chdir", "pushd") and len(toks) > 1:
            nxt = toks[1] if os.path.isabs(toks[1]) else os.path.join(cwd, toks[1])
            if os.path.isdir(nxt):
                cwd = nxt
            continue
        if head in _BUILTINS:
            recs.append({"path": None, "kind": "builtin", "command": head})
            continue
        # The one-word runners are anchored for a whole-string match above, so "pytest -q" or "pytest tests/x.py"
        # (three tokens joined) never matched them and was recorded as a bare tool with no test identity
        # (drockclaude 2026-09-26, feature-topwords): the head alone decides for those.
        if _TEST_RUNNERS.search(" ".join(toks[:3])) or head in _ONE_WORD_RUNNERS or (head in ("python", "python3", "py") and "-m" in toks
                                                            and toks[toks.index("-m") + 1:toks.index("-m") + 2] in (["pytest"], ["unittest"])):
            targets = [t for t in toks[1:] if not t.startswith("-") and t not in ("-m", "pytest", "unittest", "discover", "test", "run")
                       and not re.fullmatch(r"\d+|[a-z]+", t) or t.endswith(".py")]
            targets = [t for t in targets if t.lower() not in ("pytest", "unittest", "discover", "test", "run", "-q", "-v")]
            # "unittest test_new" names a module: test_new.py (or pkg/mod.py for pkg.mod); "::test_x" a test in a file
            files = [_resolve(t.split("::")[0], cwd, ("", ".py")) or _resolve(t.replace(".", os.sep), cwd, (".py",)) for t in targets]
            if targets and not all(files):
                recs.append({"path": None, "kind": "tests", "unresolved": [t for t, f in zip(targets, files) if not f]})
            elif files:
                recs += [_record(executor, f, "tests") for f in files]
            else:
                recs.append({"path": _norm(cwd), "kind": "tests", "test_files": [
                    {"path": f, "existed_before": existed_before(executor, f)} for f in _test_files(cwd)]})
            continue
        if head in _COMPILERS:
            srcs = [_resolve(t, cwd, ("",)) for t in toks[1:] if t.lower().endswith(SOURCE_EXT + (".c",))]
            srcs = [s for s in srcs if s]
            out = None
            for i, t in enumerate(toks):
                if t == "-o" and i + 1 < len(toks):
                    out = toks[i + 1]
                elif t.startswith("-o") and len(t) > 2:
                    out = t[2:]
                elif t.lower().startswith(("/fe", "-fe")):
                    out = t[3:].lstrip(":")
            if out is None and srcs and head in ("g++", "gcc", "clang", "clang++", "cc", "c++"):
                out = "a.exe" if os.name == "nt" else "a.out"
            elif out is None and srcs and head == "cl":
                out = os.path.splitext(os.path.basename(srcs[0]))[0] + ".exe"
            if out is not None:
                full = out if os.path.isabs(out) else os.path.join(cwd, out)
                if os.name == "nt" and not full.lower().endswith(".exe") and head != "javac":
                    full += ".exe"
                full = _norm(full)
                if ok:
                    executor.builds = getattr(executor, "builds", {}) or {}
                    executor.builds[full] = {"sources": srcs, "step": getattr(executor, "_step_hint", None)}
                recs.append({"path": full, "kind": "build", "sources": srcs})
            else:
                recs.append({"path": None, "kind": "build", "sources": srcs})
            continue
        if head in _INTERPRETERS:
            script = None
            args = toks[1:]
            if head in ("python", "python3", "py", "pythonw") and "-c" in args:
                recs.append({"path": None, "kind": "inline", "command": head})
                continue
            if head in ("pwsh", "powershell") and "-command" in [a.lower() for a in args]:
                recs.append({"path": None, "kind": "inline", "command": head})
                continue
            for t in args:
                if t.startswith("-") or t.lower() in ("run", "-file", "--allow-all", "-jar"):
                    continue
                script = _resolve(t, cwd, ("",))
                if script or "." in os.path.basename(t):
                    break
            if script:
                recs.append(_record(executor, script, "script", interpreter=head))
            else:
                recs.append({"path": None, "kind": "interpreter", "command": head})
            continue
        exe = _resolve(toks[0], cwd)
        if exe:
            recs.append(_record(executor, exe, "script" if exe.endswith((".ps1", ".bat", ".cmd", ".sh")) else "binary"))
        else:
            recs.append({"path": None, "kind": "tool", "command": head})
    return recs


# ---------------------------------------------------------------------------------------------- designated
_NAMED_FILE = re.compile(r"(?<![\w.-])((?:[A-Za-z]:)?[\w./\\-]*?[\w-]+(?:" + "|".join(re.escape(e) for e in CODE_EXT + (".exe",)) +
                         r"))(?![\w-])", re.I)
# the project's tests as the proof: "the tests fail", "so the tests pass", "unit tests", "test suite", "tests in
# this folder" - not "test it" (a person asking for the program to be tried)
_TESTS_WORD = re.compile(r"\b(?:the|its|my|all|unit|failing|existing|project'?s)\s+tests?\b|\btests?\s+(?:pass|fail|are|is|in|for|still)\w*\b"
                         r"|\btest\s+(?:suite|files?)\b|\bspecs?\s+(?:pass|fail)\w*\b"
                         r"|\b(?:pytest|unittest|npm\s+test|yarn\s+test|pnpm\s+test|cargo\s+test|go\s+test|ctest|make\s+test|"
                         r"dotnet\s+test|mvn\s+test|gradle\s+test)\b[^.?!]{0,40}?\bpass(?:es)?\b", re.I)
# "implement X so that python -m pytest passes" asks for the same proof as a fix: the project's suite, green,
# after the change (drockclaude 2026-09-26, feature-topwords: a test file DRock wrote itself ran green while
# the named suite still failed). Not when the request asks for tests to be written: those are made in the task.
_PASS_INTENT = re.compile(r"\b(?:so\s+(?:that\s+)?|until\s+|make\s+|making\s+)[^.?!]{0,60}?\bpass(?:es)?\b", re.I)
_WRITE_TESTS = re.compile(r"\b(?:write|add|create|generate)\b[^.?!]{0,40}?\btests?\b", re.I)
_FIX_INTENT = re.compile(r"\b(?:fix|debug|repair|crash(?:es|ed|ing)?|broken|won'?t\s+(?:build|run|compile|work)|"
                         r"will\s+not\s+(?:build|run|compile|work)|does\s*n[o']?t\s+(?:work|run|build|compile|copy)|"
                         r"doesn'?t\s+(?:work|run|build|compile)|make\s+(?:it|them)\s+work)\b", re.I)
_CREATE_INTENT = re.compile(r"\b(?:create|write|make|build|scaffold|generate)\s+(?:a\s+|an\s+|the\s+)?(?:new\s+)?[\w.-]+\.\w+", re.I)


def designated(goal: str, root: str) -> dict:
    """What the task names as the thing whose run proves the fix: files it names that existed when the task
    started, and whether the project's tests are the proof ("the tests fail", "so the tests pass")."""
    files = []
    for m in _NAMED_FILE.finditer(goal or ""):
        raw = m.group(1).strip().strip("`'\".,;:)")
        cand = raw if os.path.isabs(raw) else os.path.join(root, raw)
        if not os.path.isfile(cand):
            base = os.path.basename(raw.replace("\\", "/"))
            cand = os.path.join(root, base)
        # a named file counts even when it is gone now (moved or deleted during the task): the request
        # designated THAT path, and a renamed copy is not it
        if os.path.isfile(cand) or not os.path.exists(cand):
            n = _norm(cand)
            if n not in files:
                files.append(n)
    tests = bool(_TESTS_WORD.search(goal or ""))
    return {"files": files, "tests": tests}


def applies(goal: str) -> bool:
    """The invariant governs goals that ask for a repair to be proven (fix/debug/crash/fails wording), and goals
    that make the project's tests the proof of new work ("so that pytest passes") without asking for tests
    to be written."""
    g = goal or ""
    return bool(_FIX_INTENT.search(g)) or bool(_PASS_INTENT.search(g) and _TESTS_WORD.search(g) and not _asks_for_tests(g))


def _asks_for_tests(goal: str) -> bool:
    """Does the request ask for tests to be written ("add tests for X"), not merely forbid it ("do not create
    new tests", "without adding tests")?"""
    for m in _WRITE_TESTS.finditer(goal):
        if not re.search(r"(?:\bnot|n't|\bnever|\bwithout|\bno)\s+(?:\w+\s+){0,2}$", goal[max(0, m.start() - 24):m.start()], re.I):
            return True
    return False


def _runs_designated(rec: dict, target: dict) -> str | None:
    """Why this executed record counts as running the designated target, or None."""
    files = target["files"]
    stems = {os.path.splitext(os.path.basename(f))[0].lower() for f in files}
    kind, path = rec.get("kind"), rec.get("path")
    if kind == "tests":
        # None (a trace recorded without a workspace snapshot) is not evidence of creation; False is
        pre = [t for t in rec.get("test_files", []) if t.get("existed_before") is not False] if "test_files" in rec else (
            [rec] if rec.get("existed_before") is not False else [])
        if not pre:
            return None
        if target["tests"]:
            return "the project's own tests"
        # a pre-existing test of a named program is a declared dependency of its correctness
        for t in pre:
            try:
                text = Path(t["path"]).read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            if any(re.search(rf"\b{re.escape(s)}\b", text) for s in stems):
                return f"a pre-existing test of {', '.join(sorted(stems))}"
        return None
    if not path:
        return None                       # a builtin, an inline -c, a tool on PATH: no artifact identity
    npath = _norm(path)
    built = [_norm(x) for x in rec.get("built_from") or []]
    if built:
        hits = [b for b in built if b in files]
        return ("the program built this task from " + ", ".join(os.path.basename(b) for b in hits)) if hits else None
    if npath in files and not (target["tests"] and not _TEST_FILE.search(npath)):
        # the path the request designated; rewriting it IS the fix. When the request makes the tests the
        # proof ("so the tests pass"), running the module under test is not a test run (smoke 2026-09-26:
        # `python pricing.py` exits 0 doing nothing, and counted)
        return "the named program"
    if rec.get("existed_before") is False or rec.get("created_in_task"):
        return None                       # made during the task: it cannot stand in for the designated target
    if kind == "script" and _TEST_FILE.search(npath):
        if target["tests"]:
            return "the project's own tests"
        try:   # a pre-existing test of a named program is a declared dependency of its correctness
            text = Path(npath).read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            text = ""
        if any(re.search(rf"\b{re.escape(s)}\b", text) for s in stems):
            return f"a pre-existing test of {', '.join(sorted(stems))}"
    return None


def verdict(goal: str, trace, root: str) -> dict | None:
    """The verification receipt for a repair goal, or None when the invariant does not apply.

    ok is True only when the last run that executed the designated target was green and came after the
    last successful code change. Every run considered is listed with what it executed and why it did or
    did not count."""
    if not applies(goal):
        return None
    target = designated(goal, root)
    if not target["files"] and not target["tests"]:
        return {"ok": None, "designated": target, "reason": "the request names no program or tests to verify"}
    runs, last_change, build_at = [], -1, {}
    _shim = _Shim(root)
    for i, s in enumerate(trace or []):
        a, r = s.get("action") or {}, s.get("result") or {}
        n = a.get("action")
        if n in ("RUN_COMMAND", "RUN_TESTS"):
            recs = r.get("executed")
            if recs is None:   # recorded before executors kept identity: parse the command, origin unknown
                recs = executed(_shim, a, bool(r.get("ok")))
            why = next((w for w in (_runs_designated(x, target) for x in recs if x.get("kind") != "build") if w), None)
            runs.append({"step": s.get("step", i + 1), "index": i, "command": a.get("command") or a.get("path") or "",
                         "ok": bool(r.get("ok")), "executed": recs, "counts": why, "seq": s.get("seq")})
            if r.get("ok"):
                for x in recs:
                    if x.get("kind") == "build" and x.get("path"):
                        build_at[_norm(x["path"])] = i
            if r.get("ok") and any(x.get("kind") == "build" for x in recs):
                continue
        elif r.get("ok") and n in _CODE_CHANGERS:
            p = a.get("path") or a.get("destination") or ""
            touched = r.get("touched") or {}
            # a change to a file the task itself created (a stub, a copy, a decoy) cannot alter the designated
            # program's behaviour unless the program imports it, and a pre-existing program cannot import a
            # file that did not exist; only changes to pre-existing (or unknown) files reset the proof
            if touched.get("existed_before") is False and touched.get("path") not in target["files"]:
                continue
            if isinstance(p, str) and (p.lower().endswith(CODE_EXT) or n == "APPLY_PATCH"):
                last_change = i
    counting = [x for x in runs if x["counts"]]
    base = {"designated": {"files": [os.path.relpath(f, root) if f.startswith(_norm(root)) else f for f in target["files"]],
                           "tests": target["tests"]},
            "runs_considered": len(runs), "substitute_runs": [x["command"][:160] for x in runs if x["ok"] and not x["counts"]
                                                               and any(e.get("path") for e in x["executed"])]}
    if not counting:
        return {**base, "ok": False, "reason": _no_run_reason(target, runs)}
    last = counting[-1]
    if not last["ok"]:
        return {**base, "ok": False, "verifying_step": last["step"], "command": last["command"],
                "reason": f"the last run of the designated target (step {last['step']}) failed"}
    stale = [e for e in last["executed"] if e.get("built_from") and build_at.get(_norm(e["path"]), -1) < last_change]
    if stale:
        return {**base, "ok": False, "verifying_step": last["step"], "command": last["command"],
                "reason": "the program that ran was built before the last code change; rebuild it and run it again"}
    if last["index"] < last_change:
        return {**base, "ok": False, "verifying_step": last["step"], "command": last["command"],
                "reason": "the code changed after the last green run of the designated target; run it again"}
    return {**base, "ok": True, "verifying_step": last["step"], "command": last["command"], "seq": last["seq"],
            "executed": last["executed"], "counts_because": last["counts"]}


class _Shim:
    """An executor stand-in for parsing a recorded command: the root as cwd, no snapshot, builds remembered."""

    def __init__(self, root: str):
        self.cwd, self.snapshot_files, self.builds = root, None, {}

    def _p(self, p: str) -> Path:
        return Path(p) if os.path.isabs(p) else Path(self.cwd, p)


_CODE_CHANGERS = {"WRITE_FILE", "EDIT_FILE", "REPLACE_LINE", "INSERT_LINES", "DELETE_LINES", "APPEND_FILE", "CREATE_FILE",
                  "APPLY_PATCH", "REVERT_FILE", "MOVE_FILE", "COPY_FILE", "DELETE_FILE"}


def _no_run_reason(target: dict, runs: list[dict]) -> str:
    what = []
    if target["tests"]:
        what.append("the project's own tests (RUN_TESTS in the folder, or its pre-existing test files)")
    if target["files"] and not target["tests"]:
        what.append(" or ".join(os.path.basename(f) for f in target["files"]) + " itself (or the program built from it)")
    subs = [x for x in runs if x["ok"] and any(e.get("path") for e in x["executed"])]
    tail = ""
    if subs:
        made = [e for x in subs for e in x["executed"] if e.get("created_in_task")]
        tail = (" A green run of a file made during this task does not count" if made else
                " The green runs so far executed something else") + "; the proof has to run the designated target."
    return "no green run of " + " and ".join(what) + " after the fix." + tail


def gate_message(v: dict | None) -> str | None:
    """The unmet requirement the gate reports, or None."""
    if not v or v.get("ok") is not False:
        return None
    return "verification integrity: " + v["reason"]


def receipt(goal: str, trace, root: str, task_id=None, defect_log=None) -> dict:
    """Everything needed to reconstruct why a task was (or was not) considered verified. Evidence, not a log
    line: the designated target, the run that verified it (command, resolved artifact, exit code, sequence
    number), the defect state at that moment, and every file changed before it with its origin."""
    v = verdict(goal, trace, root)
    rec = {"task_id": task_id, "applies": v is not None}
    if v is None:
        return rec
    rec.update({k: v.get(k) for k in ("ok", "reason", "designated", "counts_because", "substitute_runs", "runs_considered")})
    step = v.get("verifying_step")
    idx = next((i for i, s in enumerate(trace or []) if s.get("step") == step), None) if step is not None else None
    if idx is not None:
        s = trace[idx]
        r = s.get("result") or {}
        m = re.search(r"exit code (-?\d+)", str(r.get("verification") or "") + " " + str(r.get("error") or ""))
        rec["verification"] = {
            "step": step, "seq": s.get("seq"), "command": (s.get("action") or {}).get("command") or (s.get("action") or {}).get("path"),
            "exit_code": int(m.group(1)) if m else None, "ok": bool(r.get("ok")),
            "executed": [{k: e.get(k) for k in ("path", "kind", "existed_before", "created_in_task", "built_from")}
                         for e in (r.get("executed") or [])]}
        before = trace[:idx]
    else:
        before = list(trace or [])
    rec["modified_before_verification"] = [
        {"step": s.get("step"), "action": (s.get("action") or {}).get("action"), **((s.get("result") or {}).get("touched") or {})}
        for s in before if (s.get("result") or {}).get("touched")]
    if defect_log:
        at = [d for d in defect_log if step is None or (d.get("step") or 0) <= step]
        rec["defect_state_at_verification"] = at[-1] if at else None
    return rec
