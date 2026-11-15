"""QUERY_TABLE: a number read out of a table is computed by sqlite, never by the model.

The read-task twin of false success (tribunal, 2026-10-12): "which month sold most in sales.csv"
is answered today by READ_FILE into a 4B context, then truncation, then an invented sum. The gate
checks that evidence exists, not that the model's arithmetic over it is right, so the wrong number
passes. Here the executor loads the file into an in-memory sqlite database and runs the query, so
the number in the answer is a fact in the trace.

What a file becomes:
  * .csv / .tsv / .txt      -> one table `t`, delimiter sniffed, first row the header unless every
                               cell in it parses as a number
  * .xlsx                   -> one table `t` from one sheet (`sheet=` by name or 1-based number);
                               a formula cell contributes the value Excel cached, not the formula
  * .db / .sqlite / .sqlite3 -> the file's own tables, opened read-only

Every cell that reads as a number is stored as one (1,200, $1,200, 12%, (5) for -5), so SUM and AVG
work without the model writing a CAST. The rest stay text. sqlite is dynamically typed per value,
so a column of mostly-numbers keeps its one stray "n/a" as text and SUM still adds the numbers.

Read-only three ways over: the statement has to start with SELECT or WITH, sqlite3 itself refuses
more than one statement per execute, and an authorizer denies every action but reading.
"""
from __future__ import annotations

import csv
import io
import re
import sqlite3
import zipfile
from pathlib import Path

MAX_ROWS = 200_000      # rows loaded from one file; past this the answer says what it left out
MAX_OUT_ROWS = 50       # rows rendered back to the model
MAX_CELL = 60           # characters per rendered cell
TABLE = "t"

SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".db", ".sqlite", ".sqlite3")
_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")

# sqlite3 authorizer codes we allow: anything else is denied.
_ALLOW = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, 33}  # 33 = SQLITE_RECURSIVE


class TableError(ValueError):
    """The file cannot be read as a table, or the query is not a read."""


def _num(cell: str):
    """The number a spreadsheet cell means, or None. '1,200' -> 1200, '$1,200.50' -> 1200.5,
    '12%' -> 12.0, '(5)' -> -5. A bare '-' or '' is not a number."""
    s = str(cell).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()
    s = s.lstrip("$£€").rstrip("%").strip()
    s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s) else s
    if not re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s):
        return None
    v = float(s) if ("." in s or "e" in s.lower()) else int(s)
    if neg:
        v = -v
    return -v if neg and v > 0 else v


def _cell(cell: str):
    n = _num(cell)
    return n if n is not None else str(cell)


def _column_names(header: list[str]) -> list[str]:
    """Header text -> sqlite identifiers the model can type: lowercase, words joined by _."""
    out: list[str] = []
    for i, h in enumerate(header):
        name = re.sub(r"[^0-9a-z]+", "_", str(h).strip().lower()).strip("_")
        if not name or name[0].isdigit():
            name = "c" + str(i + 1) + ("_" + name if name else "")
        while name in out:
            name += "_2"
        out.append(name)
    return out


def _rows_csv(path: Path) -> list[list[str]]:
    text = path.read_bytes().decode("utf-8-sig", "replace")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" or "\t" in sample else csv.excel
    rows = []
    for row in csv.reader(io.StringIO(text), dialect):
        if any(c.strip() for c in row):
            rows.append(row)
        if len(rows) > MAX_ROWS:
            break
    return rows


def _col_index(ref: str) -> int:
    """'C' -> 2 in a cell reference like C12."""
    n = 0
    for ch in ref:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def xlsx_rows(path: Path, sheet=None) -> tuple[list[list[str]], str, list[str]]:
    """One sheet of an .xlsx as rows of strings, plus the sheet's name and all sheet names.
    A formula cell gives the value Excel cached for it; a formula never written by Excel (openpyxl
    writes none) gives an empty cell."""
    with zipfile.ZipFile(path) as z:
        def xml(name):
            try:
                return z.read(name).decode("utf-8", "replace")
            except KeyError:
                return ""
        book = xml("xl/workbook.xml")
        names = re.findall(r'<sheet [^>]*name="([^"]+)"', book)
        parts = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                       key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
        if not parts:
            raise TableError("no worksheet inside this workbook")
        names = names or [f"Sheet{i + 1}" for i in range(len(parts))]
        which = 0
        if sheet is not None and str(sheet).strip():
            s = str(sheet).strip()
            if s.isdigit() and 1 <= int(s) <= len(parts):
                which = int(s) - 1
            elif s.lower() in [n.lower() for n in names]:
                which = [n.lower() for n in names].index(s.lower())
            else:
                raise TableError(f"no sheet {s!r} in this workbook (it has: {', '.join(names)})")
        shared = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", si))
                  for si in re.findall(r"<si>.*?</si>", xml("xl/sharedStrings.xml"), re.S)]
        rows: list[list[str]] = []
        for rx in re.findall(r"<row[^>]*>.*?</row>|<row[^>]*/>", xml(parts[which]), re.S):
            row: list[str] = []
            for ref, attrs, inner in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', rx, re.S):
                i = _col_index(ref)
                while len(row) <= i:
                    row.append("")
                t = re.search(r't="(\w+)"', attrs)
                kind = t.group(1) if t else "n"
                v = re.search(r"<v>([^<]*)</v>", inner or "")
                val = v.group(1) if v else ""
                if kind == "s" and val != "":
                    val = shared[int(val)] if int(val) < len(shared) else ""
                elif kind == "inlineStr":
                    val = "".join(re.findall(r"<t[^>]*>([^<]*)</t>", inner or ""))
                row[i] = val
            if any(c.strip() for c in row):
                rows.append(row)
            if len(rows) > MAX_ROWS:
                break
        return rows, names[which], names


def _load_rows(conn: sqlite3.Connection, rows: list[list[str]]) -> dict:
    """rows (first one the header unless it is all numbers) -> table `t`. Returns what was loaded."""
    if not rows:
        raise TableError("the file has no rows")
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    first = rows[0]
    headed = not all(_num(c) is not None for c in first if str(c).strip())
    cols = _column_names(first) if headed else [f"c{i + 1}" for i in range(width)]
    body = rows[1:] if headed else rows
    truncated = len(body) >= MAX_ROWS
    body = body[:MAX_ROWS]
    conn.execute(f'CREATE TABLE {TABLE} ({", ".join(f"[{c}]" for c in cols)})')
    conn.executemany(f'INSERT INTO {TABLE} VALUES ({", ".join("?" * width)})',
                     [[_cell(c) for c in r] for r in body])
    numeric = [c for i, c in enumerate(cols)
               if body and sum(1 for r in body if _num(r[i]) is not None) > len(body) / 2]
    return {"table": TABLE, "columns": cols, "header_row": headed, "rows": len(body),
            "numeric_columns": numeric, "truncated": truncated,
            "original_header": first if headed else None}


def open_table(path: Path, sheet=None) -> tuple[sqlite3.Connection, dict]:
    """The file as a queryable read-only database, plus what it holds."""
    suffix = path.suffix.lower()
    if suffix in _SQLITE_SUFFIXES:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        if not names:
            conn.close()
            raise TableError(f"{path.name} is a database with no tables")
        info = {"tables": names, "columns": {}, "rows": {}}
        for n in names:
            info["columns"][n] = [r[1] for r in conn.execute(f'PRAGMA table_info("{n}")')]
            try:
                info["rows"][n] = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
            except sqlite3.Error:
                info["rows"][n] = None
        info["table"] = names[0]
    elif suffix == ".xlsx":
        rows, used, names = xlsx_rows(path, sheet)
        conn = sqlite3.connect(":memory:")
        info = _load_rows(conn, rows)
        info["sheet"], info["sheets"] = used, names
    elif suffix in (".csv", ".tsv", ".txt"):
        conn = sqlite3.connect(":memory:")
        info = _load_rows(conn, _rows_csv(path))
    else:
        raise TableError(f"{path.name}: QUERY_TABLE reads {', '.join(SUFFIXES)}; "
                         f"use READ_FILE for other files")
    conn.commit()
    conn.set_authorizer(lambda code, *_rest: sqlite3.SQLITE_OK if code in _ALLOW else sqlite3.SQLITE_DENY)
    return conn, info


def query(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    """One read of the loaded table. Column names and rows, or TableError."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise TableError("sql is empty")
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise TableError("sql must be a single SELECT (or WITH ... SELECT); QUERY_TABLE never changes a file")
    try:
        cur = conn.execute(s)
    except sqlite3.Error as e:
        raise TableError(str(e)) from e
    names = [d[0] for d in cur.description or []]
    rows = cur.fetchmany(MAX_OUT_ROWS + 1)
    return names, rows


def render_schema(info: dict) -> str:
    """What the model needs before it can write a query."""
    if "tables" in info:
        out = [f"database with {len(info['tables'])} table(s):"]
        for n in info["tables"]:
            out.append(f'  "{n}" ({info["rows"].get(n)} rows): ' + ", ".join(info["columns"][n]))
        return "\n".join(out)
    out = [f"table {TABLE}: {info['rows']} rows, columns: " + ", ".join(info["columns"])]
    if info.get("original_header"):
        # Only where the name really changed shape: "Month -> month" tells the model nothing.
        pairs = ", ".join(f"{str(h).strip()} -> {c}" for h, c in zip(info["original_header"], info["columns"])
                          if str(h).strip().lower() != c)
        if pairs:
            out.append("column names as the file spells them: " + pairs)
    if not info.get("header_row"):
        out.append("the first row is data, not a header, so the columns are named c1, c2, ...")
    if info.get("numeric_columns"):
        out.append("numbers in: " + ", ".join(info["numeric_columns"]))
    if info.get("sheets") and len(info["sheets"]) > 1:
        out.append(f"sheet {info['sheet']!r} of {len(info['sheets'])}: " + ", ".join(info["sheets"])
                   + " (sheet= to read another)")
    if info.get("truncated"):
        out.append(f"ONLY THE FIRST {MAX_ROWS} ROWS WERE LOADED: any total over this file is incomplete")
    return "\n".join(out)


def render_rows(names: list[str], rows: list[tuple]) -> str:
    """The query's answer as a small fixed-width table."""
    if not rows:
        return "no rows matched"
    cut = rows[:MAX_OUT_ROWS]

    def text(v):
        if v is None:
            return ""
        if isinstance(v, float):
            v = f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) < 1e15 else repr(v)
        s = str(v)
        return s[:MAX_CELL - 1] + "…" if len(s) > MAX_CELL else s

    grid = [[text(c) for c in r] for r in cut]
    widths = [max(len(names[i]), max((len(r[i]) for r in grid), default=0)) for i in range(len(names))]
    lines = ["  ".join(n.ljust(widths[i]) for i, n in enumerate(names)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in grid]
    if len(rows) > MAX_OUT_ROWS:
        lines.append(f"... more rows than shown; {MAX_OUT_ROWS} of them are above")
    else:
        lines.append(f"{len(cut)} row(s)")
    return "\n".join(lines)


def answer(path: Path, sql: str | None = None, sheet=None) -> dict:
    """Load the file and, when sql is given, run it. Everything the executor needs for one Result."""
    conn, info = open_table(path, sheet)
    try:
        out = {"info": info, "schema": render_schema(info)}
        if not (sql and sql.strip()):
            names, rows = query(conn, f"SELECT * FROM {_first_table(info)} LIMIT 5")
            out["text"] = out["schema"] + "\n\nfirst rows:\n" + render_rows(names, rows)
            out["witness"] = f"read {_count(info)} row(s) from {path.name}; no query asked for yet"
            return out
        names, rows = query(conn, sql)
        out["columns"], out["rows"] = names, rows
        out["text"] = render_rows(names, rows)
        out["witness"] = (f"sqlite computed this over {_count(info)} row(s) read from {path.name}: "
                          f"{sql.strip()[:200]}")
        if info.get("truncated"):
            out["witness"] += f" (only the first {MAX_ROWS} rows were loaded)"
        return out
    finally:
        conn.close()


def _first_table(info: dict) -> str:
    return f'"{info["table"]}"' if "tables" in info else TABLE


def _count(info: dict):
    if "tables" in info:
        return sum(v for v in info["rows"].values() if isinstance(v, int))
    return info["rows"]


def demo() -> None:
    """Self-check: the sum the model would have guessed, against the one sqlite computes."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    p = d / "sales.csv"
    p.write_text("Month,Units Sold,Revenue\nJan,10,\"$1,200\"\nFeb,7,$900.50\nMar,21,\"$2,400\"\n", encoding="utf-8")
    a = answer(p, "SELECT month, revenue FROM t ORDER BY revenue DESC LIMIT 1")
    print("got:     ", a["text"].splitlines()[2].split())
    print("expected:", ["Mar", "2400"])
    assert a["text"].splitlines()[2].split() == ["Mar", "2400"], a["text"]
    tot = answer(p, "SELECT SUM(units_sold), ROUND(AVG(revenue), 2) FROM t")
    print("got:     ", tot["rows"][0], " expected: (38, 1500.17)")
    assert tot["rows"][0] == (38, 1500.17), tot["rows"]
    schema = answer(p)
    print("schema:", schema["schema"].splitlines()[0])
    assert "units_sold" in schema["schema"] and "3 rows" in schema["schema"]
    for bad in ("DELETE FROM t", "DROP TABLE t", "UPDATE t SET month='x'",
                "SELECT 1; DROP TABLE t", "INSERT INTO t VALUES ('a',1,2)"):
        try:
            answer(p, bad)
        except TableError as e:
            print(f"refused {bad!r}: {str(e)[:60]}")
        else:
            raise AssertionError(f"{bad!r} was not refused")
    print("tabular demo: ok")


if __name__ == "__main__":
    demo()
