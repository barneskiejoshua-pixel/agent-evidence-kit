"""Exact arithmetic for CALCULATE, and the fast path's reading of a plain math question.

Live 2026-09-26 (drockclaude q100): once the web was reachable again, "what is 17 * 23?" compiled to
SEARCH_WEB, the answer came back as a calculator site's snippet, and the gate refused the model's
"39"; offline the same question went to the 4B model's head. Arithmetic is not a fact about the
world and not a guess: the executor computes it, and the answer is the executor's output.

evaluate() walks a Python AST of numbers, + - * / // % **, unary minus, parentheses and a few named
functions. No names, attributes, calls to anything else, or eval(). Exponents and results are
bounded so "9**9**9" cannot hang the executor.

question() turns the common English shapes into an expression, or returns None: the fast path only
compiles a question it read completely, and everything else goes to the model as before.
"""
from __future__ import annotations

import ast
import datetime as _dt
import math
import operator
import re
from fractions import Fraction

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
class CalcError(ValueError):
    pass


def _isprime(n) -> bool:
    if n != int(n) or n < 2:
        return False
    n = int(n)
    if n > 10 ** 12:
        raise CalcError("isprime is limited to numbers up to 10**12")
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def _weekday(y, m, d) -> str:
    try:
        return _dt.date(int(y), int(m), int(d)).strftime("%A")
    except ValueError as e:
        raise CalcError(str(e)) from None


def _countchar(word, ch) -> int:
    if not isinstance(word, str) or not isinstance(ch, str) or len(ch) != 1 or not ch.isalpha():
        raise CalcError("countchar needs one letter and a word")
    if not word.isalpha() or len(word) > 40:
        raise CalcError("countchar needs one word of letters")
    return word.count(ch)


def _revdigits(n) -> int:
    if isinstance(n, bool) or n != int(n) or n < 0:
        raise CalcError("revdigits needs a whole number that is not negative")
    return int(str(int(n))[::-1])


def _letter(n) -> str:
    if isinstance(n, bool) or n != int(n) or not 1 <= int(n) <= 26:
        raise CalcError("letter needs a whole number from 1 to 26")
    return chr(64 + int(n))


_DIRS = ("north", "east", "south", "west")


def _facing(start, *turns) -> str:
    if isinstance(start, bool) or start != int(start) or not 0 <= int(start) <= 3:
        raise CalcError("facing needs a direction 0-3")
    i = int(start)
    for t in turns:
        if isinstance(t, bool) or t != int(t) or int(t) not in (1, -1):
            raise CalcError("a turn is 1 or -1")
        i = (i + int(t)) % 4
    return _DIRS[i]


def _wordlen(word) -> int:
    if not isinstance(word, str) or not word.isalpha() or not 1 <= len(word) <= 40:
        raise CalcError("wordlen needs one word of letters")
    return len(word)


_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "factorial": math.factorial,
          "floor": math.floor, "ceil": math.ceil, "isprime": _isprime, "max": max, "min": min, "weekday": _weekday,
          "countchar": _countchar, "revdigits": _revdigits, "letter": _letter, "facing": _facing,
          "wordlen": _wordlen}
_ARITY = {"max": 20, "min": 20, "weekday": 3, "facing": 8}
_STR_FUNCS = {"countchar", "wordlen"}
_CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXP = 10_000
MAX_DIGITS = 4_000


def _normal(expr: str) -> str:
    src = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", src)          # 1,234 -> 1234


def split(expr: str) -> list[str]:
    """Several expressions in one CALCULATE, separated by ';' or ',' outside parentheses. "7/8, 5/6"
    (live 2026-09-26) was the model's own way to compare two fractions, and it was refused."""
    parts, depth, cur = [], 0, ""
    for ch in _normal(expr):
        depth += ch in "([" and 1 or ch in ")]" and -1 or 0
        if ch in ",;" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def computes(expr: str) -> bool:
    """Does the expression do any arithmetic? A bare number computes nothing: live 2026-09-26 the
    model worked out 60 * 2.5 in its head as 135, sent CALCULATE(expression="135"), and that read as
    an executor-verified result."""
    try:
        tree = ast.parse(_normal(expr), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.BinOp, ast.Call, ast.Compare)) for n in ast.walk(tree))


def evaluate(expr: str):
    """The exact value of one arithmetic expression. Raises CalcError for anything else."""
    src = _normal(expr)
    if not src or len(src) > 500:
        raise CalcError("empty or too long")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise CalcError(f"not an arithmetic expression: {expr!r}") from None
    return _eval(tree.body)


def _eval(n, strings=False):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if strings and isinstance(n, ast.Constant) and isinstance(n.value, str):
        if len(n.value) > 40:
            raise CalcError("text too long")
        return n.value
    if isinstance(n, ast.Name) and n.id in _CONSTS:
        return _CONSTS[n.id]
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
        return _CMP[type(n.ops[0])](_eval(n.left), _eval(n.comparators[0]))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _eval(n.operand)
        return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > MAX_EXP:
            raise CalcError("exponent too large")
        if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise CalcError("division by zero")
        v = _BIN[type(n.op)](a, b)
        if isinstance(v, int) and v.bit_length() > MAX_DIGITS * 3.33:
            raise CalcError("result too large")
        return v
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
            and not n.keywords and 1 <= len(n.args) <= _ARITY.get(n.func.id, 2)):
        # A word is allowed only as an argument of countchar. "'a' * 3" is still rejected.
        args = [_eval(a, n.func.id in _STR_FUNCS) for a in n.args]
        if n.func.id == "factorial" and (args[0] != int(args[0]) or not 0 <= args[0] <= 1000):
            raise CalcError("factorial needs a whole number from 0 to 1000")
        try:
            return _FUNCS[n.func.id](*[int(x) if n.func.id == "factorial" else x for x in args])
        except (ValueError, OverflowError) as e:
            raise CalcError(str(e)) from None
    raise CalcError("only numbers, + - * / // % ** ^, one comparison (> < == ...), parentheses and sqrt/abs/"
                    "round/log/log10/sin/cos/tan/factorial/floor/ceil/isprime/max/min/weekday(y, m, d)/"
                    "countchar/revdigits/letter/facing are allowed; "
                    "separate several expressions with ';'")


def fmt(v) -> str:
    """A result as a person writes it: 391, 37.78, 1048576, never 391.0 or 37.77777777777778."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}" if abs(v) >= 1e-4 else f"{v:.6g}"
    return str(v)


# ---------- English -> expression ----------
_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_LEAD = re.compile(r"^(?:please\s+)?(?:(?:what|whats|what's)\s+(?:is|are|does|do)\s+|calculate\s+|compute\s+|"
                   r"work\s+out\s+|evaluate\s+|solve\s+|how\s+much\s+is\s+|what\s+do\s+you\s+get\s+(?:for|from)\s+)", re.I)
_WORD_OPS = [(r"\bmultiplied\s+by\b|\btimes\b|\bx\b", "*"), (r"\bdivided\s+by\b|\bover\b", "/"),
             (r"\bplus\b|\badded\s+to\b", "+"), (r"\bminus\b|\bless\b", "-"),
             (r"\bto\s+the\s+power\s+of\b|\braised\s+to(?:\s+the\s+power\s+of)?\b", "**"),
             (r"\bmod(?:ulo)?\b", "%")]
_EXPR_OK = re.compile(r"^[\d\s.+\-*/%()^,]+$")

# Unit conversions with fixed factors. Temperature is the only affine one.
_UNITS = {
    ("km", "mi"): 0.621371, ("mi", "km"): 1.609344, ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("cm", "in"): 1 / 2.54, ("in", "cm"): 2.54, ("kg", "lb"): 2.20462262, ("lb", "kg"): 0.45359237,
    ("l", "gal"): 0.264172, ("gal", "l"): 3.78541, ("g", "oz"): 0.03527396, ("oz", "g"): 28.3495231,
    ("m", "cm"): 100, ("cm", "m"): 0.01, ("km", "m"): 1000, ("m", "km"): 0.001,
}
_UNIT_NAMES = {"km": "km|kilometers?|kilometres?", "mi": "mi|miles?", "m": "m|meters?|metres?", "ft": "ft|feet|foot",
               "cm": "cm|centimeters?|centimetres?", "in": "in|inch|inches", "kg": "kg|kilograms?|kilos?",
               "lb": "lbs?|pounds?", "l": "l|liters?|litres?", "gal": "gal|gallons?", "g": "g|grams?", "oz": "oz|ounces?",
               "c": "c|°c|celsius|centigrade", "f": "f|°f|fahrenheit"}


def _unit(word: str) -> str | None:
    w = word.strip().lower()
    for key, pat in _UNIT_NAMES.items():
        if re.fullmatch(pat, w):
            return key
    return None


_TIME_S = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
_DATE = r"(\d{4}-\d{2}-\d{2})"


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def _parse_date(s: str) -> tuple[int, int, int] | None:
    """(year, month, day) from "July 4 1976", "july 4th, 1976", "4 July 1976", "1976-07-04"; None otherwise."""
    s = s.strip().rstrip("?.! ").lower().replace(",", " ")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        month = r"(" + "|".join(_MONTHS) + r"|" + "|".join(x[:3] for x in _MONTHS) + r")\.?"
        m = re.fullmatch(month + r"\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})", s)
        if m:
            name, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        else:
            m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + month + r"\s+(\d{4})", s)
            if not m:
                return None
            d, name, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = next(i + 1 for i, x in enumerate(_MONTHS) if x.startswith(name[:3]))
    try:
        _dt.date(y, mo, d)
    except ValueError:
        return None
    return y, mo, d


# Three sequence rules, and only those: a constant difference, a constant ratio, or a constant
# second difference when the first difference is not constant. Fibonacci matches none and falls
# through. A constant run (difference 0 and ratio 1) matches two rules and falls through too.
# One unknown, and only a line through x=0, 1 and 2. A bare "x = 5" is a claim, not an equation.
_EQ_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789+*/(). -")
_DIR_AT = {"north": 0, "east": 1, "south": 2, "west": 3}


def _num(x: Fraction) -> str:
    if x.denominator == 1:
        n = x.numerator
        return str(n) if n >= 0 else f"({n})"
    n, d = x.numerator, x.denominator
    return f"({n}/{d})" if n >= 0 else f"({n}/{d})"


def _as_frac(v) -> Fraction:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CalcError("not a number")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise CalcError("not a number")
        return Fraction(v).limit_denominator(10 ** 6)
    return Fraction(v)


def _sample(side: str, var: str, k: int) -> Fraction:
    return _as_frac(evaluate(re.sub(rf"\b{var}\b", f"({k})", side)))


def _linear(eq: str) -> tuple[str, str] | None:
    """(variable, expression) for one straight line, or None. The expression is the division the
    executor recomputes, not a number worked out here."""
    m = re.fullmatch(r"([a-z0-9+*/(). -]+)=([a-z0-9+*/(). -]+)", eq.strip())
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip()
    letters = set(re.findall(r"[a-z]", left + "=" + right))
    if len(letters) != 1:
        return None
    var = letters.pop()
    if ((re.fullmatch(var, left) and re.fullmatch(r"-?\d+(?:\.\d+)?", right))
            or (re.fullmatch(var, right) and re.fullmatch(r"-?\d+(?:\.\d+)?", left))):
        return None
    try:
        l0, l1, l2 = (_sample(left, var, k) for k in (0, 1, 2))
        r0, r1, r2 = (_sample(right, var, k) for k in (0, 1, 2))
    except (CalcError, ZeroDivisionError, OverflowError):
        return None
    if (l2 - l1) - (l1 - l0) != 0 or (r2 - r1) - (r1 - r0) != 0:
        return None
    slope = (l1 - l0) - (r1 - r0)
    if slope == 0:
        return None
    return var, f"({_num(r0)} - ({_num(l0)})) / ({_num(slope)})"


def _next_term(xs: list[Fraction]) -> tuple[str, Fraction] | None:
    if len(xs) < 3:
        return None
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    const_diff = all(d == diffs[0] for d in diffs)
    ratio = None
    if all(x != 0 for x in xs[:-1]):
        rats = [xs[i + 1] / xs[i] for i in range(len(xs) - 1)]
        if all(r == rats[0] for r in rats):
            ratio = rats[0]
    second = None
    if not const_diff and len(xs) >= 4:
        secs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        if secs[0] != 0 and all(s == secs[0] for s in secs):
            second = secs[0]
    hits = []
    if const_diff:
        hits.append("d")
    if ratio is not None:
        hits.append("r")
    if second is not None:
        hits.append("s")
    if len(hits) != 1:
        return None
    last = xs[-1]
    if hits[0] == "d":
        return f"{_num(last)} + ({_num(diffs[0])})", last + diffs[0]
    if hits[0] == "r":
        return f"{_num(last)} * ({_num(ratio)})", last * ratio
    return (f"{_num(last)} + (({_num(diffs[-1])}) + ({_num(second)}))",
            last + diffs[-1] + second)


def _pieces(body: str) -> list[str] | None:
    """Comma-separated terms. A comma with no space after it is '1,234', not a list.
    A trailing '?' is the blank the question is asking for, not a term."""
    body = re.sub(r"(?:,\s*)?\?\s*$", "", body).strip(" .,:")
    if re.search(r",\S", body):
        return None
    parts = [p.strip() for p in re.split(r"\s*,\s+|\s+and\s+", body) if p.strip()]
    return parts if len(parts) >= 3 else None


_SEQ_LEAD = re.compile(
    r"^(?:what(?:'s| is) the next (?:number|term|letter)(?: in(?: the sequence|in this sequence)?)?|"
    r"what (?:number|letter|term) comes next(?: in this sequence)?|"
    r"what comes next(?: in(?: the sequence)?)?|"
    r"next (?:number|term|letter)(?: in(?: the sequence)?)?)[:,]?\s+")


def _sequence(low: str) -> tuple[str, str] | None:
    body = _SEQ_LEAD.sub("", low).strip()
    parts = _pieces(body)
    if not parts:
        return None
    if all(re.fullmatch(r"-?\d+(?:\.\d+)?", p) for p in parts):
        nxt = _next_term([Fraction(p) for p in parts])
        if not nxt:
            return None
        return nxt[0], "the next term"
    if all(re.fullmatch(r"[a-z]", p) for p in parts):
        nxt = _next_term([Fraction(ord(p) - 96) for p in parts])
        if not nxt or nxt[1].denominator != 1 or not 1 <= nxt[1].numerator <= 26:
            return None
        return f"letter({nxt[0]})", "the next letter"
    return None


def _closed(low: str, body: str) -> tuple[str, str] | None:
    """A question read in full, or None. A near-miss returns None so the model still sees it."""
    m = re.fullmatch(rf"({_NUM}) (squared|cubed)", body)
    if m:
        n = m.group(1).replace(",", "")
        exp = "2" if m.group(2) == "squared" else "3"
        return f"{n} ** {exp}", f"{n} {m.group(2)}"
    m = re.fullmatch(r"repeat these digits in reverse order[:,]?\s+((?:\d\s+)+\d)", low)
    if m:
        n = "".join(m.group(1).split())
        return f"revdigits({n})", f"digits {m.group(1)} reversed"
    m = re.fullmatch(rf"reverse (?:the )?digits of ({_NUM})", low) or re.fullmatch(
        rf"digits of ({_NUM}) reversed", low)
    if m:
        n = m.group(1).replace(",", "")
        if "." in n or n.startswith("-"):
            return None
        n = n.lstrip("0") or "0"
        return f"revdigits({n})", f"digits of {n} reversed"
    m = re.fullmatch(
        r"how many (?:times does )?(?:the letter )?([a-z])(?:'s|s)?(?: appear| are| is)?(?: there)?"
        r" in (?:the word )?([a-z]+)", low)
    if m:
        return f'countchar("{m.group(2)}", "{m.group(1)}")', f"how many {m.group(1)} in {m.group(2)}"
    m = re.fullmatch(r"how many letters (?:are there |are )?in (?:the word )?([a-z]+)", low)
    if m:
        return f'wordlen("{m.group(1)}")', f"letters in {m.group(1)}"
    turned = _turns(low)
    if turned:
        return turned
    seq = _sequence(low)
    if seq:
        return seq
    eq = re.sub(r"^(?:please\s+)?(?:solve\s+|what\s+is\s+[a-z]\s+if\s+|find\s+[a-z]\s*(?::|where)?\s*)",
                "", low).strip()
    hit = _linear(eq)
    if hit:
        return hit[1], f"solved for {hit[0]}"
    return None


_TURN_OK = {"you", "turn", "turns", "right", "left", "then", "again", "and", "which", "direction",
             "do", "face", "now", "way", "are", "facing", "i", "am", "what"}


def _turns(low: str) -> tuple[str, str] | None:
    """'face north, turn right' and 'You face north. You turn right, then right again, then left.'
    Anything that is not a turn ('sit', 'jump') is a near-miss."""
    m = re.search(r"\b(?:face|facing) (north|east|south|west)\b", low)
    if not m:
        return None
    rest = re.sub(r"\bwhich (?:direction|way)\b.*$", "", low[m.end():])
    words = re.findall(r"[a-z]+", rest)
    turns = [w for w in words if w in ("right", "left")]
    if not turns or len(turns) > 7 or any(w not in _TURN_OK for w in words):
        return None
    args = ", ".join("1" if t == "right" else "-1" for t in turns)
    return f"facing({_DIR_AT[m.group(1)]}, {args})", f"facing {m.group(1)}"


def question(text: str) -> tuple[str, str] | None:
    """(expression, how it was read) for a plain math question, or None."""
    q = re.sub(r"\s+", " ", text.strip()).strip()
    low = q.lower()
    # "Answer with the number." is a format instruction. It is not part of the question.
    low = re.sub(r"\s+answer with (?:one word|the number|yes or no|the letter|the name)\b[.!]?\s*$",
                 "", low).strip()
    low = low.rstrip("?!. ").strip()

    m = re.fullmatch(rf"(?:convert\s+|what\s+is\s+|how\s+many\s+\w+\s+(?:is|are)\s+)?({_NUM})\s*([a-z°]+)\s+(?:to|in|into)\s+([a-z°]+)", low)
    if m:
        a, b = _unit(m.group(2)), _unit(m.group(3))
        v = m.group(1).replace(",", "")
        if (a, b) == ("f", "c"):
            return f"({v} - 32) * 5 / 9", f"{v} °F in °C"
        if (a, b) == ("c", "f"):
            return f"{v} * 9 / 5 + 32", f"{v} °C in °F"
        if (a, b) in _UNITS:
            return f"{v} * {_UNITS[(a, b)]}", f"{v} {a} in {b}"
        return None

    m = re.fullmatch(r"how many (seconds|minutes|hours|days) are (?:there )?in (?:a|an|one) (minute|hour|day|week)", low)
    if m:
        small, big = m.group(1)[:-1], m.group(2)
        if _TIME_S[big] > _TIME_S[small]:
            return f"{_TIME_S[big]} / {_TIME_S[small]}", f"{small}s in a {big}"

    m = re.fullmatch(rf"how many days (?:are there )?between {_DATE} and {_DATE}", low)
    if m:
        try:
            a, b = _dt.date.fromisoformat(m.group(1)), _dt.date.fromisoformat(m.group(2))
        except ValueError:
            return None
        return f"{b.toordinal()} - {a.toordinal()}", f"days from {m.group(1)} to {m.group(2)} (as day numbers)"

    # "what day of the week was July 4 1976?" (q100: a list of links, "Sunday" only inside the snippets)
    m = re.fullmatch(r"(?:what|which) day(?: of the week)? (?:was|is|will be|falls on|fell on) (.+)", low)
    if m:
        ymd = _parse_date(m.group(1))
        if ymd:
            return f"weekday({ymd[0]}, {ymd[1]}, {ymd[2]})", f"the weekday of {ymd[0]}-{ymd[1]:02d}-{ymd[2]:02d}"

    m = re.fullmatch(rf"is ({_NUM}) (?:a )?prime(?: number)?", low)
    if m:
        return f"isprime({m.group(1).replace(',', '')})", f"whether {m.group(1)} is prime"

    m = re.fullmatch(r"(?:which|what) is (bigger|larger|greater|more|smaller|less|lower)[,:]? (.+?) or (.+)", low)
    if m:
        a, b = m.group(2).strip(), m.group(3).strip()
        if all(_EXPR_OK.match(x) and re.search(r"\d", x) and "," not in x for x in (a, b)):
            try:
                evaluate(a), evaluate(b)
            except CalcError:
                return None
            return f"{a}; {b}; {a} > {b}", f"which of {a} and {b} is {m.group(1)}"

    body = _LEAD.sub("", low).strip()
    m = re.fullmatch(rf"({_NUM})\s*(?:%|percent)\s+of\s+({_NUM})", body)
    if m:
        return f"{m.group(1)} / 100 * {m.group(2)}".replace(",", ""), f"{m.group(1)}% of {m.group(2)}"
    m = re.fullmatch(rf"the\s+square\s+root\s+of\s+({_NUM})|square\s+root\s+of\s+({_NUM})|sqrt\s*\(?\s*({_NUM})\s*\)?", body)
    if m:
        n = next(g for g in m.groups() if g).replace(",", "")
        return f"sqrt({n})", f"square root of {n}"
    m = re.fullmatch(rf"(?:the\s+)?(?:average|mean)\s+of\s+((?:{_NUM}\s*(?:,|and|,\s*and)?\s*)+)", body)
    if m:
        xs = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        if len(xs) >= 2:
            return f"({' + '.join(xs)}) / {len(xs)}", f"average of {len(xs)} numbers"
    m = re.fullmatch(rf"({_NUM})\s+(?:factorial|!)|({_NUM})!|(?:the\s+)?factorial\s+of\s+({_NUM})", body)
    if m:
        n = next(g for g in m.groups() if g)
        return f"factorial({n})", f"{n}!"
    closed = _closed(low, body)
    if closed:
        return closed
    expr = body
    for pat, op in _WORD_OPS:
        expr = re.sub(pat, f" {op} ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    if _EXPR_OK.match(expr) and re.search(r"\d\s*(?:\*\*|[-+*/%^])\s*[\d(]", expr):
        try:
            evaluate(expr)
        except CalcError:
            return None
        return expr, expr
    return None


def _eq_spans(text: str) -> list[str]:
    """Equations inside a reply. A word of two or more letters ends a span, so the sentence around
    the equation is not part of it."""
    spans = []
    for part in re.split(r"[^a-z0-9+*/().=\s-]+", text.lower()):
        for bit in re.split(r"[a-z]{2,}", part):
            bit = bit.strip()
            m = re.fullmatch(r"[a-z0-9+*/(). -]+=[a-z0-9+*/(). -]+", bit.strip())
            if m:
                # A sentence period is not a decimal: "x = 10." is the claim "x = 10".
                spans.append(m.group(0).strip().rstrip("."))
    return spans


def correct_equation(said: str) -> str | None:
    """A chat reply that states a linear equation and a different result. None when there is no
    equation, no stated result, or the stated result is already the one the executor gets.
    Wrong setup still gives a wrong answer: this only recomputes the equation that was written."""
    if not said or "=" not in said:
        return None
    solved = None
    for span in _eq_spans(said):
        hit = _linear(span)
        if hit:
            solved = hit
    if not solved:
        return None
    var, expr = solved
    try:
        value = fmt(evaluate(expr))
    except CalcError:
        return None
    claims = list(re.finditer(rf"\b{var}\s*(?:=|is|equals)\s*(-?\d+(?:\.\d+)?)\b", said.lower()))
    if not claims:
        return None
    stated = claims[-1].group(1)
    try:
        same = Fraction(stated) == Fraction(value)
    except (ValueError, ZeroDivisionError):
        same = stated == value
    if same:
        return None
    return f"[calculated] {var} = {value} ({expr} = {value})"
