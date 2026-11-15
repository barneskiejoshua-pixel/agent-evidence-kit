"""Secret values are replaced before any tool output reaches the model or the screen.

Everything an action produces is repeated: a READ_FILE of .env goes into the RESULT message, the
model quotes it in its reasoning, the trace records both, the terminal prints it, and a later
FETCH_URL or RUN_COMMAND can carry it out of the machine. None of those copies needs the value of
AWS_SECRET_ACCESS_KEY to do its job. The fact that the setting exists is useful; the forty characters
after the equals sign are not, and a 4B model has no judgement about where it is safe to repeat them.

So the rule is: the VALUE of a secret is replaced with a marker that names what kind of secret it
was — `AWS_SECRET_ACCESS_KEY=[redacted:aws-secret-key]` — and the key name, the header name, the
user and host of a connection string are kept, so the reader still learns what the file or output
contains. Two kinds of thing are recognised:

  * well-known token shapes, whatever they are labelled (AKIA..., ghp_..., sk-..., xoxb-..., a JWT,
    a PEM private-key block, `Bearer` in an Authorization header, user:password@ in a URL);
  * a key whose NAME says it holds a secret (secret, token, password, api_key, access_key,
    private_key, client_secret, auth) followed by a value of at least six characters that is not
    obviously something else — a boolean, a small number, a log level, a path, a URL, a placeholder
    like <your-key> or ${VAR}, or a marker this module already put there.

Line structure is preserved exactly — same number of lines, same line endings — so the numbered
lines READ_FILE hands the model stay aligned; a private-key block becomes one marker and the same
count of empty (still numbered) lines. Nothing here raises: an unexpected type is str()'d and a text
without secrets comes back unchanged. The cost of a false positive is a value the model cannot see;
the cost of a false negative is a credential in a trace file. The rules lean towards the former, but
not so far that LOG_LEVEL=info or APP_NAME=agent1 disappear.

Speed matters because this sits in front of every RUN_COMMAND output: a lower-cased copy is
searched with str.find for each literal stem a rule can start with (regex alternations are several
times slower at this), and the precise, cased patterns run only at the positions found. 200k
ordinary lines take well under a second.

Run from the directory that contains drock/:  python -m unittest drock.tests.test_redact
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

__all__ = ["redact", "findings", "has_secret", "marker"]


def marker(kind: str) -> str:
    """The text that replaces a secret of the given kind."""
    return f"[redacted:{kind}]"


# ---------------------------------------------------------------------------------------------
# Candidate scan.  Every rule begins at one of these literal stems.  A lower-cased copy of the
# text is searched with str.find for each stem — measured at ~60 MB/s for the whole list,
# against ~13 MB/s for the same words as a regex alternation and 4 MB/s once that alternation
# carries named groups — and only the positions found are handed to the precise, cased rules.
# The scan is a superset of the rules (it finds "token" in "tokenizer"); the rules decide.
# ---------------------------------------------------------------------------------------------

_STEMS: list[tuple[str, str]] = [
    # (lower-cased literal, rule group).  A stem that is a prefix of another ("auth" in
    # "authorization", "secret" in "client_secret") hits at both; sorting by position and
    # then by group order below settles who is asked first.
    ("-----begin", "pk"),
    ("authorization", "ah"),
    ("bearer", "be"),
    ("secret", "kv"), ("token", "kv"), ("passw", "kv"), ("passphrase", "kv"),
    ("apikey", "kv"), ("api_key", "kv"), ("api-key", "kv"),
    ("accesskey", "kv"), ("access_key", "kv"), ("access-key", "kv"),
    ("privatekey", "kv"), ("private_key", "kv"), ("private-key", "kv"),
    ("clientsecret", "kv"), ("client_secret", "kv"), ("client-secret", "kv"),
    ("auth", "kv"),
    ("akia", "aws"), ("asia", "aws"),
    ("ghp_", "gh"), ("gho_", "gh"), ("ghu_", "gh"), ("ghs_", "gh"), ("ghr_", "gh"), ("github_pat_", "gh"),
    ("sk-", "sk"),
    ("sk_live_", "stripe"), ("sk_test_", "stripe"), ("rk_live_", "stripe"), ("rk_test_", "stripe"),
    ("xoxa-", "slack"), ("xoxb-", "slack"), ("xoxp-", "slack"), ("xoxr-", "slack"), ("xoxs-", "slack"),
    ("xoxe-", "slack"), ("xapp-", "slack"),
    ("eyj", "jwt"),
    ("aiza", "google"),
    ("npm_", "npm"),
    ("://", "cs"),
]
_GROUP_ORDER = {g: i for i, g in enumerate(["pk", "ah", "kv", "be", "aws", "gh", "sk", "stripe", "slack", "jwt", "google", "npm", "cs"])}
_SCAN = [(word, _GROUP_ORDER[group], group) for word, group in _STEMS]
# str.lower() can change a string's length (one dotted capital I becomes two code points), which
# would shift every offset after it; this table folds only ASCII and is used when that happens.
_ASCII_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


def _ci(word: str) -> str:
    """Case-insensitive literal without the IGNORECASE flag."""
    return "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else re.escape(c) for c in word)


# The key-name stems of the "kv" rule, cased; must stay in step with the "kv" literals above.
_KEY_STEMS = [
    _ci("secret"),
    _ci("token"),
    _ci("passw") + "(?:" + _ci("or") + ")?" + _ci("d"),
    _ci("passphrase"),
    _ci("api") + "[_-]?" + _ci("key"),
    _ci("access") + "[_-]?" + _ci("key"),
    _ci("private") + "[_-]?" + _ci("key"),
    _ci("client") + "[_-]?" + _ci("secret"),
    _ci("auth"),
]
# Where a key's stem may end: at a non-letter (SECRET_KEY, token=, tokens:), or at a camelCase
# boundary (secretKey, authToken).  Not inside a word: tokenizer, secretary, author, AUTHORITY.
_STEM_END = r"[sS]?(?:(?![A-Za-z])|(?<=[a-z])(?=[A-Z]))"
_SEP = r"[\w.-]*[\"']?[ \t]*(?:=>|:=|={1,2}|:)"

# ---------------------------------------------------------------------------------------------
# Rules.  Each is (pattern, handler); a handler returns (replacement, kind) or None to decline,
# in which case later stems inside the same span still get their turn — a declined
# AUTH_URL=https://u:p@h still has its password caught by the connection-string rule at "://".
# ---------------------------------------------------------------------------------------------

# A whole block through its END line; or, when the END line is missing (a READ_FILE line limit
# cut it off), the header and every base64 body line that follows it, numbered or not.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----"
    r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----"
    r"|(?:(?:\r\n|\r|\n)[^\S\r\n]*(?:\d{1,7}[\t:.)])?[^\S\r\n]*[A-Za-z0-9+/=]{16,}[^\S\r\n]*(?=\r|\n|\Z))*)",
    re.DOTALL,
)
_EOL_SPLIT = re.compile(r"(\r\n|\r|\n)")
# The line-number prefix READ_FILE adds ("12\t", "3: "), kept on the emptied lines of a block.
_LNO = re.compile(r"[^\S\r\n]*\d{1,7}[\t:.)][^\S\r\n]*")

_AUTH_HEADER = re.compile(
    r"([Aa]uthorization[ \t]*:[ \t]*)([Bb]earer|[Bb]asic|[Tt]oken|[Dd]igest|[Aa]pi[Kk]ey)"
    r"([ \t]+)([^\s\"',;<>]{6,})"
)
# "Bearer" followed by something that is not just a word ("Bearer authentication" is prose).
_BEARER = re.compile(r"([Bb]earer[ \t]+)(?![A-Za-z]+(?![A-Za-z0-9._~+/=-]))([A-Za-z0-9._~+/=-]{8,})")

_KV = re.compile(
    "(?:" + "|".join(_KEY_STEMS) + ")" + _STEM_END + _SEP
    + r"[ \t]*(?:\"([^\"\r\n]*)\"|'([^'\r\n]*)'|([^\s\"',;&<>`]+))"
)
_KEY_TAIL = re.compile(r"[\w.-]*")
_KEY_REV = re.compile(r"[\w.-]*")             # applied to the reversed text before the stem
_URL_AUTHORITY_REV = re.compile(r"[^\s/@]*//:")  # "://host" reversed: the stem is in a URL host

# user:password@ after a scheme; the password cannot contain / or @ unencoded, so those end it.
_CONNECTION = re.compile(r"(://[^\s/:@\"'<>]{1,64}:)([^\s/@\"'<>]{1,256})(@)")

_SHAPES: dict[str, list[tuple[str, re.Pattern]]] = {
    "aws": [("aws-access-key", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"))],
    "gh": [("github-token", re.compile(r"(?<![A-Za-z0-9])(?:gh[opusr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})(?![A-Za-z0-9])"))],
    "sk": [
        ("anthropic-key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}")),
        ("openai-key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")),
    ],
    "stripe": [("stripe-key", re.compile(r"(?<![A-Za-z0-9])[sr]k_(?:live|test)_[A-Za-z0-9]{10,}(?![A-Za-z0-9])"))],
    "slack": [("slack-token", re.compile(r"(?<![A-Za-z0-9])(?:xox[abprse]|xapp)-[A-Za-z0-9-]{10,}"))],
    "jwt": [("jwt", re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+"))],
    "google": [("google-api-key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"))],
    "npm": [("npm-token", re.compile(r"(?<![A-Za-z0-9])npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])"))],
}
_ALL_SHAPES = [pair for pairs in _SHAPES.values() for pair in pairs]

# Values that a secret-named key may hold without holding a secret.
_PLAIN_WORDS = frozenset(
    """true false yes no on off null none nil undefined empty unset
    info debug warning warn error trace notice critical fatal verbose silent
    required optional enabled disabled default string integer number boolean object array
    bearer basic digest oauth oauth2 jwt apikey api_key api-key token secret password auth
    hs256 hs384 hs512 rs256 rs384 rs512 es256 sha1 sha256 sha512 md5 bcrypt scrypt argon2 pbkdf2
    session cookie header query plain simple unknown pending redacted hidden masked example
    placeholder changeit""".split()
)
# Key names whose LAST word says the value is not the secret itself (AUTH_USER, TOKEN_URL, ...).
_PLAIN_SUFFIXES = frozenset(
    """user username email url uri host hostname port path file filename dir name type method
    mode enabled disabled timeout ttl expiry expires expiration count length len size version
    scheme header prefix suffix algorithm alg issuer audience aud iss source ref location env
    label format required optional min max kind class status endpoint field column table""".split()
)
_NUMBERISH = re.compile(r"[+-]?\d{1,5}(?:\.\d+)?[A-Za-z%]{0,3}")
_PATHISH = re.compile(
    r"(?:/|\./|\.\./|~[/\\]|\\\\|[A-Za-z]:[\\/])"
    r"|.*\.(?:pem|key|p12|pfx|crt|cer|jks|keystore|json|ya?ml|toml|ini|cfg|conf|env|txt)$"
)
_URLISH = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
# A call or an index on the right of `=` is a computation (`strtok(line, " ")`, `line.split()[0]`, `parts[1]`), the
# shape of ordinary code assigned to a variable called token or password. A quoted literal or a bare run of
# characters does not match, so `token = "hunter2"` and `TOKEN=hunter22` stay secrets.
_CODEISH = re.compile(r"[A-Za-z_][\w.>:-]*[(\[]")   # the captured value stops at the first space or comma: `strtok(line,`
_PLACEHOLDER_CHARS = frozenset("x*.-_?#")
_SUFFIX_SPLIT = re.compile(r"[_.\-]+|(?<=[a-z0-9])(?=[A-Z])")


def _looks_placeholder(v: str) -> bool:
    """<your-key>, ${VAR}, {{ x }}, ********, a marker this module already wrote."""
    low = v.lower()
    if v[0] in "<${[({%":
        return True
    if set(low) <= _PLACEHOLDER_CHARS:
        return True
    return low.startswith(("your-", "your_", "your ", "insert-", "insert_", "replace-", "replace_"))


def _looks_plain(value: str) -> bool:
    """True when a value under a secret-named key is clearly not a secret."""
    v = value.strip()
    if len(v) < 6:
        return True
    if v.lower() in _PLAIN_WORDS:
        return True
    if _looks_placeholder(v):
        return True
    if _NUMBERISH.fullmatch(v):
        return True                      # 3600, 86400s, 12.5
    if _PATHISH.match(v):
        return True                      # /etc/x, C:\x, ./x, service-account.json
    if _URLISH.match(v):
        return True                      # a URL: the connection-string rule looks inside it
    if _CODEISH.match(v):
        return True                      # strtok(line, " "), parts[1]: code, not a credential
    return False


def _plain_key(key: str) -> bool:
    """True when the key's last word says the value is not the secret (AUTH_USER, TOKEN_URL)."""
    words = [w for w in _SUFFIX_SPLIT.split(key) if w]
    return bool(words) and words[-1].lower() in _PLAIN_SUFFIXES


def _kind_for_key(key: str, value: str) -> str:
    for kind, rx in _ALL_SHAPES:
        if rx.search(value):
            return kind
    k = key.lower()
    if "aws" in k and "secret" in k:
        return "aws-secret-key"
    if "aws" in k and "access" in k:
        return "aws-access-key"
    if "passw" in k or "passphrase" in k:
        return "password"
    if "private" in k and "key" in k:
        return "private-key"
    return "generic-secret"


def _h_private_key(m: re.Match, text: str):
    parts = _EOL_SPLIT.split(m.group(0))      # text, eol, text, eol, ..., text
    out = [marker("private-key")]
    for i in range(1, len(parts)):
        if i % 2:
            out.append(parts[i])              # the line ending, exactly as it was
        else:
            lno = _LNO.match(parts[i])
            out.append(lno.group(0) if lno else "")
    return "".join(out), "private-key"


def _h_auth_header(m: re.Match, text: str):
    if _looks_placeholder(m.group(4)):
        return None
    scheme = m.group(2).lower()
    kind = "bearer" if scheme == "bearer" else "basic-auth" if scheme == "basic" else "auth-header"
    return m.group(1) + m.group(2) + m.group(3) + marker(kind), kind


def _h_bearer(m: re.Match, text: str):
    return m.group(1) + marker("bearer"), "bearer"


def _h_kv(m: re.Match, text: str):
    g = 1 if m.group(1) is not None else 2 if m.group(2) is not None else 3
    value = m.group(g)
    if _looks_plain(value):
        return None
    start = m.start()
    # Look back from the stem, reading the window reversed so both checks are anchored matches
    # (a $-anchored search over the window backtracks from every position and cost more than
    # the rest of the module put together).
    rev = text[max(0, start - 96):start][::-1]
    if _URL_AUTHORITY_REV.match(rev):
        return None                           # https://tokenserver:9000/ — a host and a port
    whole = m.group(0)
    # The match starts at the stem; the rest of the key name (AWS_ in AWS_SECRET_ACCESS_KEY) is
    # just before it and stays in the output untouched.  Rebuild the full name for the decisions.
    key = (_KEY_REV.match(rev).group(0)[::-1] + _KEY_TAIL.match(whole).group(0)).lstrip("-")
    if _plain_key(key):
        return None
    kind = _kind_for_key(key, value)
    vs, ve = m.start(g) - start, m.end(g) - start
    return whole[:vs] + marker(kind) + whole[ve:], kind


def _h_connection(m: re.Match, text: str):
    if _looks_placeholder(m.group(2)):
        return None
    return m.group(1) + marker("connection-string") + m.group(3), "connection-string"


def _h_shape(kind: str):
    def handler(m: re.Match, text: str):
        return marker(kind), kind
    return handler


_RULES: dict[str, list[tuple[re.Pattern, Callable]]] = {
    "pk": [(_PRIVATE_KEY, _h_private_key)],
    "ah": [(_AUTH_HEADER, _h_auth_header)],
    "kv": [(_KV, _h_kv)],
    "be": [(_BEARER, _h_bearer)],
    "cs": [(_CONNECTION, _h_connection)],
}
for _name, _pairs in _SHAPES.items():
    _RULES[_name] = [(rx, _h_shape(kind)) for kind, rx in _pairs]
assert set(_RULES) == set(_GROUP_ORDER) == {group for _, group in _STEMS}


def _candidates(text: str) -> list[tuple[int, int, str]]:
    """(position, group order, group) for every stem occurrence, in text order."""
    low = text.lower()
    if len(low) != len(text):
        low = text.translate(_ASCII_LOWER)
    hits: list[tuple[int, int, str]] = []
    find = low.find
    for word, order, group in _SCAN:
        i = find(word)
        while i >= 0:
            hits.append((i, order, group))
            i = find(word, i + 1)
    hits.sort()
    return hits


def _process(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()
    out: list[str] = []
    cursor = 0
    for s, _, group in _candidates(text):
        if s < cursor:
            continue
        for rx, handler in _RULES[group]:
            m = rx.match(text, s)
            if m is None:
                continue
            result = handler(m, text)
            if result is None:
                continue
            replacement, kind = result
            out.append(text[cursor:s])
            out.append(replacement)
            cursor = m.end()
            counts[kind] += 1
            break
    if not out:
        return text, counts
    out.append(text[cursor:])
    return "".join(out), counts


def _coerce(text) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, (bytes, bytearray)):
        return bytes(text).decode("utf-8", "replace")
    try:
        return str(text)
    except Exception:
        return ""


def redact(text) -> str:
    """Return the text with every secret value replaced by a [redacted:<kind>] marker.

    Key names, header names, connection-string users and hosts, the number of lines and the line
    endings are all kept. Idempotent: redacting a redacted text changes nothing.
    """
    text = _coerce(text)
    try:
        return _process(text)[0]
    except Exception:
        return text


def findings(text) -> list[tuple[str, int]]:
    """The kinds of secret found and how many of each, most frequent first — never the values."""
    text = _coerce(text)
    try:
        counts = _process(text)[1]
    except Exception:
        return []
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def has_secret(text) -> bool:
    """True when redact() would change the text."""
    return bool(findings(text))


def scrub(obj, depth: int = 0):
    """redact() over a tool result of any shape: strings in place, lists and dicts element by
    element (keys kept), anything else untouched. The executor calls this on every Result."""
    if isinstance(obj, str):
        return redact(obj)
    if depth > 12:
        return obj
    if isinstance(obj, list):
        return [scrub(x, depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(scrub(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return {k: scrub(v, depth + 1) for k, v in obj.items()}
    return obj
