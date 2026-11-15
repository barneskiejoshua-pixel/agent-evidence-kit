"""Grounded answers for conversation: the notes are the authority, the model is the writer.

A 4B model given a primer as prose context still "elaborates": it invented a Frankfurt case called
"willing woodkote", said RSA "relies on primes being sparse", chose e "relatively prime to n"
(drockclaude 2026-09-25). Prompting against it ("do not hallucinate") is a request; this module makes
it a check, the way expect.py checks a task's claims:

  1. fact_table(): each retrieved primer is cut into numbered facts [F1] … [Fn].
  2. The model answers from that table (GROUNDED), tagging a sentence that states a fact with its id.
  3. validate(): every high-risk detail in the answer (a proper name, a number or year, a quotation, a
     term put in quotes) must appear in the facts or in the question. Tags must name real facts.
  4. The caller regenerates once with the list of unsupported details; compile_answer() then drops any
     sentence that still carries one, strips the tags and adds where the answer came from.

Modes: GROUNDED (notes found), GENERAL (no note: the model answers from what it knows, told to avoid
specifics it is unsure of, and the answer says it is general knowledge), UNKNOWN (the model says it
cannot answer reliably; passed through as is).

Standard library only; deterministic; no model calls here.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

GROUNDED, GENERAL, UNKNOWN = "grounded", "general", "unknown"
UNKNOWN_LINE = "I don't have enough information to answer that reliably."
GENERAL_TAG = "(general knowledge: DRock has no note on this topic, so treat specifics with care)"
MAX_FACTS = 40


@dataclass
class Fact:
    id: str
    text: str
    source: str          # the primer's title
    section: str = ""


def _sentences(par: str) -> list[str]:
    """Split prose into sentences without breaking "e.g.", "300 BCE.", "p1*...*pn", decimals."""
    par = re.sub(r"\s+", " ", par).strip()
    if not par:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", par)
    out: list[str] = []
    for p in parts:
        if out and (re.search(r"\b(?:e\.g|i\.e|etc|vs|approx|ca|Dr|St|Mr|Mrs|Ms)\.$", out[-1]) or len(p) < 12):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def fact_table(cards: list[dict], start: int = 1) -> list[Fact]:
    """The primers cut into numbered facts: a bullet stays whole, prose is split into sentences, and
    each fact keeps the heading it sat under (a 'Common misconceptions' line is a myth being corrected,
    not a fact to repeat)."""
    facts: list[Fact] = []
    n = start
    for c in cards:
        section, buf = "", []
        header = None

        def flush():
            nonlocal n
            for s in _sentences(" ".join(buf)):
                if len(facts) < MAX_FACTS:
                    facts.append(Fact(f"F{n}", s, c.get("task", ""), section))
                    n += 1
            buf.clear()

        for line in (c.get("body") or "").splitlines():
            s = line.strip()
            if not s:
                flush()
                continue
            if s.startswith("#"):
                flush()
                section = s.lstrip("#").strip()
                continue
            if section.lower().startswith("a good way to answer"):
                continue   # advice to the writer, not a fact about the world
            if s.startswith("|"):
                # a table row is one fact, read with its header ("task: list dir | Windows: dir | Linux: ls")
                flush()
                cells = [x.strip() for x in s.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", x) for x in cells if x):
                    continue
                if header is None or header[0] == "":
                    header = cells
                    continue
                text = "; ".join(f"{h}: {v}" if h else v for h, v in zip(header, cells) if v)
                if len(facts) < MAX_FACTS:
                    facts.append(Fact(f"F{n}", text, c.get("task", ""), section))
                    n += 1
                continue
            header = None if not s.startswith("|") else header
            if re.match(r"^(?:[-*]|\d+[.)])\s+", s):
                flush()
                text = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", s).replace("**", "")
                if len(facts) < MAX_FACTS:
                    facts.append(Fact(f"F{n}", text, c.get("task", ""), section))
                    n += 1
                continue
            buf.append(s.replace("**", ""))
        flush()
    return facts


def render_table(facts: list[Fact]) -> str:
    lines, last = [], None
    for f in facts:
        if f.source != last:
            lines.append(f"SOURCE: {f.source}")
            last = f.source
        sec = f" ({f.section})" if f.section and f.section.lower() not in ("in one paragraph",) else ""
        lines.append(f"[{f.id}]{sec} {f.text}")
    return "\n".join(lines)


# Prefill is what a chat answer costs: 3,698 prompt tokens took 24.27 s of the 30.67 s spent on
# "/chat what is the boiling point of water at sea level in fahrenheit?" (latency.jsonl 2026-10-14),
# and 5,967 of those characters were a 32-fact table holding one fact the answer needed. select_facts
# ranks the table against the question before it is rendered. The selected list is what the model sees
# AND what validate() checks, so a dropped fact cannot be credited or cut on evidence the model lacked.
_FACT_STOPWORDS = frozenset("""a an and are as at be been being by can could did do does for from give has have
how i in into is it me of on or please say tell that the their them there these this to was were what when where
which who why will with would you your""".split())
_FACT_SYNONYMS = {"fahrenheit": "f", "celsius": "c", "centigrade": "c"}
# A broad ask wants the table, not a slice of it, and so does a question with two parts.
_BROAD_Q = re.compile(r"\b(compare|list|all|each|overview|summari[sz]e|explain|describe|pros and cons"
                      r"|differences|similarities)\b", re.I)
_TWO_PART_Q = re.compile(r"\b(?:and|versus|vs\.?)\b.+\b(?:and|versus|vs\.?)\b", re.I)
_MISCONCEPTION = re.compile(r"\b(?:misconception\w*|myth\w*)\b")


def _fact_tokens(text: str) -> list[str]:
    """Ordered content words for matching: lowercased, stopwords dropped, lightly de-inflected."""
    out: list[str] = []
    for raw in re.findall(r"[a-z]+|\d+(?:\.\d+)?", (text or "").lower()):
        tok = _FACT_SYNONYMS.get(raw, raw)
        if tok in _FACT_STOPWORDS:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("ing") and len(tok) > 5:
            tok = tok[:-3]
        elif tok.endswith("ed") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
            tok = tok[:-1]
        if tok:
            out.append(tok)
    return out


def _is_misconception_section(section: str) -> bool:
    """'Common misconceptions', 'Myths vs facts', 'Misconceptions to avoid': one heading, many spellings."""
    return bool(_MISCONCEPTION.search(re.sub(r"[^a-z0-9]+", " ", (section or "").lower())))


def who_when_question(question: str) -> bool:
    """The question asks who or when, so the rules switch to the brief answer."""
    return bool(_WHO_Q.match(question or ""))


def select_facts(facts: list[Fact], question: str, limit: int | None = None,
                 who_when: bool = False) -> list[Fact]:
    """The facts that bear on the question, in table order, with their ids untouched.

    Ranked by idf-weighted term overlap plus a bonus for a pair of words the question itself puts
    side by side. A misconception correction that touches the question is kept outside the limit:
    without it in the table the model loses the correction and can state the myth as true. A broad
    or two-part question, or a match on a single term, keeps the whole table - a smaller prompt is
    not a win if the fact the answer needed is the one dropped."""
    n_keep = limit if limit is not None else (6 if who_when else 10)
    if n_keep < 1:
        raise ValueError("limit must be at least 1")
    if len(facts) <= n_keep:
        return list(facts)
    q_tokens = _fact_tokens(question)
    q_terms = set(q_tokens)
    if not q_terms or _BROAD_Q.search(question or "") or _TWO_PART_Q.search(question or ""):
        return list(facts)
    if len(re.findall(r"[?;]", question or "")) > 1:
        return list(facts)

    fact_tokens = [_fact_tokens(f.text) for f in facts]
    fact_terms = [set(t) for t in fact_tokens]
    pairs = set(zip(q_tokens, q_tokens[1:]))
    freq: dict[str, int] = {}
    for terms in fact_terms:
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    idf = {t: math.log((len(facts) + 1) / (freq.get(t, 0) + 1)) + 1.0 for t in q_terms}

    scored: list[tuple[float, int, int]] = []
    musts: set[int] = set()
    for i, f in enumerate(facts):
        overlap = q_terms & fact_terms[i]
        if overlap and _is_misconception_section(f.section):
            musts.add(i)
        if not overlap:
            continue
        score = sum(idf[t] for t in overlap)
        score += 2.0 * len(pairs & set(zip(fact_tokens[i], fact_tokens[i][1:])))
        scored.append((score, i, len(overlap)))
    if not scored:
        return list(facts)
    best = max(s for s, _, _ in scored)
    if max(n for _, _, n in scored) == 1:   # one shared word is a coincidence, not a match
        return list(facts)
    keep = sorted(((s, i) for s, i, n in scored if n >= 2 or s >= best * 0.5),
                  key=lambda it: (-it[0], it[1]))
    chosen = set(musts) | set([i for _, i in keep if i not in musts][:n_keep])
    return [f for i, f in enumerate(facts) if i in chosen]


_ANSWER_FULLY = """ANSWER FULLY: use every fact in the table that bears on the question (a good answer uses 5 to 10 of them),
in 2 to 4 short paragraphs of plain sentences. For a question without one settled answer, give each main
view the table lists and who held it."""
# A who/when question wants the name or the date. Told to use 5 to 10 facts, the model wrote 336 tokens around
# "Who first proved that the square root of two is irrational?", credited Euler (the table's 1737 fact is about
# e), and with the rewrite the answer took 52 s (console 2026-09-27).
_ANSWER_BRIEFLY = """ANSWER BRIEFLY: the question asks who or when. Give the person or the date in your first sentence only
if a fact in the table says that person or date did exactly what the question asks; if no fact does, say
that your notes do not say. Then add at most one short paragraph of the facts that bear on it."""
# A question whose answer is one value got the 2-to-4-paragraph rules: "/chat what is the boiling point of
# water at sea level in fahrenheit?" generated 177 tokens, 12.4 s of a 29.8 s turn, to say 212 (A/B
# 2026-10-14, three runs an arm). The value is the answer; the rest is the model filling a quota.
_ANSWER_VALUE = """ANSWER THE VALUE: the question asks for one value. Give it in your first sentence, with its
units, and put the id of the fact it comes from in brackets. If no fact in the table gives that value, say
that your notes do not have it. Then stop, or add one short sentence only if a fact qualifies the value."""
GROUNDED_RULES = """KNOWLEDGE STATUS: GROUNDED. Answer the question from the SOURCE FACTS below.
""" + _ANSWER_FULLY + """ Never say something is absent when a fact in the table covers it.
ALLOWED: state facts the table contains; make simple inferences from them; explain in plain words.
FORBIDDEN: a name, date, number, quotation, experiment, case or technical term that is not in the table;
turning an inference into a documented fact; filling a gap with a plausible detail.
A fact listed under "Common misconceptions" is a myth being corrected: never state the myth as true.
When the table gives a proof or a mechanism, follow its steps in order.
After each sentence that states a fact from the table, put its id in brackets, e.g. [F3] or [F3][F7].
If the table does not cover part of the question, say that part is not in your notes. Answer only this question."""

# "what is the boiling point of water at sea level in fahrenheit?", "what is the pH of pure water?",
# "how hot is the surface of the sun?", "how many bones are in the human body?" ask for a value. The words
# after "what is the" that name a question of kind rather than quantity are excluded, so "what is the
# difference between heat and temperature?" and "what is the meaning of this?" keep the full answer, and so
# does any question with no "of/at/in/on/for" phrase after the noun ("what is a solution?").
_HOW_VALUE = re.compile(r"^how\s+(?:hot|cold|high|low|long|far|deep|heavy|much|many|fast|old|tall|wide|thick|big)\b", re.I)
_WHAT_HEAD = re.compile(r"^what(?:'s|s|\s+is|\s+are|\s+was|\s+were)\s+(?:the\s+|a\s+|an\s+)?", re.I)
# The head noun that names a kind of question rather than a quantity. "what is the role of insulin in the
# body?" and "what is the best film of all time?" are the same grammar as "what is the boiling point of
# water?" and want the paragraphs, so the head word decides, not the shape.
_NOT_A_VALUE = frozenset("""difference differences meaning definition purpose reason cause causes process
effect effects relationship role function history significance importance point best worst advantage
advantages disadvantage disadvantages benefit benefits""".split())
_UNIT_NOUN = re.compile(r"(?:number|value|date|year|time|temperature|pressure|speed|mass|volume|distance"
                        r"|age|height|length|weight|ph)\b", re.I)
_NOUN_OF = re.compile(r"[\w-]+(?:\s+[\w-]+){0,5}\s+(?:of|at|in|on|for)\b", re.I)
# A value asked for together with an explanation needs both, so it keeps the full answer.
_VALUE_AND_MORE = re.compile(r"\b(?:and|but)\s+(?:why|how|what|when|where)\b|\bexplain\b|\balso\s+tell\b", re.I)


def value_question(question: str) -> bool:
    """The question asks for one value: a measurement, a count, a quantity with units."""
    q = (question or "").strip()
    if not q or _VALUE_AND_MORE.search(q):
        return False
    if _HOW_VALUE.match(q):
        return True
    head = _WHAT_HEAD.match(q)
    if not head:
        return False
    rest = q[head.end():]
    first = re.match(r"[\w-]+", rest)
    if not first or first.group(0).lower() in _NOT_A_VALUE:
        return False
    return bool(_UNIT_NOUN.match(rest) or _NOUN_OF.match(rest))


def grounded_rules(question: str) -> str:
    if _WHO_Q.match(question or ""):          # who/when keeps precedence: the name or the date is the answer
        return GROUNDED_RULES.replace(_ANSWER_FULLY, _ANSWER_BRIEFLY)
    # DROCK_VALUE_RULES=0 sends the paragraph rules to a value question, the before side of the A/B.
    if value_question(question) and os.environ.get("DROCK_VALUE_RULES", "1") != "0":
        return GROUNDED_RULES.replace(_ANSWER_FULLY, _ANSWER_VALUE)
    return GROUNDED_RULES


# The escape hatch is for a question that needs a fact the model lacks. Put last and unconditioned, the 4B model
# took it on questions of meaning: "why are humans on earth?" -> "I don't have enough information to answer that
# reliably." (drockclaude 2026-09-30, twice). A question of why, meaning, purpose or value has no fact to fetch;
# the views are the answer, and the prompt now says so before it offers the hatch.
GENERAL_RULES = """KNOWLEDGE STATUS: GENERAL. DRock has no note on this topic, so answer from what you know.
Keep to what is well established. Leave out names, dates, exact numbers, quotations and study results
unless you are certain of them; say "roughly", "probably" or "I think" where you are not. Say what is
uncertain or debated.
A question of why, meaning, purpose, value or opinion has no single fact to look up: answer it anyway, in 2 or 3
short paragraphs of plain talk that give the main views (what the sciences say, what different philosophies or
traditions say), each with a concrete example, and where they disagree. Never answer such a question by saying you
lack information.
Only when the question needs a specific fact you do not have (a date, a number, a name, a quotation, what a
particular document or event contained), reply with exactly this sentence and nothing else: """ + UNKNOWN_LINE

# The retry when the model took the hatch anyway on an open question (chat._stream_out holds the refusal back and
# sends the question once more under these rules): same status, no hatch, because the views ARE the answer.
PERSPECTIVES_RULES = """KNOWLEDGE STATUS: GENERAL. This is an open question: there is no single fact to look up, and
"I don't have enough information" is not an answer to it. Write 2 or 3 short paragraphs of plain talk that give the
main views people hold (what the sciences say, what different philosophies or traditions say), each with a concrete
example, and where they disagree. Leave out names, dates and numbers you are not certain of. If the question is about a particular person,
place or event you know nothing about, say so in one sentence and then give what can be said in general."""


# ---------- the validator ----------

# Capitalised words that are ordinary English at the start of a sentence, or common enough in any
# answer that they carry no claim.
_COMMON = set("""the a an this that these those it its it's there their they them then than thus so but and or yet
    if in on at by for from of to with without as when where while which who whom whose what why how whether
    however moreover furthermore therefore instead also still even only just often sometimes perhaps maybe
    firstly secondly thirdly finally first second third next last one two three some many most much more few
    all any each every both either neither no not yes here now today we you i he she my your our his her
    let imagine consider think suppose because since although though unlike like just rather overall
    in fact for example ultimately essentially basically importantly interestingly crucially notably simply
    drock ai okay ok well sure great good right yes indeed actually honestly clearly equally similarly
    god earth sun moon english latin greek universe other others another despite subsequent subsequently several
    although during after before later earlier early modern recent recently today yet whereas meanwhile nevertheless
    nonetheless additionally alternatively conversely consequently accordingly specifically generally typically
    many much certain various such each whether whatever whichever whoever everyone someone nobody people""".split())
_MONTHS = set("january february march april may june july august september october november december".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_NUM = re.compile(r"(?<![A-Za-z_\d.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d])")
_QUOTED = re.compile("[\"“‘]([^\"”’\n]{3,80})[\"”’]")   # a straight ' is an apostrophe, never a quote
_GARBLED = re.compile(r"\b(?:by|and|from|including|like|such\s+as)\s+(?:and|or|,|;|\.|respectively)\b|\bby\s+and\b"
                      r"|\b(?:by|including|like|such\s+as|championed\s+by|held\s+by)\s+(?:a|an|some|certain|several)\s+"
                      r"(?:mathematician|philosopher|scientist|thinker|thinkers|figures?|researchers?|person|people)s?\b"
                      r"|\bsuch\s+as\s+some\b|\bthinkers\s+like\s+some\b"
                      # the table itself cited in prose ("detailed in facts through", "according to the facts")
                      r"|\b(?:in|by|from|see|per)\s+(?:the\s+)?facts?\b(?:\s+(?:through|F?\d+))?"
                      r"|\baccording\s+to\s+(?:the\s+|our\s+|my\s+)?(?:facts?|table|notes)\b", re.I)
_WHO_Q = re.compile(r"^\s*(?:who|when|in\s+what\s+year|which\s+(?:person|scientist|mathematician|philosopher))\b", re.I)
_TAG = re.compile(r"\s*\[(F\d+(?:\s*(?:\]\s*\[|,|;|and)\s*F\d+)*)\]")   # [F3], [F3][F7], [F1, F7, F8]
_TAG_ID = re.compile(r"F\d+")


@dataclass
class Issue:
    sentence: str
    kind: str      # name | number | quote | tag
    token: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    cited: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.issues

    def terms(self) -> list[str]:
        seen: list[str] = []
        for i in self.issues:
            if i.kind != "tag" and i.token not in seen:
                seen.append(i.token)
        return seen


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("’", "'")).strip()


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for par in re.split(r"\n\s*\n|\n(?=\s*[-*\d])", text or ""):
        out += _sentences(par) if par.strip() else []
    return out


def validate(answer: str, facts: list[Fact], question: str = "", extra: str = "") -> Report:
    """Every high-risk detail of `answer` checked against the fact table (and the question, and `extra`:
    what the user said earlier). A detail is supported when it appears, case-insensitively, in that text."""
    known = _norm(" ".join(f.text + " " + f.source for f in facts) + " " + question + " " + extra)
    known_words = set(known.split())
    known_nums = {n.replace(",", "") for n in _NUM.findall(" ".join(f.text for f in facts) + " " + question + " " + extra)}
    ids = {f.id for f in facts}
    rep = Report()
    for s in split_sentences(answer):
        for m in _TAG.finditer(s):
            for i in _TAG_ID.findall(m.group(1)):
                if i in ids:
                    rep.cited.add(i)
                else:
                    rep.issues.append(Issue(s, "tag", i))
        plain = _TAG.sub("", s)
        # a name blanked out by the rewrite ("first by and, respectively", "including a mathematician"):
        # the model removed a detail instead of saying the notes do not give it (replay 2026-09-25)
        g = _GARBLED.search(plain)
        if g:
            rep.issues.append(Issue(s, "garbled", g.group(0).strip()))
            continue
        # quotations and terms put in quotes
        for m in _QUOTED.finditer(plain):
            q = _norm(m.group(1))
            if q and q not in known and not all(w in known_words or w in _COMMON for w in q.split()):
                rep.issues.append(Issue(s, "quote", m.group(1)))
        # numbers and years (small counts and the ones in the notes or question are fine)
        for m in _NUM.finditer(plain):
            n = m.group(1).replace(",", "")
            try:
                small = float(n) <= 12
            except ValueError:
                small = False
            if not small and n not in known_nums:
                rep.issues.append(Issue(s, "number", m.group(1)))
        # proper names: a capitalised word the notes and the question never use
        words = list(_WORD.finditer(plain))
        for k, m in enumerate(words):
            w = m.group(0)
            if not w[0].isupper() or len(w) < 3:
                continue
            lw = w.lower().strip("'-")
            if lw.endswith("'s"):
                lw = lw[:-2]
            if lw in known_words or lw in _COMMON or lw.rstrip("s") in known_words:
                continue
            if k == 0 or plain[max(0, m.start() - 2):m.start()].strip() in (".", ":", "!", "?", "(", "–", "-", '"', "“"):
                # a sentence opener: only a name when it is not an ordinary word (-ly adverbs, common words)
                if lw.endswith("ly") or lw in _COMMON or lw not in _MONTHS and not _looks_like_name(w, plain, m.end()):
                    continue
            rep.issues.append(Issue(s, "name", w))
        if not any(i.sentence == s for i in rep.issues):
            why = _pairing(plain, facts, question) or _credit(plain, facts, question) or _negation(plain, facts)
            if why:
                rep.issues.append(Issue(s, "pairing", why))
    return rep


# ---------- negation: the notes say X lacks Y, the answer gives X a Y ----------
# q100 rerun (2026-09-27), "what is the difference between a virus and bacteria?": the notes said "Prokaryotes,
# such as bacteria and archaea, are small and lack a nucleus"; the answer said bacteria are "characterized by a
# nucleus and internal compartments -- a feature known as being eukaryotic". Every word was in the notes.
_NEG_VERB = re.compile(r"(?:\b(?:lacks?|lacking|have\s+no|has\s+no|without|do\s+not\s+have|does\s+not\s+have|don't\s+have|"
                       r"doesn't\s+have|cannot|can't|never\s+have|contain\s+no|contains\s+no)|(?:^|[:,]\s*|\b(?:small|and|with),?\s+)no)"
                       r"\s+(?:a\s+|an\s+|any\s+|the\s+)?([a-z][\w-]{3,})", re.I)
_NEG_WORDS = re.compile(r"\b(?:no|not|never|lacks?|lacking|without|unlike|whereas|while|but|however|instead|rather|except|"
                        r"absent|neither|nor|n't|cannot|contrast|contrary|only)\b|n't\b", re.I)
_SUBJ_STOP = {"such", "small", "large", "larger", "smaller", "also", "most", "many", "some", "which", "that", "these",
              "those", "they", "their", "have", "with", "from", "into", "more", "less", "very", "each", "both", "other"}


def _negation(plain: str, facts: list[Fact]) -> str | None:
    s = plain.lower()
    if _NEG_WORDS.search(s):
        return None
    for f in facts:
        for sent in re.split(r"(?<=[.;])\s+", f.text):
            for m in _NEG_VERB.finditer(sent):
                obj = m.group(1).lower()
                subj = [w for w in re.findall(r"[a-z][a-z-]{3,}", sent[:m.start()].lower()) if w not in _SUBJ_STOP]
                if not re.search(rf"\b{re.escape(obj)}\w*", s):
                    continue
                hit = next((w for w in subj if re.search(rf"\b{re.escape(w.rstrip('s'))}s?\b", s)), None)
                if hit:
                    return f"the notes say {hit} lack {obj} ({sent.strip()[:90]})"
    return None


# ---------- pairing: supported tokens, wrongly combined ----------
# "Lambert later provided an independent proof in 1761" (answering who proved sqrt(2) irrational): Lambert
# and 1761 are both in the notes, together even, but the notes say Lambert proved pi irrational. Each
# token passes on its own; the claim is the combination. Two checks:
#   name-date   a name and a number in one sentence must appear together in one fact;
#   attribution "NAME proved/showed/argued/... OBJECT" must match a fact that mentions NAME, by the
#               object's own specific words, or, when the object is vague ("an independent proof",
#               "similar conclusions"), by the question's specific subject words.
_ATTRIB = re.compile(r"\b([A-Z][a-z]{2,}(?:'s)?)\b((?:\s+\w+){0,3}?)\s+(proved|proves|proven|showed|shows|shown|demonstrated|"
                     r"discovered|found|argued|argues|proposed|proposes|provided|provides|developed|established|"
                     r"published|introduced|derived|gave|formulated|coined|invented|refuted|disproved|arrived|reached|came|obtained|achieved|solved|confirmed|completed|answered|settled|worked|studied)\b((?:\s+[\w\-']+){1,9})")
_POSSESSIVE = re.compile(r"\b([A-Z][a-z]{2,})['’]s\s+(proof|work|result|argument|theorem|idea|view|discovery|experiment|"
                         r"method|approach|contribution|finding|demonstration|conclusion)s?\b")
_VAGUE = set("""a an the his her their its this that it same similar independent own earlier later first another further
    new original different related work works result results proof proofs conclusion conclusions idea ideas view
    views version versions argument arguments approach method way ways line reasoning theory answer answers thing
    things something much more some also later lead case cases point points""".split())
_STOPWORDS = _COMMON | _VAGUE | set("is are was were be been being of to in on for by with as at from into about than".split())


def _stem(w: str) -> str:
    return w.lower()[:5]


def _content(words) -> set[str]:
    return {_stem(w) for w in words if len(w) >= 4 and w.lower() not in _STOPWORDS and not w.isdigit()}


_SQRT = re.compile(r"(?:the\s+)?square\s+root\s+of\s+(\w+)|sqrt\s*\(\s*(\w+)\s*\)|√\s*(\w+)", re.I)
_NUMWORD = {"two": "2", "three": "3", "five": "5", "ten": "10"}


def _toks(text: str) -> set[str]:
    """Content stems of a text, with the spellings of one idea made one token ("square root of two",
    "sqrt(2)" and the radical sign all become sqrt2)."""
    text = _SQRT.sub(lambda m: " sqrt" + _NUMWORD.get((m.group(1) or m.group(2) or m.group(3)).lower(),
                                                      (m.group(1) or m.group(2) or m.group(3))) + " ", text)
    return _content(re.findall(r"[A-Za-z][\w\-']*", text))


def _qanchors(question: str, texts: list[str], drop: frozenset[str] = frozenset()) -> set[str]:
    """The question's most specific words: those found in the fewest facts (and in at least one). A fact
    that credits someone with the question's subject has to contain all of them."""
    df = {w: sum(w in _toks(t) for t in texts) for w in _toks(question) - drop}
    df = {w: n for w, n in df.items() if n >= 1}
    if not df:
        return set()
    low = min(df.values())
    return {w for w, n in df.items() if n == low}


def _pairing(sentence: str, facts: list[Fact], question: str) -> str | None:
    texts = [f.text for f in facts]
    low = [t.lower() for t in texts]
    names = [m.group(0) for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", sentence)
             if m.group(0).lower() not in _COMMON and any(m.group(0).lower() in t for t in low)]
    nums = [n.replace(",", "") for n in _NUM.findall(sentence)]
    nums = [n for n in nums if not (n.replace(".", "").isdigit() and float(n) <= 12)]
    q = question.lower()
    # name-date
    for n in names:
        for d in nums:
            if d in q and n.lower() in q:
                continue
            if not any(n.lower() in t and d in t.replace(",", "") for t in low):
                return f"{n} + {d} (no fact pairs them)"
    # attribution
    # possessive attribution with no verb: "Euler's proof followed the same line" (about the question's subject)
    for m in _POSSESSIVE.finditer(sentence):
        who = m.group(1)
        if who.lower() in _COMMON or not any(who.lower() in t for t in low):
            continue
        rest = _toks(sentence[m.end():m.end() + 80])
        own = [_toks(t) for t in low if who.lower() in t]
        subject = _qanchors(question, texts)
        # about the question's own subject ("Euler's proof ... if sqrt(2) were rational"): the person's fact
        # must carry that subject; a shared generic word is not enough (replay 2026-09-25)
        if subject and subject & (rest | _toks(sentence)):
            if not any(subject <= o for o in own):
                return f"{who}'s {m.group(2)} (no fact ties {who} to this question)"
            continue
        if rest and any(rest & o for o in own):
            continue
        if subject and not any(subject <= o for o in own):
            return f"{who}'s {m.group(2)} (no fact ties {who} to this question)"
    for m in _ATTRIB.finditer(sentence):
        who = m.group(1).replace("'s", "")
        if who.lower() in _COMMON or not any(who.lower() in t for t in low):
            continue
        own = [t for t in low if who.lower() in t]
        obj = re.findall(r"[A-Za-z][\w\-']*", m.group(4))
        anchors = _content(obj)
        if not anchors:
            # vague object: it is whatever the question is about; use the question's specific words
            # (the ones few facts share), so "irrational" alone does not tie Lambert to sqrt(2)
            anchors = _qanchors(question, texts)
            if not anchors:
                continue
            if not any(anchors <= _toks(t) for t in own):
                return f"{who} {m.group(3)} {' '.join(obj[:5])} (no fact ties {who} to this question)"
            continue
        if not any(anchors & _toks(t) for t in own):
            return f"{who} {m.group(3)} {' '.join(obj[:5])} (no fact about {who} says that)"
    return None


# ---------- credit: the question's own subject given to someone the notes never credit with it ----------
# Console 2026-09-27, "Who first proved that the square root of two is irrational?" (the notes name no one): "The
# first to prove this was Euler, who established the irrationality of √2 in 1737." Euler and 1737 share a fact (e
# irrational), so name-date passed, and _ATTRIB needs "Euler proved ..." word order. In the answer to a credit
# question, a sentence that restates the subject, or says nothing the notes attach to the person ("the first to
# prove this was Euler"), needs a fact clause that names the person with the subject, whatever the word order.
_CREDIT_Q = re.compile(r"^\s*(?:who|when|in\s+what\s+year|which\s+(?:person|scientist|mathematician|philosopher))\b"
                       r"|\b(?:first|proved?|discover\w*|invent\w*|credited|attributed|came\s+up\s+with)\b", re.I)
_CREDIT_CUE = re.compile(r"\b(?:first|earliest|originally|prov(?:e[sdn]?|ing|ided)|proofs?|show(?:ed|n|s)?|demonstrat\w*|"
                         r"discover\w*|establish\w*|found|credit\w*|attribut\w*|invent\w*|introduc\w*|deriv\w*|"
                         r"formulat\w*|publish\w*|coined|gave|develop\w*)\b", re.I)
_CREDIT_WORDS = frozenset(_stem(w) for w in """first earliest originally prove proved proves proven proving provided proof
    proofs show showed shown shows demonstrated discovered discovery established found credited credit attributed
    invented invention introduced derived formulated published coined gave developed""".split())
_DENIAL = re.compile(r"\b(?:not|never|nobody|neither|nor|unknown|unclear|uncertain)\b|\bno\s+one\b|n't\b", re.I)
_SHORT_STOP = set("a an i is it of to in on at by as or be we he so if no do up me my us am vs".split())


def _short(text: str) -> set[str]:
    """One- and two-letter symbols that _toks leaves out: "Euler showed that e is irrational" is about e."""
    return {w for w in re.findall(r"(?<![a-z0-9'])[a-z]{1,2}(?![a-z0-9'])", text.lower()) if w not in _SHORT_STOP}


def _credit(sentence: str, facts: list[Fact], question: str) -> str | None:
    cue = _CREDIT_CUE.search(sentence)
    if not cue or not _CREDIT_Q.search(question or ""):
        return None
    deny = _DENIAL.search(sentence)
    if deny and deny.start() < cue.start():
        return None                      # "Euler did not prove it", "nobody knows who first proved it"
    texts = [f.text for f in facts]
    subject = _qanchors(question, texts, drop=_CREDIT_WORDS)
    if not subject:
        return None
    clauses = [c for t in texts for c in re.split(r";\s*", t)]
    # a name is capitalised inside a clause, not only as its first word ("Write a = 2c", "Libet (1983) found")
    proper = {m.group(0).lower() for c in clauses for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", c)
              if c[:m.start()].rstrip()[-1:] not in ("", ".", "!", "?")}
    said, asked = _toks(sentence), _toks(question)
    names = {_stem(w) for w in re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)}
    extra = (said - subject - _CREDIT_WORDS - names - asked) | (_short(sentence) - _short(question))
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", sentence):
        name = m.group(0).lower()
        if name not in proper or name in _COMMON or re.search(rf"\b{name}\b", question.lower()):
            continue
        if re.search(r"\b(?:not|never)\W+(?:\w+\W+){0,3}$", sentence[:m.start()], re.I):
            continue                     # "the first to prove this was not Euler"
        own = [c.lower() for c in clauses if re.search(rf"\b{name}\b", c.lower())]
        if any(subject <= _toks(c) for c in own):
            continue
        about_own = any(extra & (_toks(c) | _short(c)) for c in own)    # "Euler (1737) showed that e is irrational"
        if subject <= said or not about_own:
            return f"{m.group(0)} credited with it (no fact ties {m.group(0)} to this question)"
    return None


_NAME_VERBS = re.compile(r"\s+(?:argued|argues|said|says|showed|shows|proposed|proposes|claimed|claims|wrote|writes|"
                         r"found|finds|proved|proves|demonstrated|believed|believes|suggested|suggests|called|coined|"
                         r"discovered|developed|introduced|famously|himself|herself)\b|'s\s", re.I)


def _looks_like_name(word: str, text: str, end: int) -> bool:
    """A sentence-initial capitalised word is treated as a name when a reporting verb or another
    capitalised word follows it ("Frankfurt argued", "Benjamin Libet")."""
    if _NAME_VERBS.match(text[end:end + 30]):
        return True
    nxt = re.match(r"\s+([A-Z][a-z]+)", text[end:end + 30])      # case-sensitive: "Harry Frankfurt", "Benjamin Libet"
    return bool(nxt) and nxt.group(1).lower() not in _COMMON


def compile_answer(answer: str, report: Report, facts: list[Fact], mode: str, question: str = "") -> tuple[str, list[str]]:
    """The text shown: sentences carrying an unsupported detail are removed, tags stripped, and a
    provenance line added. A who/when question whose kept answer names no one the notes name gets a
    plain "my notes don't say" line. Returns (text, removed sentences)."""
    bad = {i.sentence for i in report.issues if i.kind != "tag"}
    kept, removed = [], []
    for par in re.split(r"(\n\s*\n)", answer or ""):
        if re.fullmatch(r"\n\s*\n", par or ""):
            kept.append(par)
            continue
        sents = _sentences(par) if par.strip() else []
        keep = [s for s in sents if s not in bad]
        removed += [s for s in sents if s in bad]
        if keep:
            kept.append(" ".join(_TAG.sub("", s).strip() for s in keep))
    text = re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()
    # "…, according to [F3]." loses its tag and would end "according to." (replay 2026-09-25)
    text = re.sub(r",?\s*\b(?:according\s+to|as\s+(?:stated|shown|noted)\s+in|see)\s*(?=[.,;:])", "", text, flags=re.I)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    if mode == GROUNDED and _WHO_Q.match(question or ""):
        wh = "when" if re.match(r"^\s*(?:when|in\s+what\s+year)", question, re.I) else "who"
        fact_names = {w for f in facts for w in re.findall(r"\b[A-Z][a-z]{2,}\b", f.text)} - {w.capitalize() for w in _COMMON}
        named = any(re.search(rf"\b{re.escape(n)}\b", text) for n in fact_names) if wh == "who" else bool(_NUM.search(text))
        # a name the rewrite blanked, or an unsupported one that was cut, is the model reaching for an answer
        # the notes do not hold: say so plainly instead of leaving the reader to guess
        reached = any(i.kind in ("garbled", "name") for i in report.issues)
        # the model may have said so already: "My notes don't say who. My notes do not say who first proved ..."
        said = re.search(r"\bnotes\s+(?:do|does)\s*(?:not|n't)\s+(?:say|name|mention|give|state|record)\b", text, re.I)
        if (not named or reached) and not said:
            text = (f"My notes don't say {wh}. " + text).strip()
    if mode == GROUNDED:
        sources = []
        for f in facts:
            if f.id in report.cited and f.source not in sources:
                sources.append(f.source)
        if not sources:
            sources = list(dict.fromkeys(f.source for f in facts))
        note = f"(from DRock's notes: {'; '.join(sources[:3])}"
        note += f"; {len(removed)} unsupported sentence{'s' if len(removed) != 1 else ''} left out)" if removed else ")"
        text = f"{text}\n{note}" if text else "I couldn't give an answer my notes support."
    elif mode == GENERAL and text and UNKNOWN_LINE not in text:
        text += "\n" + GENERAL_TAG
    return text, removed


def rewrite_worth_it(answer: str, report: Report) -> bool:
    """Is a regeneration worth its ~21 s, or does cutting the one bad sentence do the whole job?

    compile_answer already removes every sentence an issue points at and strips every tag that names
    no fact, so a rewrite only buys something when there is something left to save. Measured over
    the 158 grounded answers in traces.jsonl: 38 carried unsupported details, 24 of them in a single
    sentence. False means skip the call and let the cut stand.
    """
    bad = {i.sentence for i in report.issues if i.kind != "tag"}
    if not bad:
        return False              # tag-only: the tags are stripped, there is nothing to re-say
    if len(bad) != 1:
        return True               # several bad sentences: regenerating can keep the points
    sents = [s for par in re.split(r"\n\s*\n", answer or "") for s in _sentences(par) if s.strip()]
    if len(sents) < 3:
        return True               # cutting one of two sentences leaves no answer
    return next(iter(bad)) == sents[0]   # the first sentence is the answer itself


def feedback(report: Report) -> str:
    """What the regeneration is told."""
    terms = report.terms()
    msg = ("Your answer used details that are not in the SOURCE FACTS: " + ", ".join(repr(t) for t in terms[:12])
           + ". Rewrite the answer without those particular details only: keep every name, number and fact that IS in the "
           "table (use them), and keep the [F#] tags.") if terms else ""
    bad_tags = sorted({i.token for i in report.issues if i.kind == "tag"})
    if bad_tags:
        msg += f" These tags name no fact in the table: {', '.join(bad_tags)}."
    return msg.strip()
