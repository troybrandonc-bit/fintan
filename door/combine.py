"""Questions that need several learnt facts put together (25 Nov 2026; Troy: "put it in the release").

A question that names a learnt subject and has a comparing or chaining word outside the subject names ("which of the
two", "difference between", "of that ...") is answered in steps: the plain model works out which single facts it
needs and asks for them one at a time; each one-fact question is answered by the update that learnt it (and only when
that update's own list of subjects and relations has it); then the plain model answers the person from those facts.
Facts from different updates (different days) can be combined in one answer. In a development test on 120 questions
that each need two to four learnt facts, this went from under 2% to 95.8%.

    idx = build_index([(night, facts), ...])
    if wanted(question, idx): text, lookups = run(question, idx, model_of, base, post)
Plain Python; `post(body) -> reply dict` sends one request to the engine.
"""
import re

ROUNDS = 7
PLAN = ("You are answering a question about things you do not know yet, but you have a reliable lookup that answers "
        "simple one-fact questions. Never say something does not exist: ask the lookup instead.\n\n"
        "Question: {q}\n\nFacts from the lookup so far:\n{facts}\n\n"
        "If these facts are enough, reply with exactly one line: Answer: <the answer only>\n"
        "Otherwise reply with exactly one line asking for ONE missing fact, naming its subject exactly as written: "
        "Ask: <a simple question about one thing>\n\n"
        "Rules: a question may describe something only through another thing (\"X has a Y. What is the Z of that Y?\"): "
        "first ask what X's Y is, then ask about that Y by its name. If the lookup says it has no record, read what "
        "it says it does know and ask a different question.\n\n"
        "Before that line, write one or two short sentences: what the question needs, and which of those facts you "
        "already have. Answer only when every fact the question needs is in the list.")
FINAL = ("Question: {q}\n\nFacts:\n{facts}\n\nThe answer worked out from these facts: {a}\n\n"
         "Answer the question in one or two plain sentences, using only these facts.")
CUE = re.compile(r"(?i)\b(?:which of the (?:two|three)|which one|compare[ds]?|comparison|differences?|differ|between|"
                 r"larger|smaller|bigger|older|younger|higher|lower|earlier|later|both|each of|in total|combined|"
                 r"of that|of those)\b")
STOP = set("the a an of in is are was what which who how many much does do has have that this its it for to and with "
           "number".split())


def norm(x):
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(x).lower()).split())


def build_index(nights):
    """[(night, facts)] -> {subject: {relation: night}} (names only; values stay in the weights). Later nights win."""
    idx = {}
    for night, facts in nights:
        for f in facts or []:
            s, r = str(f.get("subject") or "").strip(), str(f.get("relation") or "").strip()
            if len(norm(s)) >= 3 and r:
                idx.setdefault(s, {})[r] = night
    return idx


def subjects_in(text, idx):
    t = " " + norm(text) + " "
    found = [s for s in idx if " " + norm(s) + " " in t]
    return [s for s in found if not any(s != o and norm(s) in norm(o) for o in found)]     # longest names only


def wanted(text, idx):
    """does this question need several learnt facts put together?"""
    if not idx or not text:
        return False
    subj = subjects_in(text, idx)
    if not subj:
        return False
    # (25 Nov) news questions often name two learnt things ("Nvidia", "Jensen Huang") and need one fact; what marks a
    # question that needs several is a comparing or chaining word, and not one inside a subject's own name
    rest = " " + norm(text) + " "
    for s_ in sorted(subj, key=len, reverse=True):
        rest = rest.replace(" " + norm(s_) + " ", " ")
    return bool(CUE.search(rest))


def gate(question, idx):
    """-> (night, None) to ask that update, or (None, reply) when no update learnt that fact"""
    subj = subjects_in(question, idx)
    if not subj:
        return None, "No record of anything named in that question."
    s = max(subj, key=len)
    words, sw = norm(question).split(), norm(s).split()
    label = {words[i - 1] for i in range(1, len(words) - len(sw) + 1) if words[i:i + len(sw)] == sw}
    qw = [w for w in words if w not in STOP and w not in sw and w not in label]
    for r, night in idx[s].items():
        rw = [w for w in norm(r).split() if w not in STOP]
        if any(len(w) >= 4 and any(q[:5] == w[:5] for q in qw) for w in rw):
            return night, None
    return None, "No record of that. Known about %s: %s." % (s, ", ".join(idx[s]))


# (26 Nov 2026) a property asked of a learnt subject that only OTHER subjects have: "the adult weight in grams" of a
# city. The update then mixed up facts (it gave the city's population as its weight) on 16 of 40 such questions; the
# plain model is asked instead. Strict on purpose: the property must be written out word for word (two words or
# more) and share no word stem, place word or time word with anything learnt about that subject. Offline check:
# 0 of 60 learnt one-fact questions and 0 of 172 news questions would be sent away, 18 of 40 never-taught ones would.
GROUPS = [set("locat city town place where based situa addre site regio distr headq".split()),
          set("year date when found estab start began opene since".split())]
_STOPW = set("the of a an in at to for and is was with by on as from that this its measured value all them what which "
             "who how many much does do has have".split())


def _stems(text):
    st = {w[:5] for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPW}
    return st | {"#%d" % i for i, g in enumerate(GROUPS) if st & g}


def other_property(question, idx):
    """True when the question asks one learnt subject for a property only other subjects have"""
    subj = subjects_in(question, idx)
    if len(subj) != 1:
        return False
    s = subj[0]
    rest = " " + norm(question).replace(norm(s), " ") + " "
    qs = _stems(rest)
    if any(qs & _stems(r) for r in idx[s]):
        return False
    own = {norm(r) for r in idx[s]}
    for o in idx:
        for r in idx[o]:
            n = norm(r)
            if n not in own and len(n.split()) >= 2 and " " + n + " " in " " + " ".join(rest.split()) + " ":
                return True
    return False


def build_values(nights):
    """[(night, facts)] -> {subject: [values]}: what each subject's learnt facts say, to check an answer against"""
    vals = {}
    for _night, facts in nights:
        for f in facts or []:
            s, v = str(f.get("subject") or "").strip(), str(f.get("value") or "").strip()
            if len(norm(s)) >= 3 and v:
                vals.setdefault(s, []).append(v)
    return vals


def _says(answer, value):
    """loose on purpose (a right answer in other words must pass): a value with a number counts when its first number
    is in the answer ("838 thousand" / "838,000"); otherwise half of its words, spelling variants allowed"""
    nums = re.findall(r"\d+", value)
    if nums:
        return nums[0] in re.findall(r"\d+", answer.replace(",", ""))
    vw = [w for w in norm(value).split() if len(w) > 2]
    aw = norm(answer).split()
    if not vw:
        return norm(value) in norm(answer)
    hit = sum(1 for w in vw if w in aw or (len(w) >= 5 and any(x[:5] == w[:5] for x in aw)))
    return hit >= max(1, 0.5 * len(vw))


def supported(answer, question, idx, vals):
    """(26 Nov 2026, Fintan-4B on real news: 25 of 60 answers from an update were wrong, most of them made up about a
    subject it had learnt other things about) an update's answer counts as supported when it states something learnt
    about a subject the question names; with no learnt subject named it is not checked"""
    subj = subjects_in(question, idx)
    if not subj:
        return True
    return any(_says(answer, v) for s in subj for v in vals.get(s, []))


# (26 Nov 2026, Troy: "the model itself should know ceo is chief executive officer, and it should know translations")
# word matching cannot tell that "CEO" and "chief executive officer", or "la junta" and "the board", say the same thing;
# the plain model judges whether an update's answer states one of the facts learnt about what the question asks
JUDGE = ("Facts that were learnt:\n{facts}\n\nQuestion: {q}\nAnswer given: {a}\n\nDoes the answer state one of the learnt "
         "facts above, in any wording or language, as the answer to this question? Reply with one word: yes or no.")


def build_fact_text(nights):
    """[(night, facts)] -> {subject: ["relation: value", ...]}: the learnt facts in words, for the judge"""
    out = {}
    for _night, facts in nights:
        for f in facts or []:
            s, r, v = (str(f.get(k) or "").strip() for k in ("subject", "relation", "value"))
            if len(norm(s)) >= 3 and r and v:
                out.setdefault(s, []).append("%s: %s" % (r, v))
    return out


def learnt_lines(question, idx, ftext, limit=16, answer="", extra=None):
    """the learnt facts the judge sees: those about the subjects the question names, then those sharing the most words
    with the question and the answer. (26 Nov 2026: with the named subjects only, 8 of the 9 right answers the judge
    blocked had their supporting fact filed under another subject, "Where did the strikes occur?" under "Russian
    warship")"""
    named = ["- %s | %s" % (s, rv) for s in subjects_in(question, idx) for rv in ftext.get(s, [])]
    # (26 Nov) a question routed by meaning names no subject: its closest facts in meaning come next
    named += ["- " + l for l in (extra or []) if ("- " + l) not in named]
    qa = {w for w in norm(question + " " + (answer or "")).split() if len(w) > 3 and w not in _STOPW}
    scored = []
    for s, rvs in ftext.items():
        for rv in rvs:
            line = "- %s | %s" % (s, rv)
            if line in named:
                continue
            n = len(qa & {w for w in norm(s + " " + rv).split() if len(w) > 3})
            if n:
                scored.append((n, line))
    more = [l for _n, l in sorted(scored, key=lambda x: -x[0])]
    return "\n".join((named + more)[:limit])


def judged_supported(answer, question, idx, ftext, base, post, extra=None):
    """the plain model's yes or no (True when the facts state it); None when it could not be asked"""
    facts = learnt_lines(question, idx, ftext, answer=answer, extra=extra)
    if not facts:
        return None
    try:
        said = _text(post({"model": base, "messages": [{"role": "user", "content": JUDGE.format(
            facts=facts, q=question, a=answer[:600])}], "max_tokens": 4, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}))
    except Exception:  # noqa: BLE001
        return None
    w = said.strip().lower()
    return True if w.startswith("yes") else False if w.startswith("no") else None


# (27 Nov 2026) recall, then act: a request with tools that needs a learnt fact. The update answers its question of
# fact from its weights (no tools); the plain model does the tool work with that answer in front of it
RECALL_Q = ("%s\n\nAnswer only the question of fact in this request, in one or two sentences, from what you know. "
            "Do not do anything else.")
RECALLED = ("What you learnt that this request needs (answered just now from your own updates): %s\n"
            "Use it where it matters, and do the rest as asked.")
# (27 Nov) the value check: a full-sentence answer missing the learnt value is asked for once more (front_door.py)
VALUE_AGAIN = ("Answer in one sentence that states exactly what you learnt about this, with the exact names, numbers and "
               "words of the fact. Do not add anything else.")
NO_GUESS = ("Answer only with what you learnt. If you did not learn the answer to this question, reply exactly: "
            "There is no record of that.")
UNKNOWN = ("Question: {q}\n\nWhat was learnt about {s}: {known}.\nNone of it answers the question. In one or two plain "
           "sentences, say that you do not have that information, and mention what you do know about {s} if it helps.")


def unknown_reply(question, idx, base, post, max_tokens=None):
    """the plain model says it does not know, and what is known (the answer still comes from a model)"""
    subj = subjects_in(question, idx)
    if not subj:
        return None
    s = max(subj, key=len)
    return _text(post({"model": base, "messages": [{"role": "user", "content": UNKNOWN.format(
        q=question, s=s, known=", ".join(list(idx[s])[:8]))}], "max_tokens": max_tokens or 120, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}})) or None


def _text(reply):
    return (((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def run(question, idx, model_of, base, post, max_tokens=None):
    """-> (answer text, [(asked, got)], the short answer worked out) or None when no answer was reached (the caller
    answers the usual way)"""
    facts, lookups = [], []
    hint = ""
    for rnd in range(ROUNDS):
        plan = _text(post({"model": base, "messages": [{"role": "user", "content": PLAN.format(
            q=question, facts="\n".join(facts) or "(none yet)") + hint}], "max_tokens": 160, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}))
        lines = [x.strip() for x in plan.split("</think>")[-1].splitlines() if x.strip()]
        tagged = [x for x in lines if x.lower().startswith(("ask:", "answer:"))]
        line = tagged[-1] if tagged else ""
        if line and "<" in line and ">" in line:
            # (26 Nov 2026, Fintan-4B: the small plain model wrote "Answer: <the answer only>", the instruction's own
            # placeholder, on most two-step questions) asked again once with a reminder
            if not hint:
                hint = ("\n\n(Write a real question or the real answer after 'Ask:' or 'Answer:', not the words in "
                        "angle brackets.)")
                continue
            return None
        if line.lower().startswith("answer:"):
            if not facts:
                return None                          # nothing was looked up: not a question about learnt facts
            ans = line[7:].strip()
            final = _text(post({"model": base, "messages": [{"role": "user", "content": FINAL.format(
                q=question, facts="\n".join(facts), a=ans)}], "max_tokens": max_tokens or 160, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}))
            return (final or ans), lookups, ans
        if not line.lower().startswith("ask:") or rnd == ROUNDS - 1:
            return None
        ask = line[4:].strip()
        night, reply = gate(ask, idx)
        if night is not None and night in model_of:
            got = _text(post({"model": model_of[night], "messages": [{"role": "user", "content": ask}], "max_tokens": 64,
                              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))
            got = got.splitlines()[0].strip() if got else ""
        else:
            got = reply or "No record of that."
        facts.append("- %s %s" % (ask, got))
        lookups.append((ask, got))
    return None
