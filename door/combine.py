"""Questions that need several learnt facts put together.

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


def _text(reply):
    return (((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def run(question, idx, model_of, base, post, max_tokens=None):
    """-> (answer text, [(asked, got)], the short answer worked out) or None when no answer was reached (the caller
    answers the usual way)"""
    facts, lookups = [], []
    for rnd in range(ROUNDS):
        plan = _text(post({"model": base, "messages": [{"role": "user", "content": PLAN.format(
            q=question, facts="\n".join(facts) or "(none yet)")}], "max_tokens": 160, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}))
        lines = [x.strip() for x in plan.split("</think>")[-1].splitlines() if x.strip()]
        tagged = [x for x in lines if x.lower().startswith(("ask:", "answer:"))]
        line = tagged[-1] if tagged else ""
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
