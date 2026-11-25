"""Night router, second version: stricter subject matching than night_router.py.

With many nights loaded, night_router.py has two faults:
- Questions about an early day go to a later night. Most are vague questions whose only match is a common name
  ("Putin", "Trump", "Attacks", "Europe"); scores tie and the newer night wins.
- Ordinary requests, such as maths questions, open a night through junk subjects: "She", "WHO" matching "who",
  "students", "value", "today".

Changes, all general rules:
1. A subject made only of pronouns, stop words or time words ("She", "Who", "Today") is not routable.
2. A one-word subject, or an all-capitals one (WHO, EIB), matches only in the case it was written. It may also match
   in any case when the question shares a relation or value word with the fact.
3. A tie in score goes to the longer subject ("Houthi attacks" beats "Attacks"). Then a full subject match beats a
   partial one, then the newer night wins, as before.
4. A question that rewords the subject still routes when it names at least 2 of the subject's content words and at
   least 60% of them ("the woman who tested positive for Ebola in Abuja").

    r = NightRouter2()
    r.add_night(1, [{"subject": "Alberta", "relation": "referendum date", "value": "19 October", "date": "2026-10-12"}])
    r.route("When will Alberta hold its referendum?")   # -> 1, or None when no night's facts match the question
"""
import re

from night_router import DATE_BONUS, STOP, dates, words

_WORD = re.compile(r"[^\W_]+", re.U)
NOT_SUBJECT = set("i me my he him his she her hers they them their we us our you your it its who whom whose this that "
                  "these those here there today yesterday tomorrow now".split())
PARTIAL_MIN_WORDS = 2
PARTIAL_MIN_SHARE = 0.6


def _pattern(w, case_sensitive):
    return re.compile(r"(?<![^\W_])" + r"[\W_]+".join(re.escape(x) for x in w) + r"(?![^\W_])",
                      re.U if case_sensitive else re.I | re.U)


class NightRouter2(object):
    def __init__(self, date_bonus=DATE_BONUS):
        self.date_bonus = date_bonus
        # (night, pattern, any-case pattern or None, subject words, fact words, fact dates, subject length)
        self.facts = []
        self.nights = set()

    def add_night(self, night, facts):
        """night: a number that grows with time. facts: [{subject, relation, value, date}] learnt that night."""
        self.nights.add(night)
        for f in facts:
            s = f.get("subject", "")
            w = _WORD.findall(s or "")
            if not w or all(x.lower() in NOT_SUBJECT or x.lower() in STOP for x in w):
                continue
            exact = len(w) == 1 or all(x.isupper() for x in w)
            self.facts.append((night, _pattern(w, exact), _pattern(w, False) if len(w) == 1 else None, words(s),
                               words(f.get("relation", "") + " " + f.get("value", "")), dates(f.get("date", "")),
                               len(w)))

    def candidates(self, question):
        """{night: best key} for the nights with a fact the question names; key = (score, subject length, full)"""
        q = question or ""
        qw, qd = words(q), dates(q)
        best = {}
        for night, p, loose, sw, fw, fd, n in self.facts:
            score = len(qw & (fw - sw)) + (self.date_bonus if qd & fd else 0)
            full = bool(p.search(q)) or (loose is not None and score >= 1 and bool(loose.search(q)))
            if not full:
                hit = len(qw & sw)
                if len(sw) < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_SHARE * len(sw):
                    continue
            key = (score, n, 1 if full else 0)
            if key > best.get(night, (-1, 0, 0)):
                best[night] = key
        return best

    def route(self, question, upto=None):
        """the night whose knowledge answers the question (nights after `upto` ignored), or None: the model as it is"""
        c = {n: k for n, k in self.candidates(question).items() if upto is None or n <= upto}
        if not c:
            return None
        return max(c, key=lambda n: (c[n], n))
