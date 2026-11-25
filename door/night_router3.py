"""Night router, third version: a common word cannot open a night by itself.

Under night_router2, one-word subjects that are also common words ("Information", "States", "President", "Value",
"Russia") open a night for many ordinary requests, and the answers to those requests get worse.

One new rule, general: a subject whose own pattern is found in COMMON_SHARE or more of a background set of ordinary
requests is a common subject. A common subject opens its night only when the question also shares at least two
relation or value words with the fact. Every other rule is night_router2's.
The background set must not contain the questions a test is scored on.

    r = NightRouter3(background=[...ordinary requests...])
    r.add_night(1, facts); r.route(question)
"""
import math

from night_router import dates, words
from night_router2 import PARTIAL_MIN_SHARE, PARTIAL_MIN_WORDS, NightRouter2

COMMON_SHARE = 0.0006          # 3 of 5,000 background requests
COMMON_NEEDS = 2               # fact words a common subject needs in the question


class NightRouter3(NightRouter2):
    def __init__(self, background=None, **kw):
        NightRouter2.__init__(self, **kw)
        self.background = list(background or [])
        self._common = {}          # pattern text -> bool

    def common(self, p):
        k = (p.pattern, p.flags)
        if k not in self._common:
            need = max(1, int(math.ceil(COMMON_SHARE * len(self.background)))) if self.background else None
            hits = 0
            if need:
                for t in self.background:
                    if p.search(t):
                        hits += 1
                        if hits >= need:
                            break
            self._common[k] = bool(need and hits >= need)
        return self._common[k]

    def common_subjects(self):
        return sorted({" ".join(sorted(sw)) for night, p, loose, sw, fw, fd, n in self.facts if self.common(p)})

    def candidates(self, question):
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
            if score < COMMON_NEEDS and self.common(p):
                continue
            key = (score, n, 1 if full else 0)
            if key > best.get(night, (-1, 0, 0)):
                best[night] = key
        return best
