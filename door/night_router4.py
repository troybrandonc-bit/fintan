"""Night router, fourth version: everyday words cannot open a night.

Under night_router3, ordinary requests can still reach a news update in two ways. A common subject such as
"Information" opens its night when the question shares any two words with the fact. A subject made of everyday words
and matched in part ("one woman", "last minute measures") opens its night with no fact word in the question at all.

One idea, general: a word that is frequent in ordinary requests carries no evidence that a question is about a fact.
The background set (the same one night_router3 uses) gives each word's share of ordinary requests; a word in
RARE_SHARE or more of them is an everyday word.
  1. A common subject (night_router3's rule) opens its night only when the question shares COMMON_NEEDS fact words
     that are not everyday words.
  2. A subject matched only in part must be matched through PARTIAL_MIN_WORDS words that are not everyday words,
     and the question must share at least one fact word that is not an everyday word (or the fact's date).
Everything else is night_router3's. With no background given it behaves as night_router3.

    r = NightRouter4(background=[...ordinary requests...])
    r.add_night(1, facts); r.route(question)
"""
import collections

from night_router import dates, words
from night_router2 import PARTIAL_MIN_SHARE, PARTIAL_MIN_WORDS
from night_router3 import COMMON_NEEDS, NightRouter3

RARE_SHARE = 0.002             # a word in 0.2% or more of ordinary requests is an everyday word


class NightRouter4(NightRouter3):
    def __init__(self, background=None, rare_share=RARE_SHARE, **kw):
        NightRouter3.__init__(self, background=background, **kw)
        df = collections.Counter()
        for t in self.background:
            df.update(words(t))
        self._everyday = {w for w, c in df.items() if c >= rare_share * len(self.background)} if self.background else set()

    def everyday(self, w):
        return w in self._everyday

    def candidates(self, question):
        q = question or ""
        qw, qd = words(q), dates(q)
        qr = {w for w in qw if w not in self._everyday}
        best = {}
        for night, p, loose, sw, fw, fd, n in self.facts:
            bonus = self.date_bonus if qd & fd else 0
            score = len(qw & (fw - sw)) + bonus
            rscore = len(qr & (fw - sw)) + bonus
            full = bool(p.search(q)) or (loose is not None and score >= 1 and bool(loose.search(q)))
            if not full:
                hit = len(qw & sw)
                if len(sw) < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_SHARE * len(sw):
                    continue
                if len(qr & sw) < PARTIAL_MIN_WORDS or rscore < 1:
                    continue
            if rscore < COMMON_NEEDS and self.common(p):
                continue
            key = (score, n, 1 if full else 0)
            if key > best.get(night, (-1, 0, 0)):
                best[night] = key
        return best
