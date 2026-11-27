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
import re

from night_router import _WORD, dates, subject_pattern, words
from night_router2 import PARTIAL_MIN_SHARE, PARTIAL_MIN_WORDS
from night_router3 import COMMON_NEEDS, NightRouter3

RARE_SHARE = 0.002             # a word in 0.2% or more of ordinary requests is an everyday word
HEAD_MAX_SUBJECTS = 2         # a last word shared by more learnt subjects names none of them alone
TOPIC_MIN_WORDS = 3           # rare words a question must share with one learnt fact to reach it unnamed


def _day(md):
    """a (month, day) pair as a day of the year (the question gives no year)"""
    import datetime
    try:
        return datetime.date(2026, md[0], md[1]).toordinal()
    except ValueError:
        return 0


class NightRouter4(NightRouter3):
    def __init__(self, background=None, rare_share=RARE_SHARE, head_rule=True, topic_rule=False, date_window=True, **kw):
        self.head_rule, self.topic_rule, self.date_window = head_rule, topic_rule, date_window
        self._head_of, self._heads = {}, collections.Counter()
        self._newest = {}
        NightRouter3.__init__(self, background=background, **kw)
        df = collections.Counter()
        for t in self.background:
            df.update(words(t))
        self._everyday = {w for w, c in df.items() if c >= rare_share * len(self.background)} if self.background else set()

        self._head_of, self._heads = {}, collections.Counter()

    def add_night(self, night, facts):
        """(26 Nov 2026, Fintan-4B on real news: "Where did the warship dock?" never reached the update that learnt
        "Russian warship"; a partial match needed two of the subject's words) each subject of two or more words also
        keeps its last word, which can name it alone (see candidates)"""
        for f in facts:                                       # (27 Nov) the newest day this update learnt about
            m_ = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(f.get("date") or ""))
            if m_:
                md = (int(m_.group(2)), int(m_.group(3)))
                if md > self._newest.get(night, (0, 0)):
                    self._newest[night] = md
        NightRouter3.add_night(self, night, facts)
        for f in facts:
            raw = _WORD.findall(f.get("subject") or "")
            p = subject_pattern(f.get("subject") or "")
            # only a common noun written in lower case ("Russian warship"), never a name ("Alex Jones", "East Berlin")
            if p is not None and len(raw) >= 2 and raw[-1].islower() and p.pattern not in self._head_of:
                self._head_of[p.pattern] = raw[-1]
                self._heads[raw[-1]] += 1

    def out_of_window(self, night, question):
        """(27 Nov) the question asks only about days after everything this update learnt (the same rule candidates()
        applies), for routing by meaning too"""
        from night_router import dates as _dates
        qd = _dates(question or "")
        return bool(self.date_window and qd and night in self._newest and
                    min(_day(d) for d in qd) > _day(self._newest[night]) + 1)

    def head_ok(self, p, qw, question=None):
        """the subject's last word alone names it: in the question, 5 letters or more, not an everyday word, and the last
        word of at most HEAD_MAX_SUBJECTS learnt subjects"""
        h = self._head_of.get(p.pattern)
        # and the question points at one known thing: "the warship", not "literary devices"
        return (bool(h) and len(h) >= 5 and h in qw and h not in self._everyday and self._heads[h] <= HEAD_MAX_SUBJECTS
                and re.search(r"(?i)\bthe\s+%s\b" % re.escape(h), question or "") is not None)

    def everyday(self, w):
        return w in self._everyday

    def candidates(self, question):
        q = question or ""
        qw, qd = words(q), dates(q)
        qr = {w for w in qw if w not in self._everyday}
        best = {}
        for night, p, loose, sw, fw, fd, n in self.facts:
            # (27 Nov 2026; an outside test of Fintan-9B: 3 of 10 questions about November days it never learnt were
            # sent to the update, which answered from older facts) a question only about days after everything an
            # update learnt cannot be answered by it: the plain model says what it knows
            if self.date_window and qd and night in self._newest and \
                    min(_day(d) for d in qd) > _day(self._newest[night]) + 1:
                continue
            bonus = self.date_bonus if qd & fd else 0
            score = len(qw & (fw - sw)) + bonus
            rscore = len(qr & (fw - sw)) + bonus
            full = bool(p.search(q)) or (loose is not None and score >= 1 and bool(loose.search(q)))
            if not full:
                hit = len(qw & sw)
                if (len(sw) < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_WORDS or hit < PARTIAL_MIN_SHARE * len(sw)
                        or len(qr & sw) < PARTIAL_MIN_WORDS or rscore < 1):
                    if not (self.head_rule and self.head_ok(p, qw, q)):
                        # (26 Nov 2026, Troy: "how do we fix general untold use") a question can need a learnt fact
                        # without naming its subject ("Is the Infirmary Road bridge open again?" for "Harrowgate
                        # Bridge | reopened | ..."): TOPIC_MIN_WORDS words of the fact that are not everyday words
                        if not (self.topic_rule and len(qr & (fw | sw)) >= TOPIC_MIN_WORDS):
                            continue
            if rscore < COMMON_NEEDS and self.common(p):
                continue
            key = (score, n, 1 if full else 0)
            if key > best.get(night, (-1, 0, 0)):
                best[night] = key
        return best
