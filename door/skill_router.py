"""Which learnt skill, if any, a request is for. Plain code, no model, no internet.

A skill update (learn/fintan_learn.py, kind "skills") lists the words that marked the requests it practised
("refund", "order", ...). A request goes to that update when it carries enough of them: two, or one when the skill
has fewer than four. When several skills match, the one with most shared words wins, then the newest.
Facts come first: the front door asks this only when the fact router sent the request nowhere.
"""
import re

STOP = set(("a an the and or but if of to in on at for with from by about as is are was were be been it its this that "
            "these those i you he she we they me my your our their can could would should will do does did have has "
            "had not no yes please what which who how when where why there here some any all more most than then so "
            "just also very much many get got make want need like").split())


def words(t):
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", (t or "").lower()) if w not in STOP}


class SkillRouter(object):
    def __init__(self):
        self.skills = []                                     # (night, set of trigger words)

    def add_night(self, night, trigger_words):
        tw = {w.lower() for w in trigger_words or [] if w}
        if tw:
            self.skills.append((int(night), tw))

    def route(self, text):
        q = words(text)
        best = None
        for night, tw in self.skills:
            hit = len(q & tw)
            if hit >= (2 if len(tw) >= 4 else 1):
                key = (hit, night)
                if best is None or key > best:
                    best = key
        return best[1] if best else None
