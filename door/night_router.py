"""Which night's learnt knowledge answers a question.

Each night learns that day's facts into its own adapter. A story runs over several days, so several nights can name the
same subject, and the newest night that names it does not always know the asked fact. This router reads only the
server's own fact table: among the nights whose facts name a subject in the question, it picks the night with the
best-matching fact about that subject. The match counts the fact's relation and value words the question shares, plus
DATE_BONUS when the question names the fact's date. A tie goes to the newer night, so a changed fact still answers
with its newest value. It is a word match per request, with no tokens or GPU.

    r = NightRouter()
    r.add_night(1, [{"subject": "Alberta", "relation": "referendum date", "value": "19 October", "date": "2026-10-12"}])
    r.route("When will Alberta hold its referendum?")   # -> 1, or None when no night names a subject in it
"""
import re

DATE_BONUS = 2
STOP = set("the a an of to in on for and or by with at from as is are was were be been what which who whom whose when "
           "where why how does do did has have had this that these those it its their there about into than after "
           "before over under per via".split())
MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november "
                                         "december".split())}
_WORD = re.compile(r"[^\W_]+", re.U)
_DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})\b|\b(\d{4})-(\d{2})-(\d{2})\b", re.I)


def words(text):
    """content words: lower case, not a stop word, longer than 2 characters, not a year or month name"""
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in STOP and len(w) > 2
            and not re.fullmatch(r"(19|20)\d\d", w) and w.lower() not in MONTHS}


def dates(text):
    """the (month, day) pairs a text names: 'November 3', '2026-11-03'"""
    out = set()
    for m in _DATE.finditer(text or ""):
        if m.group(1):
            out.add((MONTHS[m.group(1).lower()], int(m.group(2))))
        else:
            out.add((int(m.group(4)), int(m.group(5))))
    return out


def subject_pattern(subject):
    """the subject's words in order, any case, any punctuation between them, whole words only; None for no words"""
    w = _WORD.findall(subject or "")
    if not w:
        return None
    return re.compile(r"(?<![^\W_])" + r"[\W_]+".join(re.escape(x) for x in w) + r"(?![^\W_])", re.I | re.U)


class NightRouter(object):
    def __init__(self, date_bonus=DATE_BONUS):
        self.date_bonus = date_bonus
        self.facts = []          # (night, subject pattern, subject words, fact words, fact dates)
        self.nights = set()

    def add_night(self, night, facts):
        """night: a number that grows with time. facts: [{subject, relation, value, date}] learnt that night."""
        self.nights.add(night)
        for f in facts:
            p = subject_pattern(f.get("subject", ""))
            if p is None:
                continue
            self.facts.append((night, p, words(f["subject"]), words(f.get("relation", "") + " " + f.get("value", "")),
                               dates(f.get("date", ""))))

    def candidates(self, question):
        """{night: best score} for the nights with a fact whose subject the question names"""
        qw, qd = words(question), dates(question)
        best = {}
        for night, p, sw, fw, fd in self.facts:
            if p.search(question or ""):
                s = len(qw & (fw - sw)) + (self.date_bonus if qd & fd else 0)
                best[night] = max(best.get(night, -1), s)
        return best

    def route(self, question, upto=None):
        """the night whose knowledge answers the question (nights after `upto` ignored), or None: the model as it is"""
        c = {n: s for n, s in self.candidates(question).items() if upto is None or n <= upto}
        if not c:
            return None
        return max(c, key=lambda n: (c[n], n))

    def newest(self, question, upto=None):
        """the simpler rule, kept for comparison: the newest night that names a subject in the question"""
        c = [n for n in self.candidates(question) if upto is None or n <= upto]
        return max(c) if c else None
