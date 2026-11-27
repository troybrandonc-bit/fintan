"""Routing by meaning (26 Nov 2026).

Troy: "how do we fix general untold use" -> "it only has to be in cases where there is no direct route from weights,
because if not fintan is like an api". The 60-question test of that day: the 4B with its news update answered 31 of 60
questions that need a learnt fact without naming its subject (the plain 4B 13), but the word router sent only 10 of the
60 to the update ("the large naval vessel anchored in the Libyan port" shares almost no words with "Warship | docked |
in the Libyan port city of Benghazi"). So the knowledge in the weights works once the question reaches it; this
module is how the question reaches it when no word of a learnt subject is in it.

Every learnt fact is turned into a short statement and embedded by a small multilingual sentence model on the CPU
(intfloat/multilingual-e5-small, 118M, shipped with the release, never downloaded at run time: the one network door
is home/net.py). A question the word router could not place is embedded too; the update holding the closest fact gets
it when that fact is at least `threshold` close. Nothing is added to the prompt and nothing is looked up: the question
goes to the update as it is, and the update answers from its weights. The threshold is set once on ordinary requests
(the same Alpaca sample as the word router: at most 2 of 2000 sent to an update) and then frozen.

Off by default (front door setting "meaning_route"). Without the model files it stays off and says why.
"""
import hashlib
import math
import os
import re
import threading

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
THRESHOLD = 0.88          # placeholder until meaning_calib.py sets it on ordinary requests (e5 cosines sit high)
MAX_FACTS = 20000


def statement(f):
    """a learnt fact as the sentence it is embedded as"""
    s, r, v = (str(f.get(k) or "").strip() for k in ("subject", "relation", "value"))
    r = r.replace("_", " ")
    return re.sub(r"\s+", " ", "%s %s %s" % (s, r, v)).strip()


class HashEmbedder(object):
    """for the laptop tests: words hashed into 256 buckets (no model, no download); never used in serving"""

    def encode(self, texts, kind):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) > 2:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class E5Embedder(object):
    """multilingual-e5-small with mean pooling, on the CPU, two threads (the person's machine is serving the LLM)"""

    def __init__(self, path, threads=2):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.m = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def encode(self, texts, kind):
        torch, pre = self.torch, ("query: " if kind == "query" else "passage: ")
        out = []
        for i in range(0, len(texts), 32):
            enc = self.tok([pre + t for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128,
                           return_tensors="pt")
            with torch.no_grad():
                h = self.m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            e = torch.nn.functional.normalize(e, dim=-1)
            out += e.tolist()
        return out


class MeaningRouter(object):
    def __init__(self, embedder=None, model_path=None, threshold=THRESHOLD):
        self.threshold, self.lock = threshold, threading.Lock()
        self.nights, self.error, self.bank = {}, "", None
        self.emb = embedder
        if self.emb is None:
            path = model_path or os.environ.get("FINTAN_MEANING_MODEL") or ""
            if not path or not os.path.isdir(path):
                self.error = "meaning model not found (%s); routing by meaning is off" % (path or "no path")
            else:
                try:
                    self.emb = E5Embedder(path)
                except Exception as e:  # noqa: BLE001
                    self.error = "meaning model could not load: %s" % str(e)[:200]

    @property
    def ready(self):
        return self.emb is not None

    def set_bank(self, texts, cache=None):
        """(27 Nov) ordinary requests (a shipped sample of real users' first messages): a question is routed by how much
        closer it is to a learnt fact than to the closest ordinary request, not by a raw closeness every pair reaches
        (raw: 19 of 60 unasked questions at 2 of 2000 ordinary ones sent; margin with practice questions: 55 of 60)"""
        if not self.ready or not texts:
            return
        vecs = None
        if cache and os.path.exists(cache):
            try:
                import json
                got = json.load(open(cache, encoding="utf-8"))
                if got.get("n") == len(texts):
                    vecs = got["vecs"]
            except Exception:  # noqa: BLE001
                vecs = None
        if vecs is None:
            vecs = self.emb.encode(list(texts), "query")
            if cache:
                try:
                    import json
                    json.dump({"n": len(texts), "vecs": [list(map(float, v)) for v in vecs]}, open(cache, "w"))
                except Exception:  # noqa: BLE001
                    pass
        self.bank = _mat(vecs)

    def add_night(self, night, facts, questions=None):
        """embed one update's facts once (at reload), and the practice questions written for them when the update has
        them (questions.json); a night already embedded is kept as it is"""
        if not self.ready or night in self.nights:
            return
        fl = [f for f in (facts or [])[:MAX_FACTS] if statement(f)]
        if not fl:
            return
        lab = lambda f: "%s | %s: %s" % (f.get("subject"), f.get("relation"), f.get("value"))  # noqa: E731
        vecs = list(self.emb.encode([statement(f) for f in fl], "passage"))
        labels = [lab(f) for f in fl]
        qt, ql = [], []
        for e in (questions or [])[:MAX_FACTS]:
            for q in (e.get("questions") or [])[:8]:
                if q and q.strip():
                    qt.append(q.strip())
                    ql.append(lab(e))
        if qt:
            vecs += list(self.emb.encode(qt, "query"))
            labels += ql
        with self.lock:
            self.nights[night] = (labels, _mat(vecs))

    def drop_missing(self, keep):
        with self.lock:
            for n in [n for n in self.nights if n not in keep]:
                del self.nights[n]

    def _q(self, question):
        return _mat(self.emb.encode([question.strip()[:1000]], "query"))[0]

    def nearest(self, question, k=3):
        """the k learnt facts closest in meaning, as "subject | relation: value" lines (for the answer judge)"""
        if not self.ready or not (question or "").strip():
            return []
        q = self._q(question)
        with self.lock:
            items = list(self.nights.items())
        sc = []
        for _n, (labs, vecs) in items:
            for lab, s_ in zip(labs, _dots(q, vecs)):
                sc.append((s_, lab))
        out = []
        for _s, lab in sorted(sc, reverse=True):
            if lab not in out:
                out.append(lab)
            if len(out) >= k:
                break
        return out

    def scores(self, question):
        """-> [(score, night, closest fact as "subject | relation: value")], best first; with a bank of ordinary requests
        the score is the margin over the closest of them"""
        if not self.ready or not (question or "").strip():
            return []
        q = self._q(question)
        base = max(_dots(q, self.bank)) if self.bank is not None else 0.0
        out = []
        with self.lock:
            items = list(self.nights.items())
        for night, (texts, vecs) in items:
            d = _dots(q, vecs)
            if d:
                bi = max(range(len(d)), key=d.__getitem__)
                out.append((d[bi] - base, night, texts[bi]))
        return sorted(out, reverse=True)

    def route(self, question):
        """the night whose closest fact (or practice question) is at least `threshold` close (or that much closer than
        any ordinary request, with a bank), else None"""
        sc = self.scores(question)
        if sc and sc[0][0] >= self.threshold:
            return sc[0][1]
        return None


def _mat(vecs):
    """vectors as one tensor when torch is there (fast), else lists"""
    try:
        import torch
        return torch.tensor([list(map(float, v)) for v in vecs], dtype=torch.float32)
    except Exception:  # noqa: BLE001
        return [list(map(float, v)) for v in vecs]


def _dots(q, vecs):
    if vecs is None:
        return []
    try:
        import torch
        if isinstance(vecs, torch.Tensor):
            return (vecs @ q).tolist()
    except Exception:  # noqa: BLE001
        pass
    return [sum(a * b for a, b in zip(q, v)) for v in vecs]
