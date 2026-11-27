"""meaning_router.py + front_door.py: a question that names no learnt subject reaches the update whose learnt fact is
closest in meaning (26 Nov 2026). Laptop only: a word-hashing embedder stands in for the multilingual sentence model
(it tests the plumbing, not the meaning), and a fake engine stands in for the plain model and the update."""
import json
import os
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import front_door as F
import meaning_router as M

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print("  [%s] %-92s %s" % ("PASS" if cond else "FAIL", name, str(detail)[:80]))


class Engine(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model, text = body.get("model"), body["messages"][-1]["content"]
        Engine.seen.append((model, text))
        if model == "fintan-4b" and text.startswith("Facts that were learnt:"):
            facts = text.split("Question:", 1)[0]
            out = "yes" if "Benghazi" in facts and "Benghazi" in text.split("Answer given:", 1)[-1] else "no"
        elif model == "fintan-4b-day1":
            out = "A Russian warship is docked in Benghazi, in Libya."
        else:
            out = "plain answer"
        raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def chat(url, text):
    body = {"model": "fintan-4b", "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def setting(url, change):
    req = urllib.request.Request(url + "/fintan/settings", data=json.dumps(change).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    print("ROUTING BY MEANING")
    facts = [{"subject": "Warship", "relation": "docked", "value": "in the Libyan port city of Benghazi", "date": "2026-11-23"},
             {"subject": "COP31", "relation": "criticism", "value": "from prominent climate advocates", "date": "2026-11-23"}]
    mr = M.MeaningRouter(embedder=M.HashEmbedder(), threshold=0.3)
    mr.add_night(1, facts)
    q = "What is happening with the naval vessel in the Libyan port?"
    check("a question naming no learnt subject: its closest fact is found", mr.scores(q)[0][2].startswith("Warship"), mr.scores(q)[:1])
    check("  and it goes to that update when close enough", mr.route(q) == 1)
    check("an unrelated request goes nowhere", mr.route("Write a haiku about autumn leaves.") is None,
          mr.scores("Write a haiku about autumn leaves.")[:1])
    mr.threshold = 0.99
    check("the threshold holds (0.99: not close enough)", mr.route(q) is None)
    check("nearest gives the judge 'subject | relation: value' lines",
          mr.nearest(q, 1) == ["Warship | docked: in the Libyan port city of Benghazi"], mr.nearest(q, 1))
    off = M.MeaningRouter(model_path=os.path.join(tempfile.gettempdir(), "no-such-meaning-model"))
    check("without the model files it is off and says why", not off.ready and "not found" in off.error, off.error)
    check("  and routes nothing", off.route(q) is None)
    mr.drop_missing(set())
    check("a removed update is dropped", mr.route(q) is None and not mr.nights)

    # (27 Nov) practice questions as keys, and the margin over ordinary requests
    m2 = M.MeaningRouter(embedder=M.HashEmbedder(), threshold=0.05)
    m2.add_night(1, facts, [{"subject": "Warship", "relation": "docked", "value": "in the Libyan port city of Benghazi",
                             "questions": ["Which ship came into harbour at Benghazi this week?"]}])
    m2.set_bank(["Write a haiku about autumn leaves.", "Which harbour city is the oldest in the world?"])
    q2 = "Which ship came into harbour at Benghazi?"
    sc = m2.scores(q2)
    check("a question close to a practice question reaches the update (margin over ordinary requests)",
          m2.route(q2) == 1 and sc[0][2].startswith("Warship"), sc[:1])
    check("  the judge is given the fact behind that practice question",
          m2.nearest(q2, 1) == ["Warship | docked: in the Libyan port city of Benghazi"], m2.nearest(q2, 1))
    check("an ordinary request close to the bank does not", m2.route("Which harbour city is the oldest?") is None,
          m2.scores("Which harbour city is the oldest?")[:1])

    eng = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=eng.serve_forever, daemon=True).start()
    engine = "http://127.0.0.1:%d" % eng.server_address[1]
    d = tempfile.mkdtemp()
    json.dump(facts, open(os.path.join(d, "f1.json"), "w"))
    man = os.path.join(d, "manifest.json")
    json.dump({"base": "fintan-4b", "nights": [{"night": 1, "model": "fintan-4b-day1", "facts": "f1.json"}]}, open(man, "w"))
    door, srv = F.serve(man, engine, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    check("off by default: the question names nothing learnt, so the plain model answers",
          door.choose({"messages": [{"role": "user", "content": q}]})[1] is None)
    setting(url, {"meaning_route": True, "meaning_threshold": 0.3, "check_answers": True})
    door.meaning = M.MeaningRouter(embedder=M.HashEmbedder())     # the shipped model stands in for: word hashing
    door._meaning_sync()
    check("on: the same question goes to the update by meaning", door.choose({"messages": [{"role": "user", "content": q}]})[1] == 1,
          door.stats.get("by_meaning"))
    check("  but a question about a day after everything learnt does not (the dates window, 27 Nov)",
          door.choose({"messages": [{"role": "user", "content": "What is happening with the naval vessel in the Libyan port on November 30?"}]})[1] is None)
    check("  an ordinary request still goes to the plain model",
          door.choose({"messages": [{"role": "user", "content": "Write a haiku about autumn leaves."}]})[1] is None)
    Engine.seen.clear()
    r = chat(url, q)
    said = r["choices"][0]["message"]["content"]
    check("served: the update answers from its weights, nothing added to the question",
          "Benghazi" in said and any(m == "fintan-4b-day1" and t == q for m, t in Engine.seen), said)
    judge = [t for m, t in Engine.seen if t.startswith("Facts that were learnt:")]
    check("  the answer judge saw the fact found by meaning (the question names no subject)",
          judge and "Warship | docked" in judge[0], judge[:1])
    st = json.loads(urllib.request.urlopen(url + "/fintan/status", timeout=10).read())
    check("status says routing by meaning is on", st.get("meaning_route") == "on", st.get("meaning_route"))
    setting(url, {"meaning_route": False})
    check("switched off again: plain model", door.choose({"messages": [{"role": "user", "content": q}]})[1] is None)
    srv.shutdown()
    eng.shutdown()
    print("ROUTING BY MEANING: %d of %d checks pass" % (sum(R), len(R)))


if __name__ == "__main__":
    main()
