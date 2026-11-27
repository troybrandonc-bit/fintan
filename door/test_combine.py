"""front_door.py + combine.py: questions that need several learnt facts are answered in steps (25 Nov 2026). Laptop
only: a fake engine stands in for the plain model (it plans) and for two updates learnt on different days (each
answers its own one-fact questions)."""
import json
import os
import sys
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import combine as C
import front_door as F

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
        Engine.seen.append((model, text[:60]))
        if model == "fintan-9b" and text.startswith("Facts that were learnt:"):
            given = text.split("Answer given:", 1)[-1].split("\n")[0]
            out = "yes" if ("Pehan" in given or "Drudrorx" in given) else "no"      # the plain model as judge
        elif model == "fintan-9b" and text.startswith("You are answering") and "Placebo" in text and "angle brackets" not in text:
            out = "I know it.\nAnswer: <the answer only>"         # a small planner copying the placeholder
        elif model == "fintan-9b" and text.startswith("You are answering") and "colour" not in text:
            n = text.count("\n- ")
            out = ["The question needs the genus, then its family.\nAsk: What is the genus of the species Badrirmeth?",
                   "I have the genus.\nAsk: What is the family of the genus Pehan?",
                   "I have both facts.\nAnswer: Drudrorx"][min(n, 2)]
        elif model == "fintan-9b" and text.startswith("Question:") and "None of it answers" not in text:
            out = "The family of that genus is Drudrorx."
        elif model == "fintan-9b-day1" and "colour" in text:
            told = any("Answer only with what you learnt" in (m_.get("content") or "") for m_ in body["messages"])
            out = "There is no record of that." if told else "The colour of Badrirmeth is bright purple."
        elif model == "fintan-9b" and text.startswith("You are answering") and "colour" in text:
            out = "The facts do not say."
        elif model == "fintan-9b" and "None of it answers" in text:
            out = "I don't have that information. I know the genus of Badrirmeth."
        elif model == "fintan-9b-day1" and "classification" in text:
            out = "There is no record of that."
        elif model == "fintan-9b-day1":
            out = "The genus of Badrirmeth is Pehan."
        elif model == "fintan-9b-day2":
            out = "Pehan belongs to the family Drudrorx."
        else:
            out = "plain answer"
        raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def chat(url, text, extra=None):
    body = dict({"model": "fintan-9b", "messages": [{"role": "user", "content": text}]}, **(extra or {}))
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    print("COMBINING LEARNT FACTS")
    eng = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=eng.serve_forever, daemon=True).start()
    engine = "http://127.0.0.1:%d" % eng.server_address[1]
    d = tempfile.mkdtemp()
    f1 = [{"subject": "Badrirmeth", "relation": "genus", "value": "Pehan", "date": "2026-11-20"}]
    f2 = [{"subject": "Pehan", "relation": "family", "value": "Drudrorx", "date": "2026-11-21"}]
    json.dump(f1, open(os.path.join(d, "f1.json"), "w"))
    json.dump(f2, open(os.path.join(d, "f2.json"), "w"))
    man = os.path.join(d, "manifest.json")
    json.dump({"base": "fintan-9b", "nights": [{"night": 1, "model": "fintan-9b-day1", "facts": "f1.json"},
                                               {"night": 2, "model": "fintan-9b-day2", "facts": "f2.json"}]}, open(man, "w"))
    door, srv = F.serve(man, engine, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    # (27 Nov) the answer judge is off by default since the Fintan-9B check; these checks are about it, so it is on
    urllib.request.urlopen(urllib.request.Request(url + "/fintan/settings", data=json.dumps({"check_answers": True}).encode(),
                                                  method="POST", headers={"Content-Type": "application/json"}), timeout=10).read()
    check("the index holds what each day's update learnt (names only)",
          door.fact_index == {"Badrirmeth": {"genus": 1}, "Pehan": {"family": 2}}, door.fact_index)
    q = "The species Badrirmeth has a genus. What is the family of that genus?"
    check("a two-step question about learnt things is recognised", C.wanted(q, door.fact_index))
    check("a one-fact question is not (it goes to its update directly, as before)",
          not C.wanted("What is the genus of the species Badrirmeth?", door.fact_index))
    check("an ordinary question is not", not C.wanted("Which is larger, a cat or a dog?", door.fact_index))
    check("two learnt subjects with a comparing word are", C.wanted("Which is older, Badrirmeth or Pehan?", door.fact_index))
    check("  two learnt subjects alone are not (news questions name several things and need one fact)",
          not C.wanted("What did Badrirmeth say about Pehan?", door.fact_index))
    check("a property only other subjects have is spotted (family of a species)",
          C.other_property("What is the family name of Badrirmeth?", {"Badrirmeth": {"genus": 1}, "Pehan": {"family name": 2}}))
    check("  a property of its own, or free wording, is not",
          not C.other_property("What is the genus of Badrirmeth?", {"Badrirmeth": {"genus": 1}, "Pehan": {"family name": 2}})
          and not C.other_property("What happened to Badrirmeth last week?", {"Badrirmeth": {"genus": 1}, "Pehan": {"family name": 2}}))
    check("the gate: a learnt subject and relation -> its update", C.gate("What is the genus of Badrirmeth?", door.fact_index) == (1, None))
    n, why = C.gate("What is the weight of Badrirmeth?", door.fact_index)
    check("  a fact never learnt -> no record, and what is known about it", n is None and "Known about Badrirmeth: genus" in why, why)
    Engine.seen.clear()
    r = chat(url, q)
    lk = r.get("fintan_lookups") or []
    check("asked through the door: answered from facts of two different days' updates",
          r["choices"][0]["message"]["content"] == "The family of that genus is Drudrorx." and len(lk) == 2
          and "Pehan" in lk[0]["got"] and "Drudrorx" in lk[1]["got"] and r.get("fintan_answer") == "Drudrorx", lk)
    check("  each fact was asked of the update that learnt it, the planning and the answer by the plain model",
          [m for m, _ in Engine.seen] == ["fintan-9b", "fintan-9b-day1", "fintan-9b", "fintan-9b-day2", "fintan-9b", "fintan-9b"],
          [m for m, _ in Engine.seen])
    Engine.seen.clear()
    r = chat(url, "The species Badrirmeth has a genus. What is the family of that genus? (Placebo)")
    plans = [t for m, t in Engine.seen if m == "fintan-9b" and t.startswith("You are answering")]
    check("a planner that writes the placeholder is asked again once, with a reminder",
          len(plans) >= 2 and r.get("fintan_answer") == "Drudrorx", [m for m, _ in Engine.seen])
    Engine.seen.clear()
    r = chat(url, "What is 17 times 23?")
    check("an ordinary question is passed on exactly as before", r["choices"][0]["message"]["content"] == "plain answer"
          and Engine.seen == [("fintan-9b", "What is 17 times 23?")], Engine.seen)
    Engine.seen.clear()
    r = chat(url, "What is the genus of the species Badrirmeth?")
    check("a one-fact answer from its update is passed on as it is",
          r["choices"][0]["message"]["content"] == "The genus of Badrirmeth is Pehan." and not r.get("fintan_lookups")
          and [m for m, _ in Engine.seen] == ["fintan-9b-day1", "fintan-9b"]
          and Engine.seen[1][1].startswith("Facts that were learnt"), Engine.seen)    # the update, then the judge
    Engine.seen.clear()
    r = chat(url, "What is the classification of the species Badrirmeth?")
    check("an update's 'no record' about a learnt subject is asked again in steps (the facts still come from the updates)",
          r.get("id") == "fintan-rescued" and len(r.get("fintan_lookups") or []) == 2
          and [m for m, _ in Engine.seen][:2] == ["fintan-9b-day1", "fintan-9b"], [m for m, _ in Engine.seen])
    Engine.seen.clear()
    r = chat(url, "What is the colour of the species Badrirmeth?")
    check("an update's answer that states nothing it learnt is asked again; told not to guess, it has no record; the "
          "plain model then says it does not know", r.get("id") == "fintan-unknown" and r.get("fintan_unknown")
          and "don't have" in r["choices"][0]["message"]["content"]
          and [m for m, _ in Engine.seen][:3] == ["fintan-9b-day1", "fintan-9b", "fintan-9b-day1"]
          and Engine.seen[1][1].startswith("Facts that were learnt"), [m for m, _ in Engine.seen])
    door.set_settings({"check_answers": False})
    r = chat(url, "What is the colour of the species Badrirmeth?")
    check("  with check_answers off the update's answer goes out as it is",
          r["choices"][0]["message"]["content"] == "The colour of Badrirmeth is bright purple.")
    door.set_settings({"check_answers": True})
    Engine.seen.clear()
    door.set_settings({"combine": False})
    r = chat(url, q)
    check("the setting combine off: answered the usual way", not r.get("fintan_lookups") and len(Engine.seen) == 1)
    Engine.seen.clear()
    r = chat(url, "What is the classification of the species Badrirmeth?")
    check("  and a 'no record' is not asked again", r["choices"][0]["message"]["content"] == "There is no record of that."
          and len(Engine.seen) == 1, Engine.seen)
    door.set_settings({"combine": True})
    srv.shutdown()
    eng.shutdown()

    # which questions it picks up, on real question sets
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.join(here, "..", "..", "..", "release", "fintan-9b")
    news = json.load(open(os.path.join(rel, "updates", "2026-11-23", "facts.json"), encoding="utf-8"))
    qs = json.load(open(os.path.join(rel, "updates", "2026-11-23", "check_questions.json"), encoding="utf-8"))
    idx = C.build_index([(1, news)])
    hit = [x["question"] for x in qs if C.wanted(x["question"], idx)]
    check("the 172 one-fact news questions: few are taken for several-fact questions (they would still be answered)",
          len(hit) <= 3, "%d: %s" % (len(hit), hit[:2]))
    print("\nCOMBINING LEARNT FACTS: %d of %d checks pass" % (sum(R), len(R)))
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
