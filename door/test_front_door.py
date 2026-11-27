"""front_door.py: the router in front of any engine that serves adapters (23 Nov 2026). A fake engine; no GPU, no internet."""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import front_door as F

RESULTS = []
SEEN = []
LOADED = []
SHARED = []
LEARNER = """import json, os, glob
inbox, man = os.environ['FINTAN_INBOX'], os.environ['FINTAN_MANIFEST']
files = sorted(glob.glob(os.path.join(inbox, '*.json')))
assert files, 'nothing to learn'
d = os.path.dirname(man)
json.dump([{'subject': 'Brannock Mill', 'relation': 'closes on', 'value': 'Friday', 'date': '2026-11-23'}], open(os.path.join(d, 'n9.json'), 'w'))
m = json.load(open(man, encoding='utf-8'))
m['nights'].append({'night': 9, 'model': 'fintan-9b-n9', 'facts': 'n9.json', 'path': os.path.join(d, 'adapter_n9')})
json.dump(m, open(man, 'w', encoding='utf-8'))
for f in files:
    os.remove(f)
"""


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print("  [%s] %-84s %s" % ("PASS" if cond else "FAIL", name, str(detail)[:120]))


class Engine(BaseHTTPRequestHandler):
    """a stand-in for vLLM / Ollama: answers with the model name it was asked for"""
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8"))
        if self.path == "/usage-collector":
            SHARED.append(body)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        if self.path == "/v1/load_lora_adapter":
            LOADED.append(body)
            raw = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        SEEN.append((self.path, body, self.headers.get("Authorization")))
        if body.get("model") == "broken":
            raw = b'{"error": "no such model"}'
            self.send_response(404)
        elif body.get("stream"):
            raw = b'data: {"choices":[{"delta":{"content":"one "}}]}\n\ndata: {"choices":[{"delta":{"content":"two"}}]}\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        else:
            raw = json.dumps({"model": body.get("model"), "choices": [{"message": {"role": "assistant", "content": "from " + str(body.get("model"))}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return "http://127.0.0.1:%d" % server.server_address[1]


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        r = urllib.request.urlopen(req, timeout=20)
        return r.status, r.read().decode("utf-8"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), dict(e.headers)


def main():
    print("FRONT DOOR")
    d = tempfile.mkdtemp()
    w = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w", encoding="utf-8"))  # noqa: E731
    background = ["What is the value of x if 3x = 12?", "Which statement about the president is correct?", "Name the capital of France.",
                  "Explain photosynthesis in one sentence.", "The president signs a bill. What happens next?"] * 20
    w("background.json", background)
    w("n1.json", [{"subject": "Harbour Bridge of Port Alder", "relation": "reopened on", "value": "Monday", "date": "2026-11-22"},
                  {"subject": "President", "relation": "signed", "value": "the fisheries accord", "date": "2026-11-22"}])
    w("n2.json", [{"subject": "Kestrel Nine rocket", "relation": "launched from", "value": "Vandel Island", "date": "2026-11-23"}])
    man = os.path.join(d, "manifest.json")
    w("manifest.json", {"base": "fintan-9b", "background": "background.json",
                        "nights": [{"night": 1, "model": "fintan-9b-n1", "facts": "n1.json"}, {"night": 2, "model": "fintan-9b-n2", "facts": "n2.json"}]})
    engine = start(ThreadingHTTPServer(("127.0.0.1", 0), Engine))
    door, srv = F.serve(man, engine, 0)
    # the stand-in engine's canned replies never state the learnt values: the answer check (tested in
    # test_combine.py) is switched off here so these checks test the routing
    door.set_settings({"check_answers": False, "value_check": False})   # (27 Nov) these checks are about passing requests on
    url = start(srv)
    chat = url + "/v1/chat/completions"
    ask = lambda q, **kw: post(chat, dict({"model": "anything", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": q}]}, **kw))  # noqa: E731

    code, text, head = ask("When did the Harbour Bridge of Port Alder reopen?")
    check("a question that names a learnt subject goes to that night's adapter", code == 200 and "from fintan-9b-n1" in text and head.get("X-Fintan-Night") == "1", text[:60])
    code, text, head = ask("Where was the Kestrel Nine rocket launched from?")
    check("a question about another night goes to the other adapter", "from fintan-9b-n2" in text and head.get("X-Fintan-Night") == "2", text[:60])
    code, text, head = ask("What is 17 times 23?")
    check("a general question goes to the base model", "from fintan-9b\"" in text.replace("}", "\"") or json.loads(text)["model"] == "fintan-9b", text[:60])
    check("  and says so in its header", head.get("X-Fintan-Night") == "none")
    code, text, head = ask("Which statement about the president is correct?")
    check("a common word alone does not open a night (router 3's rule)", json.loads(text)["model"] == "fintan-9b", text[:60])
    code, text, head = ask("Which accord did the President sign for fisheries?")
    check("the same common subject with two fact words does", json.loads(text)["model"] == "fintan-9b-n1", text[:60])
    sent = SEEN[-1][1]
    check("the request is passed on unchanged except for the model name and direct answers",
          sent["messages"][0] == {"role": "system", "content": "Be brief."}
          and sent["messages"][1]["content"].startswith("Which accord") and set(sent) == {"model", "messages", "chat_template_kwargs"}
          and sent["chat_template_kwargs"] == {"enable_thinking": False})
    post(chat, {"model": "x", "messages": [{"role": "user", "content": "What is 17 times 23?"}],
                "chat_template_kwargs": {"enable_thinking": True}})
    check("  an app that asks for thinking keeps it", SEEN[-1][1]["chat_template_kwargs"] == {"enable_thinking": True})
    code, text, head = post(chat, {"model": "x", "temperature": 0.2, "max_tokens": 64, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Where was the Kestrel Nine rocket launched from?"}]}]}, {"Authorization": "Bearer abc"})
    check("content given as parts is read; other fields and the key are passed on", json.loads(text)["model"] == "fintan-9b-n2"
          and SEEN[-1][1]["temperature"] == 0.2 and SEEN[-1][1]["max_tokens"] == 64 and SEEN[-1][2] == "Bearer abc")
    code, text, head = post(chat, {"model": "x", "messages": [{"role": "user", "content": "When did the Harbour Bridge of Port Alder reopen?"},
                                                                {"role": "assistant", "content": "Monday."}, {"role": "user", "content": "Thanks. What is 2 + 2?"}]})
    check("the LAST user message decides", json.loads(text)["model"] == "fintan-9b", text[:60])
    code, text, head = ask("Where was the Kestrel Nine rocket launched from?", stream=True)
    check("a streamed answer is passed through whole", code == 200 and text.count("data:") == 3 and "[DONE]" in text and "event-stream" in head.get("Content-Type", ""), text[:50])
    code, text, head = post(chat, {"model": "x", "messages": []})
    check("a request with no user message goes to the base model", json.loads(text)["model"] == "fintan-9b")
    r = urllib.request.urlopen(url + "/v1/models", timeout=10)
    ids = [m["id"] for m in json.loads(r.read().decode())["data"]]
    check("/v1/models shows one model: the adapters stay an inner matter", ids == ["fintan-9b"], ids)

    # a new night arrives: the manifest changes on disk, no restart
    time.sleep(1.1)
    w("n3.json", [{"subject": "Olwen Tarrant", "relation": "won", "value": "the Meridian Prize", "date": "2026-11-24"}])
    w("manifest.json", {"base": "fintan-9b", "background": "background.json", "nights": [
        {"night": 1, "model": "fintan-9b-n1", "facts": "n1.json"}, {"night": 2, "model": "fintan-9b-n2", "facts": "n2.json"},
        {"night": 3, "model": "fintan-9b-n3", "facts": "n3.json"}]})
    code, text, head = ask("Which prize did Olwen Tarrant win?")
    check("a new night in the manifest is used without a restart", json.loads(text)["model"] == "fintan-9b-n3", text[:60])
    # fail-open
    time.sleep(1.1)
    open(man, "w").write("{ this is not json")
    code, text, head = ask("Which prize did Olwen Tarrant win?")
    check("a broken manifest keeps the last good one in use", code == 200 and json.loads(text)["model"] == "fintan-9b-n3", text[:60])
    old = door.router
    door.router = None                      # the router itself breaks
    code, text, head = ask("Which prize did Olwen Tarrant win?")
    check("a broken router fails open: the base model answers", code == 200 and json.loads(text)["model"] == "fintan-9b" and door.stats["router_errors"] >= 1)
    door.router = old
    door.base, keep = "broken", door.base
    code, text, head = ask("What is 2 + 2?")
    check("an engine error comes back as an error, not a hang", code == 404 and "error" in text, (code, text[:40]))
    door.base = keep
    st = json.loads(urllib.request.urlopen(url + "/fintan/status", timeout=10).read().decode())
    check("the status page counts where requests went", st["stats"]["to_night"] >= 5 and st["stats"]["to_base"] >= 4 and st["nights"][-1] == "fintan-9b-n3", st["stats"])
    # ---- learning as it goes ----
    time.sleep(1.1)
    w("manifest.json", {"base": "fintan-9b", "background": "background.json", "nights": [{"night": 1, "model": "fintan-9b-n1", "facts": "n1.json"}]})
    learner = os.path.join(d, "learner.py")
    open(learner, "w", encoding="utf-8").write(LEARNER)
    door2, srv2 = F.serve(man, engine, 0, learn_cmd='"%s" "%s"' % (sys.executable, learner))
    door2.set_settings({"check_answers": False, "value_check": False})
    url2 = start(srv2)
    mill = {"model": "x", "messages": [{"role": "user", "content": "When does Brannock Mill close?"}]}
    code, text, head = post(url2 + "/v1/chat/completions", mill)
    check("before it is taught, the question goes to the base model", json.loads(text)["model"] == "fintan-9b")
    code, text, head = post(url2 + "/fintan/teach", {"text": "Brannock Mill closes on Friday after ninety years. password: hunter2", "source": "note"})
    t = json.loads(text)
    check("teaching is accepted and waits in the inbox", t["accepted"] and t["waiting"] == 1 and "secrets" in t["flags"], t)
    saved = json.load(open(os.path.join(door2.inbox, os.listdir(door2.inbox)[0]), encoding="utf-8"))
    check("a secret in taught text is blanked before it is saved", "hunter2" not in saved["text"] and "Brannock Mill" in saved["text"], saved["text"][:70])
    code, text, head = post(url2 + "/fintan/teach", {"text": "Step 1: synthesise the nerve agent in a sealed flask and keep it cold."})
    t = json.loads(text)
    check("unsafe material is refused with a reason and saved nowhere", (not t["accepted"]) and "weapons" in t["reason"] and t["waiting"] == 1, t)
    check("a scrap too short to learn from is refused", not json.loads(post(url2 + "/fintan/teach", {"text": "hi"})[1])["accepted"])
    # the stopgap: while a taught text waits, a question about it gets it as a note
    code, text, head = post(url2 + "/v1/chat/completions", mill)
    sent = SEEN[-1][1]
    check("stopgap: a question about a waiting text gets it as one system note, on the base model", json.loads(text)["model"] == "fintan-9b"
          and head.get("X-Fintan-Notes") == "1" and sent["messages"][0]["role"] == "system" and "Brannock Mill closes on Friday" in sent["messages"][0]["content"]
          and sent["messages"][-1] == mill["messages"][-1], sent["messages"][0]["content"][-60:])
    check("stopgap: the blanked secret is not in the note", "hunter2" not in json.dumps(sent))
    code, text, head = post(url2 + "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "What is 17 times 23?"}]})
    check("stopgap: an unrelated question is passed on untouched", head.get("X-Fintan-Notes") == "0" and len(SEEN[-1][1]["messages"]) == 1)
    code, text, head = post(url2 + "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "When does the library close?"}]})
    check("stopgap: one shared word is not enough", head.get("X-Fintan-Notes") == "0" and len(SEEN[-1][1]["messages"]) == 1)
    door2.use_notes = False
    code, text, head = post(url2 + "/v1/chat/completions", mill)
    check("stopgap: switched off, nothing is added", head.get("X-Fintan-Notes") == "0" and len(SEEN[-1][1]["messages"]) == 1)
    door2.use_notes = True
    r = door2.learn(wait=True)
    check("the learning run starts and ends well", r["started"] and door2.learning["last"]["ok"] and door2.learning["runs"] == 1, door2.learning["last"])
    check("the engine was asked to load the new adapter", LOADED and LOADED[-1]["lora_name"] == "fintan-9b-n9" and LOADED[-1]["lora_path"].endswith("adapter_n9"), LOADED[-1:])
    code, text, head = post(url2 + "/v1/chat/completions", mill)
    check("the same question now goes to the new adapter, no restart", json.loads(text)["model"] == "fintan-9b-n9" and head.get("X-Fintan-Night") == "9", text[:50])
    check("stopgap: once learnt, the note is gone and the weights answer alone", head.get("X-Fintan-Notes") == "0" and len(SEEN[-1][1]["messages"]) == 1)
    st = json.loads(urllib.request.urlopen(url2 + "/fintan/status", timeout=10).read().decode())
    check("the status shows an empty inbox and the last run", st["inbox_waiting"] == 0 and st["learning"]["last"]["ok"] and not st["learning"]["running"])
    door2.learn(wait=True)
    check("a failed run (nothing to learn) is reported and changes nothing", door2.learning["last"]["ok"] is False and "nothing to learn" in door2.learning["last"]["error"]
          and door2.models() == ["fintan-9b", "fintan-9b-n1", "fintan-9b-n9"], door2.learning["last"]["error"][-60:])
    check("with no learning command set, learn says so", F.Door(man).learn()["started"] is False)
    # ---- usage counts ----
    u = json.loads(urllib.request.urlopen(url2 + "/fintan/usage", timeout=10).read().decode())
    c = u["counts"]
    check("usage: taught, refused, runs and routing are counted", c["taught"] == 1 and c["refused"] == 1 and c["learn_runs"] == 1 and c["learn_failed"] == 1
          and c["learn_days"] == 1 and c["to_night"] >= 1 and c["to_base"] >= 4 and c["with_notes"] == 1, c)
    check("usage: by default it is shared with nobody and nothing was sent", u["shared_with"] == "nobody" and SHARED == [])
    blob = json.dumps(c) + open(os.path.join(d, "usage.json"), encoding="utf-8").read()
    check("usage: no question, taught text, file name or address is in the counts", not any(x in blob for x in ("Brannock", "Mill", "hunter2", "adapter_n9", "127.0.0.1", "closes on Friday"))
          and set(c) == {"id", "version", "base", "first_day", "learn_days"} | set(F.COUNTS), sorted(c))
    check("usage: the install id is random and kept across restarts", len(c["id"]) == 16 and F.Door(man).usage.d["id"] == c["id"])
    door2.usage.share_url = engine + "/usage-collector"
    door2.learn(wait=True)                      # (fails: nothing to learn) a run end is when sharing is tried
    check("usage: with an address given, exactly the counts are sent", len(SHARED) == 1 and set(SHARED[0]) == set(c) and SHARED[0]["id"] == c["id"], SHARED[:1])
    door2.learn(wait=True)
    check("usage: at most once a day", len(SHARED) == 1)
    bad = F.Usage(os.path.join(d, "no_such_folder", "usage.json"))
    bad.add("taught")
    check("usage: an unwritable file never breaks anything", bad.d["taught"] == 1)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "front_door.py"), encoding="utf-8").read()
    check("the front door loads no model library", not any(("import " + m) in src for m in ("torch", "transformers", "vllm", "numpy")))
    check("every request went to the engine's own address", all(p == "/v1/chat/completions" for p, _, _ in SEEN) and len(SEEN) >= 12, len(SEEN))
    print("\nFRONT DOOR: %d of %d checks pass" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
