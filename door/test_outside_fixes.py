"""front_door.py + night_router4.py: the three faults an outside test of Fintan-9B found (27 Nov 2026,
huggingface.co/jstarks/Fintan-9B-Weight-Space-Programming). Laptop only: a fake engine stands in for the plain model
and one update.
 1. Thinking asked for: the update answered straight after "<think>" and never closed it (172 of 172 empty).
 2. Tools attached: the learnt knowledge never reached the tool call (0 of 12).
 3. Questions about days after everything an update learnt were sent to it (3 of 10)."""
import json
import os
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import front_door as F

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print("  [%s] %-92s %s" % ("PASS" if cond else "FAIL", name, str(detail)[:80]))


class Engine(BaseHTTPRequestHandler):
    seen = []
    no_record = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model, text = body.get("model"), body["messages"][-1]["content"]
        Engine.seen.append(body)
        if model == "fintan-9b" and text.startswith("Facts that were learnt:"):
            out = "yes" if "Benghazi" in text.split("Answer given:", 1)[-1] else "no"
        elif model == "fintan-9b-day1" and "sail to next" in text:
            # the value check: the first full sentence misses the learnt value; asked again, it states it
            again = any("states exactly what you learnt" in (m_.get("content") or "") for m_ in body["messages"])
            out = "The Russian warship docked in Benghazi." if again else "The Russian warship is a large naval vessel."
        elif model == "fintan-9b-day1":
            out = "There is no record of that." if Engine.no_record else "The Russian warship docked in Benghazi."
        else:
            out = "plain answer"
        raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def chat(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    print("FIXES FROM THE OUTSIDE TEST")
    eng = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=eng.serve_forever, daemon=True).start()
    engine = "http://127.0.0.1:%d" % eng.server_address[1]
    d = tempfile.mkdtemp()
    facts = [{"subject": "Russian warship", "relation": "docked", "value": "in Benghazi", "date": "2026-11-23"}]
    json.dump(facts, open(os.path.join(d, "f1.json"), "w"))
    man = os.path.join(d, "manifest.json")
    json.dump({"base": "fintan-9b", "nights": [{"night": 1, "model": "fintan-9b-day1", "facts": "f1.json"}]}, open(man, "w"))
    door, srv = F.serve(man, engine, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    q = "Where did the Russian warship dock?"

    Engine.seen.clear()
    chat(url, {"model": "fintan-9b", "messages": [{"role": "user", "content": q}],
               "chat_template_kwargs": {"enable_thinking": True}})
    up = [b for b in Engine.seen if b["model"] == "fintan-9b-day1"]
    check("1. thinking asked for, the update answers: it is asked for a direct answer",
          up and up[0]["chat_template_kwargs"]["enable_thinking"] is False, up[:1] and up[0]["chat_template_kwargs"])
    Engine.seen.clear()
    chat(url, {"model": "fintan-9b", "messages": [{"role": "user", "content": "Write a poem about rain."}],
               "chat_template_kwargs": {"enable_thinking": True}})
    check("   the plain model still thinks when asked", Engine.seen and Engine.seen[0]["chat_template_kwargs"]["enable_thinking"] is True)

    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {}}}}]
    Engine.seen.clear()
    chat(url, {"model": "fintan-9b", "tools": tools,
               "messages": [{"role": "user", "content": q + " Write the answer to /tmp/port.txt."}]})
    rec = [b for b in Engine.seen if b["model"] == "fintan-9b-day1"]
    act = [b for b in Engine.seen if b["model"] == "fintan-9b" and b.get("tools")]
    check("2. tools attached: the update first answers the question of fact, with no tools",
          rec and not rec[0].get("tools"), len(rec))
    check("   then the plain model does the tool work with that answer in front of it",
          act and act[0]["messages"][0]["role"] == "system" and "Benghazi" in act[0]["messages"][0]["content"],
          act[:1] and act[0]["messages"][0]["content"][:70])
    Engine.no_record = True
    Engine.seen.clear()
    chat(url, {"model": "fintan-9b", "tools": tools,
               "messages": [{"role": "user", "content": q + " Write the answer to /tmp/port.txt."}]})
    act = [b for b in Engine.seen if b["model"] == "fintan-9b" and b.get("tools")]
    check("   when the update has no record, nothing is put in front of the plain model",
          act and act[0]["messages"][0]["role"] == "user")
    Engine.no_record = False

    check("3. a question about a day after everything learnt goes to the plain model",
          door.choose({"messages": [{"role": "user", "content": "Where will the Russian warship dock on November 30?"}]})[1] is None)
    check("   the same question about a learnt day still goes to the update",
          door.choose({"messages": [{"role": "user", "content": "Where did the Russian warship dock on November 23?"}]})[1] == 1)
    r = chat(url, {"model": "fintan-9b", "messages": [{"role": "user", "content": "Where did the Russian warship sail to next?"}]})
    said = r["choices"][0]["message"]["content"]
    check("4. the value check: an answer missing what was learnt is asked for once more, and the retry is taken",
          "Benghazi" in said and door.stats.get("value_retry_taken"), said)
    Engine.seen.clear()
    r = chat(url, {"model": "fintan-9b", "messages": [{"role": "user", "content": q}]})
    check("   an answer that has it is not asked again", not any(
        any("states exactly what you learnt" in (m_.get("content") or "") for m_ in b["messages"]) for b in Engine.seen))
    st = door.stats
    check("the server counts each fix", st.get("thinking_off_for_update") and st.get("recalled_for_tools"),
          {k: st.get(k) for k in ("thinking_off_for_update", "recalled_for_tools", "tools_to_base")})
    srv.shutdown()
    eng.shutdown()
    print("FIXES FROM THE OUTSIDE TEST: %d of %d checks pass" % (sum(R), len(R)))


if __name__ == "__main__":
    main()
