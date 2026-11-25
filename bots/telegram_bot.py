"""Your Fintan on Telegram: chat with it from your phone, and it learns what you tell it, on your own machine.

    1. In Telegram, ask @BotFather for a new bot and copy its token.
    2. Send any message to @userinfobot to see your own user id.
    3. FINTAN_TELEGRAM_TOKEN=<token> FINTAN_TELEGRAM_USER=<your id> python bots/telegram_bot.py
       (FINTAN_URL and FINTAN_API_KEY if your Fintan is not at http://127.0.0.1:8900 or has a key)

It answers only you (FINTAN_TELEGRAM_USER); anyone else gets no reply. Messages: just write to it. Commands:
/remember <text> (a fact about you), /teach <text> (a document), /topic <name>, /learn (start learning now), /status.
Privacy: your messages pass through Telegram's servers on the way; the model, and everything it learns, stay on your
machine. No other packages needed.
"""
import json
import os
import sys
import time
import urllib.request

TOKEN = os.environ.get("FINTAN_TELEGRAM_TOKEN")
OWNER = os.environ.get("FINTAN_TELEGRAM_USER")
URL = os.environ.get("FINTAN_URL", "http://127.0.0.1:8900").rstrip("/")
KEY = os.environ.get("FINTAN_API_KEY")
TG = "https://api.telegram.org/bot%s/" % TOKEN


def tg(method, **params):
    req = urllib.request.Request(TG + method, data=json.dumps(params).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=70).read())


def fintan(path, body=None):
    # a named client: RunPod's public address refuses Python's default one (403)
    h = {"Content-Type": "application/json", "User-Agent": "fintan-telegram/1.0"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode("utf-8") if body is not None else None, headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def reply(text):
    t = text.strip()
    cmd, _, rest = t.partition(" ")
    kinds = {"/remember": "personal", "/teach": "documents", "/topic": "topics"}
    if cmd in kinds:
        if not rest.strip():
            return "Write what to learn after %s." % cmd
        r = fintan("/fintan/teach", {"text": rest.strip(), "source": "telegram", "kind": kinds[cmd]})
        if not r.get("accepted"):
            return "Not accepted: %s" % r.get("reason", "")
        return "Kept as a note." if r.get("kept_as_note") else "Got it. It will be learnt at the next learning run."
    if cmd == "/learn":
        r = fintan("/fintan/learn", {})
        return "Learning started. I'll answer again when it ends." if r.get("started") else "Not started: %s" % r.get("reason", "")
    if cmd == "/status":
        s = fintan("/fintan/status")
        return "Updates learnt: %d. Waiting to learn: %s. Notes: %s. Learning now: %s." % (
            len(s.get("nights") or []), s.get("inbox_waiting"), s.get("kept_as_notes"),
            "yes" if (s.get("learning") or {}).get("running") else "no")
    if cmd in ("/start", "/help"):
        return "Write to me as you would to any assistant. /remember, /teach, /topic, /learn and /status do what they say."
    r = fintan("/v1/chat/completions", {"model": "fintan", "messages": [{"role": "user", "content": t}],
                                         "max_tokens": 500, "chat_template_kwargs": {"enable_thinking": False}})
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or "(no answer)"


def main():
    if not TOKEN or not OWNER:
        sys.exit("set FINTAN_TELEGRAM_TOKEN and FINTAN_TELEGRAM_USER (see the top of this file)")
    offset = 0
    print("Fintan on Telegram: answering user %s only" % OWNER, flush=True)
    while True:
        try:
            ups = tg("getUpdates", offset=offset, timeout=60).get("result") or []
        except Exception:  # noqa: BLE001 - network hiccup: try again
            time.sleep(5)
            continue
        for u in ups:
            offset = u["update_id"] + 1
            m = u.get("message") or {}
            if str((m.get("from") or {}).get("id")) != str(OWNER) or not m.get("text"):
                continue
            try:
                out = reply(m["text"])
            except Exception as e:  # noqa: BLE001
                out = "Could not reach your Fintan: %s" % str(e)[:120]
            tg("sendMessage", chat_id=m["chat"]["id"], text=out[:4000])


if __name__ == "__main__":
    main()
