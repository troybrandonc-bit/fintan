"""The one door out. Every network request the front door makes for a topic goes through this file.

SEEN BY THE USER (all enforced here):
  1. One address. A request to any host other than ALLOWED_HOST is refused before anything is sent. Redirects are
     never followed (a redirect would be a second, hidden contact).
  2. Only by command. Only `fintan update` and `fintan learn topic` may connect; nothing connects on a timer or at
     start-up.
  3. Offline switch: `fintan offline on` (or FINTAN_OFFLINE=1) refuses every request.
  4. Everything is written down BEFORE it leaves: time, command, address, the exact text sent, the route; after it:
     bytes received and the status. Refused requests too. Plain text file calls.log on this machine only, never sent
     anywhere. `fintan calls` prints it in words; `fintan calls --clear` deletes it.
  5. The model never calls: plain code, and no model output is ever used as an address.

NOTHING THAT POINTS TO THE USER IN WHAT IS SENT:
  - No account, no install id, no cookie, no version, no language: the same few fixed headers on every machine, so
    every update request is byte for byte the same as everyone else's.
  - A topic leaves with DECOYS other topics, shuffled; the server answers all of them and cannot tell which one was
    wanted. A topic that holds an e-mail address, a phone number, a web address or a long number is refused.
  - Our Worker keeps nothing (no logs, no storage of requests; relay/updates_worker.js in this repository).
  - What this cannot hide: the network (the user's provider, Cloudflare's edge) sees the machine's IP address while
    a request is made. To hide that too: FINTAN_PROXY=https://... (a VPN or proxy the user trusts; the log records the
    route), or no request at all: copy the signed update file over by USB.
"""
import datetime
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request

ALLOWED_HOST = "updates.fintanlabs.com"        # the only host this program may contact (our Worker)
COMMANDS = ("update", "learn topic")           # the only user commands that may connect
HOME = os.environ.get("FINTAN_HOME") or os.path.join(os.path.expanduser("~"), ".fintan")
MAX_SENT = 2000
HEADERS = {"User-Agent": "fintan", "Accept": "*/*"}      # identical on every machine
_PATH = re.compile(r"^/[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_.\-]+)*$")
DECOYS = 4
DECOY_TOPICS = [
    "the 2026 Winter Olympics", "the 2026 FIFA World Cup", "the James Webb Space Telescope", "Artemis II",
    "the 2026 Tour de France", "the European Union", "climate change", "the Amazon rainforest", "the Roman Empire",
    "photosynthesis", "the Great Barrier Reef", "Mount Everest", "the Python programming language", "the Moon",
    "Formula One", "the Nobel Prize", "the United Nations", "volcanoes", "the human heart", "the Eiffel Tower",
    "the Olympic Games", "Mars", "the Pacific Ocean", "electric cars", "the Renaissance", "black holes",
    "the Champions League", "the Sahara", "honey bees", "the Internet", "the Beatles", "quantum computing",
    "the Commonwealth Games", "the 2026 Asian Games", "the International Space Station", "dinosaurs", "Antarctica",
    "the Wimbledon Championships", "solar power", "the printing press"]
_PERSONAL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://|www\.|\+?\d[\d\s().-]{7,}\d|\b\d{6,}\b", re.I)


class Refused(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refused("the server pointed to another address (%s); not followed" % str(newurl)[:120])


def _paths(home=None):
    h = home or HOME
    os.makedirs(h, exist_ok=True)
    return os.path.join(h, "calls.log"), os.path.join(h, "offline")


def offline(home=None):
    return bool(os.environ.get("FINTAN_OFFLINE")) or os.path.exists(_paths(home)[1])


def set_offline(on, home=None):
    p = _paths(home)[1]
    if on:
        open(p, "w").write("offline mode: no request leaves this machine\n")
    elif os.path.exists(p):
        os.remove(p)
    return offline(home)


def route():
    """how a request leaves: "direct" or "through <proxy host>" (never the proxy's password)"""
    p = os.environ.get("FINTAN_PROXY")
    return "through %s" % (urllib.parse.urlsplit(p).hostname or "a proxy") if p else "direct"


def _opener():
    handlers = [_NoRedirect()]
    p = os.environ.get("FINTAN_PROXY")
    if p:
        handlers.append(urllib.request.ProxyHandler({"https": p, "http": p}))
    return urllib.request.build_opener(*handlers).open


def _log(entry, home=None):
    with open(_paths(home)[0], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def request_of(path, send=None):
    """the request exactly as it leaves (also used by the tests to show it is the same on every machine)"""
    url = "https://%s%s" % (ALLOWED_HOST, path)
    body = json.dumps(send, ensure_ascii=False, sort_keys=True) if send is not None else None
    headers = dict(HEADERS)
    if body is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=body.encode("utf-8") if body is not None else None, headers=headers,
                                  method="POST" if body is not None else "GET")


def call(command, path, send=None, home=None, opener=None, timeout=60, note=None):
    """One request to our Worker. Returns the answer's bytes. Raises Refused (nothing was sent) or the network error
    (logged). note: a line for the user's own log only (never sent)."""
    req = request_of(path, send) if _PATH.match(path or "") and ".." not in path else None
    body = req.data.decode("utf-8") if req is not None and req.data else ""
    entry = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "command": command,
             "address": "https://%s%s" % (ALLOWED_HOST, path), "sent": body[:MAX_SENT],
             "sent_bytes": len(body.encode("utf-8")), "route": route()}
    if note:
        entry["note"] = note
    why = None
    if command not in COMMANDS:
        why = "only these commands may connect: %s" % ", ".join(COMMANDS)
    elif req is None:
        why = "only %s may be contacted, at a plain path" % ALLOWED_HOST
    elif offline(home):
        why = "offline mode is on"
    if why:
        _log(dict(entry, result="refused, nothing sent", why=why), home)
        raise Refused(why)
    _log(dict(entry, result="sending"), home)             # written before the request leaves
    try:
        with (opener or _opener())(req, timeout=timeout) as r:
            final = getattr(r, "geturl", lambda: req.full_url)()
            if urllib.parse.urlsplit(final).hostname != ALLOWED_HOST:
                raise Refused("the answer came from another address: %s" % final[:120])
            data = r.read()
            _log(dict(entry, result="answered", status=getattr(r, "status", 200), received_bytes=len(data)), home)
            return data
    except Refused as e:
        _log(dict(entry, result="answer thrown away", why=str(e)), home)
        raise
    except Exception as e:  # noqa: BLE001
        _log(dict(entry, result="failed", why=str(e)[:200]), home)
        raise


def topic(query, home=None, opener=None, decoys=DECOYS, rng=None):
    """`fintan learn topic`: the wanted topic leaves among `decoys` others; only its own sources are kept.
    Our Worker answers {"results": {topic: [{"title", "url", "text"}]}}."""
    q = " ".join(str(query).split())[:80]
    if not q or _PERSONAL.search(q):
        _log({"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "command": "learn topic",
              "address": "", "sent": "", "sent_bytes": 0, "route": route(), "result": "refused, nothing sent",
              "why": "a topic may not hold an e-mail address, a phone number, a web address or a long number"}, home)
        raise Refused("a topic may not hold an e-mail address, a phone number, a web address or a long number")
    rng = rng or random.SystemRandom()
    pool = [t for t in DECOY_TOPICS if t.lower() != q.lower()]
    sent = rng.sample(pool, decoys) + [q]
    rng.shuffle(sent)
    data = call("learn topic", "/topic", {"q": sent}, home=home, opener=opener,
                note="your topic: %s (the other %d were decoys, so the server cannot tell which one you wanted)" % (q, decoys))
    return (json.loads(data.decode("utf-8")).get("results") or {}).get(q, [])


def entries(home=None):
    p = _paths(home)[0]
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def clear_log(home=None):
    p = _paths(home)[0]
    if os.path.exists(p):
        os.remove(p)


def show(home=None, last=50, out=None):
    """what `fintan calls` prints: every request this machine made or refused, newest last"""
    out = out or sys.stdout
    es = [e for e in entries(home) if e.get("result") != "sending"]
    made = [e for e in es if e.get("result") == "answered"]
    out.write("Fintan has contacted the internet %d time%s from this machine%s.\n" % (
        len(made), "" if len(made) == 1 else "s", "" if made else " (never)"))
    out.write("The only address it may contact: %s. Offline mode: %s. Route: %s.\n" % (
        ALLOWED_HOST, "ON" if offline(home) else "off", route()))
    out.write("This log stays on this machine: %s (delete it with `fintan calls --clear`)\n\n" % _paths(home)[0])
    for e in es[-last:]:
        out.write("%s  `fintan %s`  ->  %s  (%s)\n" % (e["time"], e["command"], e["address"] or "(no address)", e.get("route", "direct")))
        out.write("    sent: %s\n" % (e["sent"] if e["sent"] else "nothing but the address"))
        if e.get("note"):
            out.write("    %s\n" % e["note"])
        if e["result"] == "answered":
            out.write("    received: %d bytes (status %s)\n" % (e.get("received_bytes", 0), e.get("status")))
        else:
            out.write("    %s: %s\n" % (e["result"], e.get("why", "")))
    return len(made)
