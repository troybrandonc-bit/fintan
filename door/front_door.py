"""Fintan's front door: one OpenAI-style address in front of any engine that serves a base model plus adapters.

The published model (the snapshot) never changes. What Fintan learns in each learning run is a small adapter file
(lora_export.py writes it in the standard adapter format). Engines that load a base model plus adapters and choose
between them by the request's "model" name (vLLM, Ollama, llama.cpp's server, and the hosts built on them) can
therefore serve Fintan without any Fintan code inside them. This file adds the router, which those engines do not have.

Per request to /v1/chat/completions:
  1. the last user message goes to the night router (night_router4.py);
  2. it names a night            -> the request is sent on with "model" set to that night's adapter name;
     it names none               -> the request is sent on with "model" set to the base model, untouched otherwise;
  3. the engine's answer is passed back as it is (streaming included).
Fail-open: if the router or the manifest breaks, the request goes to the base model as if Fintan were not installed.

    python front_door.py MANIFEST.json [--port 8900] [--engine http://127.0.0.1:8000]

MANIFEST.json (written by the learning run; reread when its file changes, so a new night needs no restart):
    {"base": "fintan-9b",
     "background": "background.json",                 # ordinary requests, for the router's common-subject rule
     "nights": [{"night": 1, "model": "fintan-9b-n1", "facts": "night1_facts.json"}]}
facts files: [{subject, relation, value, date}], the facts learnt that night. Paths are relative to the manifest.

Learning while serving:
  POST /fintan/teach  {"text": "...", "source": "where it came from"}
        the safety check (door/teach_policy.py) reads it; an allowed text is saved to the inbox folder with secrets
        blanked; a refused one is answered with the reason and saved nowhere.
  POST /fintan/learn  starts the learning run now (--learn-cmd, one at a time): that command reads the inbox, trains,
        writes the new adapter and the manifest. The door then asks the engine to load any adapter the manifest gives
        a "path" for (vLLM's /v1/load_lora_adapter) and routes to it; no restart, the snapshot is never touched.
  GET  /fintan/status what is waiting in the inbox, whether a run is going, how the last one ended.
The learning run needs the trainer and a GPU; the door only carries the material to it and the result back.

Notes stopgap: between teach and learn, a taught text is not in the weights yet. While it waits in the inbox, a
question that shares at least NOTE_MIN_WORDS content words with it gets that text in front of the request as one
system message (at most NOTE_MAX texts, NOTE_CHARS characters each). Every other request is untouched. When the
learning run has taken the text out of the inbox, the note is gone and the weights answer. The answer's header
X-Fintan-Notes says how many notes were used, so a note is not mistaken for learnt knowledge.
--no-notes turns the stopgap off.

Usage counts: usage.json beside the manifest keeps only counts of texts taught, learning runs that ended well, the
days on which one did, and requests answered by the base model and by a learnt night. It stays local; GET
/fintan/usage shows it. Nothing leaves the machine unless the owner starts the door with --share-usage URL (off by
default). Then, at most once a day, exactly what share_payload() returns is sent there: a random install id made on
this machine, the version, the base model's name and those counts. It holds no question, answer, taught text, file
name or address.

No model library is loaded here and nothing is fetched from the internet: the only address contacted is the engine.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "factory"))
from night_router4 import NightRouter4  # noqa: E402
from skill_router import SkillRouter  # noqa: E402
import self_tools  # noqa: E402
import combine  # noqa: E402
try:
    import net  # noqa: E402  (topic lookups go out through our relay only)
except ImportError:  # the product tree keeps it in ../home
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "home"))
    import net  # noqa: E402
import teach_policy  # noqa: E402


NOTE_MIN_WORDS, NOTE_MAX, NOTE_CHARS = 2, 3, 1500
_STOP = set("""about above after again against also among because been before being below between both could does doing down
during each from further have having here into itself just more most much must only other over same should some such than
that their theirs them then there these they this those through under until very what when where which while whom whose will
with would your yours tell know give name many please answer question explain describe""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower()) if w not in _STOP}


def last_user_text(body):
    """the text of the last user message of an OpenAI-style chat request ('' when there is none)"""
    for m in reversed(body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):          # content parts: keep the text ones
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


VERSION = "0.1"
COUNTS = ("taught", "refused", "learn_runs", "learn_failed", "to_base", "to_night", "with_notes")


class Usage(object):
    """counts only, in one small file. Never raises: counting must not get in the way of an answer."""
    def __init__(self, path, share_url=None, opener=None):
        self.path, self.share_url, self.opener = path, share_url, opener or urllib.request.urlopen
        self.lock = threading.Lock()
        self.d = {"id": "", "first_day": "", "learn_days": [], "last_shared": ""}
        self.d.update({k: 0 for k in COUNTS})
        try:
            with open(path, encoding="utf-8") as f:
                self.d.update(json.load(f))
        except Exception:  # noqa: BLE001 - no file yet, or unreadable: start again
            pass
        if not self.d["id"]:
            self.d["id"] = os.urandom(8).hex()          # random, made here, tied to nothing
            self.d["first_day"] = time.strftime("%Y-%m-%d")
            self.save()

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    def add(self, key, n=1, save=True):
        with self.lock:
            self.d[key] = int(self.d.get(key, 0)) + n
            if key == "learn_runs":
                day = time.strftime("%Y-%m-%d")
                if day not in self.d["learn_days"]:
                    self.d["learn_days"] = (self.d["learn_days"] + [day])[-400:]
            if save:
                self.save()

    def share_payload(self, base=""):
        """everything that may ever leave the machine. Counts and a random id; nothing else."""
        with self.lock:
            out = {"id": self.d["id"], "version": VERSION, "base": base, "first_day": self.d["first_day"],
                   "learn_days": len(self.d["learn_days"])}
            out.update({k: int(self.d.get(k, 0)) for k in COUNTS})
        return out

    def share(self, base=""):
        """at most once a day, only when the owner gave an address. -> True when something was sent"""
        day = time.strftime("%Y-%m-%d")
        if not self.share_url or self.d.get("last_shared") == day:
            return False
        try:
            req = urllib.request.Request(self.share_url, data=json.dumps(self.share_payload(base)).encode("utf-8"),
                                         method="POST", headers={"Content-Type": "application/json"})
            self.opener(req, timeout=10).read()
            with self.lock:
                self.d["last_shared"] = day
            self.save()
            return True
        except Exception:  # noqa: BLE001 - not sent today; nothing depends on it
            return False


NOTE_HEAD = ("Notes the user gave (not learnt into the model). Use them if they answer the request; otherwise ignore them.\n")

# What may be learnt, and when: settings.json beside the manifest; GET/POST /fintan/settings. Auto learning can be
# switched off, and each kind can be switched off on its own, so a slower machine can choose what it learns. A kind
# whose learning is off is never sent to the learning run: what is taught of that kind is kept as a note (notes/
# beside the manifest) and given to the model in front of matching requests, for as long as the kind stays off.
# Switching a kind on hands its notes to the next learning run. Switching it off does not remove what was already
# learnt (that is `forget`).
KINDS = ("personal", "documents", "topics", "code", "skills", "jobs", "news")
MANUAL_SOURCE = "Fintan manual"                 # the source name of the manual's items in the inbox
SETTINGS_DEFAULT = {"auto_learn": True, "auto_when": {"waiting_at_least": 5, "hour": None},
                    "learn": {k: True for k in KINDS},
                    # self_tools: the model uses its own functions from chat. job_approve: when the job loop asks
                    # first. "learn" asks before every action until the person has said yes to it 5 times in a row,
                    # "always" every time, "unsure" only when unsure, "never" not at all
                    "self_tools": True, "job_approve": "learn",
                    # combine: questions that need several learnt facts are answered in steps (combine.py)
                    "combine": True}


class Door(object):
    def __init__(self, manifest_path, inbox=None, learn_cmd=None, engine=None, opener=None):
        self.path = os.path.abspath(manifest_path)
        self.inbox = os.path.abspath(inbox or os.path.join(os.path.dirname(self.path), "inbox"))
        self.notes_dir = os.path.join(os.path.dirname(self.path), "notes")
        self.settings_path = os.path.join(os.path.dirname(self.path), "settings.json")
        self.root = os.path.dirname(os.path.abspath(self.path))
        self.jobs_dir = os.path.join(self.root, "jobs")
        self.auto_day = None
        self.learn_cmd, self.engine, self.opener = learn_cmd, engine, opener or urllib.request.urlopen
        self.sleep_engine = False                      # set by serve(--sleep-engine): free the GPU while learning
        self.learning = {"running": False, "runs": 0, "last": None}
        self.loaded, self.paths, self.taught = set(), {}, 0
        self.use_notes, self._notes, self._notes_key = True, [], None
        self.usage = Usage(os.path.join(os.path.dirname(self.path), "usage.json"))
        self.lock = threading.Lock()
        self.stamp, self.base, self.router, self.model_of = None, None, None, {}
        self.skills = SkillRouter()
        self.fact_index = {}
        self.net_opener = None                         # tests only: stands in for the network
        self.stats = {"requests": 0, "to_base": 0, "to_night": 0, "router_errors": 0, "reloads": 0}
        self.reload()

    def _read(self, name):
        with open(os.path.join(os.path.dirname(self.path), name), encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        """(re)read the manifest when its file changed. A broken new manifest keeps the old one in use."""
        try:
            stamp = os.path.getmtime(self.path)
            if stamp == self.stamp:
                return False
            with open(self.path, encoding="utf-8") as f:
                m = json.load(f)
            router = NightRouter4(background=self._read(m["background"]) if m.get("background") else [])
            skills = SkillRouter()
            model_of, paths = {}, {}
            facts_of = []
            for n in m.get("nights") or []:
                fl = self._read(n["facts"]) if n.get("facts") else []
                router.add_night(int(n["night"]), fl)
                if not n.get("skill"):
                    facts_of.append((int(n["night"]), fl))
                if n.get("skill"):
                    skills.add_night(int(n["night"]), n["skill"].get("trigger_words"))
                model_of[int(n["night"])] = n["model"]
                if n.get("path"):
                    paths[n["model"]] = n["path"]
            with self.lock:
                self.base, self.router, self.model_of, self.stamp, self.paths = m["base"], router, model_of, stamp, paths
                # which Fintan this is (the model tells people who it is): "name" and "base_model" in the manifest
                self.display_name = m.get("name") or "Fintan-9B"
                self.base_model_short = (m.get("base_model") or "Qwen/Qwen3.5-9B").split("/")[-1]
                self.skills = skills
                self.fact_index = combine.build_index(facts_of)
                self.stats["reloads"] += 1
            return True
        except Exception as e:  # noqa: BLE001 - fail-open: keep what was loaded
            sys.stderr.write("front door: manifest not loaded (%s)\n" % str(e)[:200])
            if self.base is None:
                raise
            return False

    def choose(self, body):
        """-> (model name to send the request to, night or None). Never raises."""
        self.stats["requests"] += 1
        try:
            self.reload()
            with self.lock:
                router, model_of, base, skills = self.router, self.model_of, self.base, self.skills
            text = last_user_text(body)
            night = router.route(text)
            if night is None:                          # no learnt fact asked about: a learnt skill, if one fits
                night = skills.route(text)
            if night is not None and night in model_of:
                self.stats["to_night"] += 1
                return model_of[night], night
        except Exception:  # noqa: BLE001
            self.stats["router_errors"] += 1
            base = self.base
        self.stats["to_base"] += 1
        return base, None

    # ---- what may be learnt ---------------------------------------------------------------------------------
    def settings(self):
        s = json.loads(json.dumps(SETTINGS_DEFAULT))
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                got = json.load(f)
            s["auto_learn"] = bool(got.get("auto_learn", s["auto_learn"]))
            s["auto_when"].update({k: v for k, v in (got.get("auto_when") or {}).items() if k in s["auto_when"]})
            s["learn"].update({k: bool(v) for k, v in (got.get("learn") or {}).items() if k in KINDS})
            s["self_tools"] = bool(got.get("self_tools", s["self_tools"]))
            s["combine"] = bool(got.get("combine", s["combine"]))
            if got.get("job_approve") in ("learn", "unsure", "never", "always"):
                s["job_approve"] = got["job_approve"]
        except (OSError, ValueError, AttributeError):
            pass
        return s

    def set_settings(self, change):
        """merge a partial change, save it, move notes and inbox items to match. -> the settings now in force"""
        s = self.settings()
        if not isinstance(change, dict):
            return {"error": "send a JSON object"}
        if "auto_learn" in change:
            s["auto_learn"] = bool(change["auto_learn"])
        for k, v in (change.get("auto_when") or {}).items():
            if k == "waiting_at_least" and isinstance(v, int) and v >= 1:
                s["auto_when"][k] = v
            if k == "hour" and (v is None or (isinstance(v, int) and 0 <= v <= 23)):
                s["auto_when"][k] = v
        if "combine" in change:
            s["combine"] = bool(change["combine"])
        if "self_tools" in change:
            s["self_tools"] = bool(change["self_tools"])
        if change.get("job_approve") in ("learn", "unsure", "never", "always"):
            s["job_approve"] = change["job_approve"]
        unknown = [k for k in (change.get("learn") or {}) if k not in KINDS]
        if unknown:
            return {"error": "unknown kinds: %s (known: %s)" % (", ".join(unknown), ", ".join(KINDS))}
        s["learn"].update({k: bool(v) for k, v in (change.get("learn") or {}).items()})
        tmp = self.settings_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, self.settings_path)
        self.sort_waiting(s)
        return s

    def _items(self, folder):
        try:
            return sorted(f for f in os.listdir(folder) if f.endswith(".json"))
        except OSError:
            return []

    def sort_waiting(self, s=None):
        """inbox items of a kind that may not be learnt become notes; notes of a kind that may be learnt go back to
        the inbox for the next run"""
        s = s or self.settings()
        moved = {"to_notes": 0, "to_inbox": 0}
        for src, dst, want_on, key in ((self.inbox, self.notes_dir, False, "to_notes"), (self.notes_dir, self.inbox, True, "to_inbox")):
            for f in self._items(src):
                try:
                    with open(os.path.join(src, f), encoding="utf-8") as h:
                        item = json.load(h)
                    kind = item.get("switch") or item.get("kind") or "personal"
                except (OSError, ValueError):
                    continue
                if bool(s["learn"].get(kind, True)) == want_on:
                    if want_on and item.get("kind") == "topics":
                        # a topic kept as a note while topics were off: its articles are fetched now, through the
                        # relay; if that fails it stays a note
                        try:
                            got = self.fetch_topic(item.get("text") or "")
                        except Exception:  # noqa: BLE001
                            got = None
                        if got:
                            os.remove(os.path.join(src, f))
                            moved[key] += 1
                        continue
                    os.makedirs(dst, exist_ok=True)
                    os.replace(os.path.join(src, f), os.path.join(dst, f))
                    moved[key] += 1
        return moved

    def auto_tick(self, now=None):
        """called once a minute by serve(): start a learning run when auto learning is on and its condition holds.
        -> the start result, or None"""
        s = self.settings()
        if not s["auto_learn"] or self.learning["running"] or not self.learn_cmd:
            return None
        now = now or time.localtime()
        hour = s["auto_when"].get("hour")
        if hour is not None:
            day = time.strftime("%Y-%m-%d", now)
            if now.tm_hour == hour and self.auto_day != day and self.waiting_taught() > 0:
                self.auto_day = day
                return self.learn()
            return None
        if self.waiting_taught() >= s["auto_when"]["waiting_at_least"]:
            return self.learn()
        return None

    def waiting_taught(self):
        """waiting items the person taught. Fintan's own manual (taught at first start) does not count towards starting
        a run by itself: on one GPU that run stopped answers for about 10 minutes straight after the first start
        (25 Nov). It is learnt with the person's first run."""
        n = 0
        for f in self._items(self.inbox):
            try:
                with open(os.path.join(self.inbox, f), encoding="utf-8") as h:
                    n += json.load(h).get("source") != MANUAL_SOURCE
            except (OSError, ValueError):
                n += 1
        return n

    # ---- learning while serving ----------------------------------------------------------------------------
    def waiting(self):
        return len(self._items(self.inbox))

    def kept_as_notes(self):
        return len(self._items(self.notes_dir))

    def teach(self, text, source="", kind="personal"):
        """the safety check, then the inbox (or the notes, when this kind may not be learnt).
        -> {"accepted", "reason", "flags", "waiting", "kept_as_note"}"""
        kind = kind or "personal"
        if kind not in KINDS:
            return {"accepted": False, "reason": "unknown kind %r (known: %s)" % (kind, ", ".join(KINDS)), "flags": [],
                    "waiting": self.waiting(), "kept_as_note": False}
        text = (text or "").strip()
        if kind == "code":                             # a code item is the path of a folder on this machine
            if not os.path.isdir(text):
                return {"accepted": False, "reason": "code: give the path of a folder on this machine", "flags": [],
                        "waiting": self.waiting()}
        elif len(text) < (3 if kind == "topics" else 20):   # a topic can be just its name ("Black holes")
            return {"accepted": False, "reason": "too short to learn from", "flags": [], "waiting": self.waiting()}
        if kind == "topics" and self.settings()["learn"].get("topics", True):
            chk = teach_policy.check_topic(text)
            if not chk.get("allow", True):
                self.usage.add("refused")
                return {"accepted": False, "reason": chk.get("reason") or "not a topic Fintan will learn", "flags": [],
                        "waiting": self.waiting()}
            try:
                n = self.fetch_topic(text)
            except Exception as e:  # noqa: BLE001
                return {"accepted": False, "reason": "could not fetch the topic: %s" % str(e)[:160], "flags": [],
                        "waiting": self.waiting()}
            if not n:
                return {"accepted": False, "reason": "no articles found for this topic", "flags": [], "waiting": self.waiting()}
            self.usage.add("taught")
            return {"accepted": True, "reason": "", "flags": [], "waiting": self.waiting(), "kept_as_note": False,
                    "articles": n}
        v = {"allow": True, "flags": []} if kind == "code" else teach_policy.check_document(text, name=source or "")
        self.usage.add("taught" if v["allow"] else "refused")
        if not v["allow"]:
            return {"accepted": False, "reason": v["reason"], "flags": v.get("flags") or [], "waiting": self.waiting()}
        if "secrets" in (v.get("flags") or []):
            text = teach_policy.redact_secrets(text)
        as_note = not self.settings()["learn"].get(kind, True)
        folder = self.notes_dir if as_note else self.inbox
        os.makedirs(folder, exist_ok=True)
        with self.lock:
            self.taught += 1
            name = "%d_%04d.json" % (int(time.time() * 1000), self.taught % 10000)
            with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                json.dump({"text": text, "source": source or "", "kind": kind, "t": int(time.time())}, f)
        return {"accepted": True, "reason": "", "flags": v.get("flags") or [], "waiting": self.waiting(), "kept_as_note": as_note}

    def topic_articles_waiting(self):
        for f in self._items(self.inbox):
            try:
                with open(os.path.join(self.inbox, f), encoding="utf-8") as h:
                    if json.load(h).get("switch") == "topics":
                        return True
            except (OSError, ValueError):
                pass
        return False

    def fetch_topic(self, name):
        """the articles of a topic, through our relay only (net.py: one address, the topic sent among decoys so the
        relay cannot tell which one was wanted, Wikipedia never sees this machine, every request written to calls.log
        beside the manifest first). Each article is queued as a document that follows the topics switch.
        -> how many were queued"""
        arts = net.topic(name, home=self.root, opener=self.net_opener)
        n = 0
        os.makedirs(self.inbox, exist_ok=True)
        for a in arts or []:
            text = ("%s. %s" % (a.get("title") or name, a.get("text") or "")).strip()
            if len(text) < 40:
                continue
            v = teach_policy.check_document(text, name=a.get("title") or name)
            if not v.get("allow", True):
                continue
            if "secrets" in (v.get("flags") or []):
                text = teach_policy.redact_secrets(text)
            with self.lock:
                self.taught += 1
                fn = "%d_%04d.json" % (int(time.time() * 1000), self.taught % 10000)
                with open(os.path.join(self.inbox, fn), "w", encoding="utf-8") as f:
                    json.dump({"text": text, "source": "%s (%s)" % (a.get("title") or name, a.get("url") or ""),
                               "kind": "documents", "switch": "topics", "topic": name, "t": int(time.time())}, f)
            n += 1
        return n

    def notes_for(self, question):
        """taught texts waiting in the inbox, or kept as notes because their kind may not be learnt, that share enough
        content words with the question. Never raises."""
        if not self.use_notes:
            return []
        names = [os.path.join(self.inbox, f) for f in self._items(self.inbox)] + \
                [os.path.join(self.notes_dir, f) for f in self._items(self.notes_dir)]
        try:
            if tuple(names) != self._notes_key:          # the inbox or the notes changed: read them again
                notes = []
                for f in names:
                    with open(f, encoding="utf-8") as h:
                        t = json.load(h).get("text") or ""
                    notes.append((content_words(t), t))
                with self.lock:
                    self._notes, self._notes_key = notes, tuple(names)
            qw = content_words(question)
            hits = sorted(((len(qw & w), i) for i, (w, t) in enumerate(self._notes) if len(qw & w) >= NOTE_MIN_WORDS), reverse=True)
            return [self._notes[i][1][:NOTE_CHARS] for n, i in hits[:NOTE_MAX]]
        except Exception:  # noqa: BLE001 - the stopgap never blocks an answer
            return []

    def load_adapters(self):
        """ask the engine to load adapters the manifest names with a path and that it was not yet asked for"""
        out = []
        if not self.engine:
            return out
        for model, path in sorted(self.paths.items()):
            if model in self.loaded:
                continue
            req = urllib.request.Request(self.engine.rstrip("/") + "/v1/load_lora_adapter", method="POST",
                                         data=json.dumps({"lora_name": model, "lora_path": path}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                self.opener(req, timeout=120).read()
                self.loaded.add(model)
                out.append(model)
            except Exception as e:  # noqa: BLE001 - not loaded: its night stays on the base model
                sys.stderr.write("front door: adapter %s not loaded (%s)\n" % (model, str(e)[:120]))
        return out

    def learn(self, wait=False):
        """start the learning run (one at a time). -> {"started", "reason"}"""
        if not self.learn_cmd:
            return {"started": False, "reason": "no learning command is set (--learn-cmd)"}
        with self.lock:
            if self.learning["running"]:
                return {"started": False, "reason": "a learning run is already going"}
            self.learning["running"] = True
        s = self.settings()
        self.sort_waiting(s)                           # a kind switched off since it was taught: a note, not learnt

        def run():
            t0, rc, err = time.time(), None, ""
            slept = self.engine_call("/sleep?level=1") if self.sleep_engine else False
            try:
                env = dict(os.environ, FINTAN_INBOX=self.inbox, FINTAN_MANIFEST=self.path,
                           FINTAN_LEARN_KINDS=",".join(k for k in KINDS if s["learn"][k] or (
                               # fetched topic articles are documents; with documents off they still follow topics
                               # (sort_waiting has already moved every other document to the notes)
                               k == "documents" and s["learn"].get("topics") and self.topic_articles_waiting())))
                r = subprocess.run(self.learn_cmd, shell=True, env=env, capture_output=True, text=True)
                rc, err = r.returncode, (r.stderr or "")[-300:]
                if rc == 0:
                    self.reload()
            except Exception as e:  # noqa: BLE001
                rc, err = -1, str(e)[:300]
            if slept:
                self.engine_call("/wake_up")
            loaded = self.load_adapters() if rc == 0 else []
            self.usage.add("learn_runs" if rc == 0 else "learn_failed")
            self.usage.share(self.base)
            with self.lock:
                self.learning.update(running=False, runs=self.learning["runs"] + 1,
                                     last={"ok": rc == 0, "code": rc, "seconds": round(time.time() - t0, 1),
                                           "error": err if rc else "", "adapters_loaded": loaded})
        th = threading.Thread(target=run, daemon=True)
        th.start()
        if wait:
            th.join()
        return {"started": True, "reason": ""}

    # ---- the job loop's queue (jobs/job_loop.py run takes tasks from here) ---------------------------------------
    def job_add(self, request):
        request = (request or "").strip()
        if len(request) < 3:
            return {"queued": False, "reason": "no task given"}
        if not os.path.exists(os.path.join(self.jobs_dir, "workplace.json")):
            return {"queued": False, "reason": "no workplace is set up (jobs/workplace.json): the job loop has no tools"}
        d = os.path.join(self.jobs_dir, "tasks")
        os.makedirs(d, exist_ok=True)
        tid = "t%d" % int(time.time() * 1000)
        with open(os.path.join(d, tid + ".json"), "w", encoding="utf-8") as f:
            json.dump({"id": tid, "text": request, "t": int(time.time())}, f)
        return {"queued": True, "task": tid}

    def jobs_status(self):
        def load(folder):
            out = []
            for f in self._items(os.path.join(self.jobs_dir, folder)):
                try:
                    with open(os.path.join(self.jobs_dir, folder, f), encoding="utf-8") as h:
                        out.append(json.load(h))
                except (OSError, ValueError):
                    pass
            return out
        done = []
        try:
            with open(os.path.join(self.jobs_dir, "done.jsonl"), encoding="utf-8") as h:
                done = [json.loads(x) for x in h.readlines()[-20:]]
        except (OSError, ValueError):
            pass
        return {"queued": load("tasks"), "questions": load("questions"), "done": done}

    def job_answer(self, task, answer):
        if not task or not (answer or "").strip():
            return {"ok": False, "reason": "give the task and an answer"}
        d = os.path.join(self.jobs_dir, "answers")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, re.sub(r"[^\w-]", "", str(task)) + ".json"), "w", encoding="utf-8") as f:
            json.dump({"task": task, "answer": answer.strip(), "t": int(time.time())}, f)
        return {"ok": True}

    def teach_manual(self):
        """on first start, Fintan's own manual (self/manual.json) is taught once, so the first learning run puts what
        the model is and how it is used into its weights"""
        mark = os.path.join(self.root, "self", ".taught")
        src = os.path.join(self.root, "self", "manual.json")
        if os.path.exists(mark) or not os.path.exists(src):
            return 0
        with open(src, encoding="utf-8") as f:
            items = json.load(f)
        n = sum(1 for it in items if self.teach(it["text"], MANUAL_SOURCE, it.get("kind", "documents")).get("accepted"))
        with open(mark, "w") as f:
            f.write(str(n))
        return n

    def engine_call(self, path):
        """lets one GPU both serve and learn: vLLM started with --enable-sleep-mode (VLLM_SERVER_DEV_MODE=1) frees its
        GPU RAM on /sleep and takes it back on /wake_up. True when the engine said yes; never raises."""
        if not self.engine:
            return False
        try:
            req = urllib.request.Request(self.engine.rstrip("/") + path, method="POST", data=b"")
            self.opener(req, timeout=120).read()
            return True
        except Exception as e:  # noqa: BLE001 - fail-open: learning still runs, it may then be short of GPU RAM
            sys.stderr.write("front door: engine %s failed (%s)\n" % (path, str(e)[:120]))
            return False

    def models(self):
        with self.lock:
            return [self.base] + [self.model_of[n] for n in sorted(self.model_of)]


def make_handler(door, engine, opener=None, timeout=600):
    opener = opener or urllib.request.urlopen

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj, extra=None):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _allowed(self):
            """FINTAN_API_KEY set: every request must carry it (Authorization: Bearer <key>, or X-Api-Key)"""
            key = os.environ.get("FINTAN_API_KEY")
            if not key:
                return True
            got = self.headers.get("Authorization") or ""
            if got == "Bearer " + key or self.headers.get("X-Api-Key") == key:
                return True
            self._json(401, {"error": {"message": "this Fintan needs its API key (Authorization: Bearer <key>)",
                                       "type": "unauthorized"}})
            return False

        def do_GET(self):
            if self.path.rstrip("/") == "/ping":
                # health check for hosts (RunPod load-balanced endpoints): 200 ready, 204 the engine is still starting;
                # no key needed and nothing about the model is shown
                try:
                    opener(urllib.request.Request(engine.rstrip("/") + "/v1/models"), timeout=5).read()
                    ready = True
                except Exception:  # noqa: BLE001
                    ready = bool(door.sleep_engine and door.learning["running"])
                if ready:
                    return self._json(200, {"status": "ok"})
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self._allowed():
                return
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": door.base, "object": "model", "owned_by": "fintan"}]})
            if self.path.rstrip("/") == "/fintan/usage":
                return self._json(200, {"shared_with": door.usage.share_url or "nobody", "counts": door.usage.share_payload(door.base)})
            if self.path.rstrip("/") == "/fintan/status":
                return self._json(200, {"base": door.base, "nights": door.models()[1:], "stats": door.stats,
                                        "inbox_waiting": door.waiting(), "kept_as_notes": door.kept_as_notes(),
                                        "learning": door.learning})
            if self.path.rstrip("/") == "/fintan/settings":
                return self._json(200, door.settings())
            if self.path.rstrip("/") == "/fintan/jobs":
                return self._json(200, door.jobs_status())
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not self._allowed():
                self.rfile.read(n) if n else None
                return
            raw = self.rfile.read(n) if n else b"{}"
            night, notes = None, []
            if self.path.rstrip("/") in ("/fintan/teach", "/fintan/learn", "/fintan/settings", "/fintan/job", "/fintan/job/answer"):
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except Exception:  # noqa: BLE001
                    return self._json(400, {"error": {"message": "the body must be JSON"}})
                if self.path.rstrip("/") == "/fintan/teach":
                    return self._json(200, door.teach(body.get("text"), body.get("source") or "", body.get("kind") or "personal"))
                if self.path.rstrip("/") == "/fintan/settings":
                    s = door.set_settings(body)
                    return self._json(400 if "error" in s else 200, s)
                if self.path.rstrip("/") == "/fintan/job":
                    return self._json(200, door.job_add(body.get("request")))
                if self.path.rstrip("/") == "/fintan/job/answer":
                    return self._json(200, door.job_answer(body.get("task"), body.get("answer")))
                return self._json(200, door.learn())
            if self.path.rstrip("/") == "/v1/chat/completions":
                if door.sleep_engine and door.learning["running"]:
                    return self._json(503, {"error": {"message": "Fintan is learning on this GPU; answers resume when the "
                                                                 "run ends (GET /fintan/status shows it)", "type": "learning"}})
                selfuse = combo = False
                try:
                    body = json.loads(raw.decode("utf-8"))
                    own = body.pop("fintan", None) or {}           # the door's own options, never sent to the engine
                    body["model"], night = door.choose(body)
                    if body.get("tools") and night is not None:
                        # an agent's request with tools goes to the plain model: sent to a learnt update, tool
                        # calls came back right 1 time in 10 (25 Nov); the plain model got 11 of 12
                        body["model"], night = door.base, None
                        door.stats["tools_to_base"] = door.stats.get("tools_to_base", 0) + 1
                    combo = (own.get("combine", True) and not body.get("stream") and not body.get("tools")
                             and door.settings().get("combine", True)
                             and combine.wanted(last_user_text(body), door.fact_index))
                    selfuse = (own.get("self_tools", True) and not body.get("stream") and not body.get("tools")
                               and door.settings().get("self_tools", True)
                               and self_tools.wanted(last_user_text(body), door.root))
                    notes = door.notes_for(last_user_text(body)) if night is None else []
                    door.usage.add("to_night" if night is not None else "to_base", save=False)
                    if notes:
                        door.usage.add("with_notes", save=False)
                        door.stats["with_notes"] = door.stats.get("with_notes", 0) + 1
                        body["messages"] = [{"role": "system", "content": NOTE_HEAD + "\n\n".join("- " + n for n in notes)}] + list(body["messages"])
                    raw = json.dumps(body).encode("utf-8")
                except Exception:  # noqa: BLE001 - not JSON: the engine answers it
                    pass
                if selfuse:
                    # the model may call its own functions (teach, learn, status, settings, job); the door runs them
                    def post(b):
                        rq = urllib.request.Request(engine.rstrip("/") + "/v1/chat/completions", method="POST",
                                                    data=json.dumps(b).encode("utf-8"), headers={"Content-Type": "application/json"})
                        return json.loads(opener(rq, timeout=timeout).read())
                    try:
                        out, actions = self_tools.converse(door, body, post)
                    except Exception as e:  # noqa: BLE001
                        return self._json(502, {"error": {"message": "engine: %s" % str(e)[:300]}})
                    out["fintan_actions"] = actions
                    door.stats["self_actions"] = door.stats.get("self_actions", 0) + len(actions)
                    return self._json(200, out)
                if combo:
                    # a question that needs several learnt facts: answered in steps (combine.py); if no answer is
                    # reached, it is answered the usual way below
                    def post_c(b):
                        rq = urllib.request.Request(engine.rstrip("/") + "/v1/chat/completions", method="POST",
                                                    data=json.dumps(b).encode("utf-8"), headers={"Content-Type": "application/json"})
                        return json.loads(opener(rq, timeout=timeout).read())
                    try:
                        with door.lock:
                            idx, model_of, base = door.fact_index, dict(door.model_of), door.base
                        got = combine.run(last_user_text(body), idx, model_of, base, post_c, body.get("max_tokens"))
                    except Exception:  # noqa: BLE001 - never blocks an answer
                        got = None
                    if got:
                        text, lookups, short = got
                        door.stats["combined"] = door.stats.get("combined", 0) + 1
                        return self._json(200, {"id": "fintan-combined", "object": "chat.completion", "model": base,
                                                "choices": [{"index": 0, "finish_reason": "stop",
                                                             "message": {"role": "assistant", "content": text}}],
                                                "fintan_lookups": [{"asked": q_, "got": g_} for q_, g_ in lookups],
                                                "fintan_answer": short})
            req = urllib.request.Request(engine.rstrip("/") + self.path, data=raw, method="POST",
                                         headers={"Content-Type": self.headers.get("Content-Type") or "application/json"})
            if self.headers.get("Authorization"):
                req.add_header("Authorization", self.headers["Authorization"])
            try:
                r = opener(req, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", 502)
                try:
                    msg = e.read().decode("utf-8", "replace")[:2000]
                except Exception:  # noqa: BLE001
                    msg = str(e)[:300]
                return self._json(code if isinstance(code, int) else 502, {"error": {"message": msg}})
            self.send_response(getattr(r, "status", 200))
            self.send_header("Content-Type", r.headers.get("Content-Type") or "application/json")
            self.send_header("X-Fintan-Night", str(night) if night is not None else "none")
            self.send_header("X-Fintan-Notes", str(len(notes)))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = r.read(4096)
                if not chunk:
                    break
                self.wfile.write(("%x\r\n" % len(chunk)).encode("ascii") + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
    return H


def serve(manifest, engine, port, host="127.0.0.1", inbox=None, learn_cmd=None, notes=True, share_usage=None,
          sleep_engine=False):
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("FINTAN_API_KEY") \
            and not os.environ.get("FINTAN_ALLOW_OPEN"):
        # reachable from other machines with no key would let anyone ask it, teach it and read its status
        raise SystemExit("front door: listening on %s needs FINTAN_API_KEY (or FINTAN_ALLOW_OPEN=1 on a network you "
                         "trust)" % host)
    door = Door(manifest, inbox=inbox, learn_cmd=learn_cmd, engine=engine)
    door.sleep_engine = bool(sleep_engine)
    try:
        door.teach_manual()                            # first start: Fintan learns its own manual at the first run
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("front door: manual not taught (%s)\n" % str(e)[:120])
    door.usage.share_url = share_usage or None
    door.use_notes = bool(notes)
    door.load_adapters()
    srv = ThreadingHTTPServer((host, port), make_handler(door, engine))

    def auto():                                        # auto learning: checked once a minute (off in settings = idle)
        while True:
            time.sleep(60)
            try:
                door.auto_tick()
            except Exception as e:  # noqa: BLE001
                sys.stderr.write("front door: auto learning check failed (%s)\n" % str(e)[:120])
    threading.Thread(target=auto, daemon=True).start()
    return door, srv


def main():
    ap = argparse.ArgumentParser(description="Fintan's front door: the router in front of any engine that serves adapters")
    ap.add_argument("manifest")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--engine", default="http://127.0.0.1:8000", help="the engine's OpenAI-style address (vLLM, Ollama, llama.cpp server)")
    ap.add_argument("--inbox", default=None, help="folder for taught material (default: inbox/ beside the manifest)")
    ap.add_argument("--learn-cmd", dest="learn_cmd", default=None,
                    help="the learning run; gets FINTAN_INBOX and FINTAN_MANIFEST in its environment")
    ap.add_argument("--sleep-engine", dest="sleep_engine", action="store_true",
                    help="one GPU: ask vLLM (--enable-sleep-mode) to free its GPU RAM while a learning run goes")
    ap.add_argument("--no-notes", dest="no_notes", action="store_true",
                    help="no stopgap: taught texts are not shown to the model while they wait to be learnt")
    ap.add_argument("--share-usage", dest="share_usage", default=None, metavar="URL",
                    help="OFF unless given: send the counts of /fintan/usage (no content, a random id) to this address once a day")
    a = ap.parse_args()
    door, srv = serve(a.manifest, a.engine, a.port, a.host, inbox=a.inbox, learn_cmd=a.learn_cmd, notes=not a.no_notes,
                      share_usage=a.share_usage, sleep_engine=a.sleep_engine)
    print("front door on http://%s:%d -> %s | base %s | nights %s" % (a.host, a.port, a.engine, door.base, door.models()[1:]), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
