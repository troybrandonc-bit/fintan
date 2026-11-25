"""fintan job: Fintan-9B doing tasks in your own systems, learning from what you tell it.

The worker (job2 in learn/engine/) runs on your tools in four steps:
  gather   code makes the lookups (records found by the ids in the request, then linked records)
  judge    one model call: which action, with which values, given the records, earlier cases and what you said
  send     the action goes to your system only after you approve it (--yes approves every action)
  after    you say whether it did what you asked, and what a colleague would tell it; that becomes a rule and a case
The model is asked through the front door (http://127.0.0.1:8900), so the learnt updates are used.

    python jobs/job_loop.py run [--once] [--poll 30]      ON ITS OWN: takes tasks (handed over through the front door,
            or from the workplace's task source), does them, learns from what the systems reply, turns those lessons
            into skills for the next learning run, and asks the person (GET /fintan/jobs, POST /fintan/job/answer)
            only when it needs to; that task waits, the others go on. Setting "job_approve": "learn" (default: asks
            before every action until you have said yes to that action 5 times in a row; a different answer is not
            sent, becomes a rule, and starts the count again), "always" (asks before every action), "unsure" (asks
            only when unsure), "never".
    python jobs/job_loop.py do "Refund order 1043, the parcel never arrived" [--workplace jobs/workplace.json]
            [--yes] [--outcome ok|wrong] [--note "what a colleague would say"] [--door URL] [--batch]
    python jobs/job_loop.py feedback --outcome ok|wrong [--note "..."]   say later how the last task went
    python jobs/job_loop.py rules              what it has learnt at this workplace
    python jobs/job_loop.py log                every call it made to your systems

jobs/workplace.json (see workplace.example.json):
  {"system": "who it works for and what the desk does",
   "tools": {"get_order": {"kind": "lookup", "args": {"order_id": "the order number"}, "http": "http://..."},
             "issue_refund": {"kind": "action", "args": {"order_id": "", "amount": ""}, "python": "tools.py:refund"}}}
A tool is called by HTTP (POST of the arguments as JSON; the reply's JSON is the record) or by a Python function in
a file beside workplace.json (it gets the arguments as a dict and returns a dict, or raises to refuse).
Every call is written to jobs/calls.log. Job notes taught through the door (kind "jobs") are read in as notices.
The learning switch for "jobs" off: what you say is kept in notes/, no rule is learnt.
"""
import argparse
import datetime
import glob
import importlib.util
import io
import json
import os
import pickle
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.environ.get("FINTAN_ENGINE") or os.path.join(ROOT, "learn", "engine"))
STATE = os.path.join(HERE, "state.pkl")
TRUST_AFTER = int(os.environ.get("FINTAN_TRUST_AFTER", "5"))   # yeses in a row before an action goes without asking
LOG = os.path.join(HERE, "calls.log")


def model_fn(door):
    """gen(prompts, max_new) and confirm(prompts) through the front door (OpenAI-style), thinking off"""
    def ask(p, max_new, sample=False):
        # the loop's own questions to the model: the door must not treat them as chat (25 Nov: with the model's own
        # functions on, Fintan-4B filed its decision "issue_refund 1043, 25.00" as a new job instead of returning it)
        body = {"model": "fintan", "messages": [{"role": "user", "content": p}], "max_tokens": int(max_new),
                "temperature": 0.7 if sample else 0.0, "chat_template_kwargs": {"enable_thinking": False},
                "fintan": {"self_tools": False, "combine": False}}
        req = urllib.request.Request(door.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        return (r["choices"][0]["message"].get("content") or "").strip()

    def gen(ps, max_new=140, sample=False):
        return [ask(p, max_new, sample) for p in ps]

    def confirm(ps):
        return [1.0 if o.lower().startswith("yes") else 0.0 for o in gen(ps, 3)]
    return gen, confirm


def tool_caller(wp_dir, tools, approve):
    """call(tool, args) -> "RESULT {json}" or "ERROR message", as the job loop expects; every call logged"""
    def call(tool, args):
        spec = tools.get(tool)
        if spec is None:
            reply = "ERROR unknown tool %s" % tool
        elif spec.get("kind") != "lookup" and not approve(tool, args):
            reply = "ERROR not sent: the person did not approve this action"
        else:
            try:
                if spec.get("http"):
                    req = urllib.request.Request(spec["http"], data=json.dumps(args).encode(), method="POST",
                                                 headers={"Content-Type": "application/json"})
                    got = json.loads(urllib.request.urlopen(req, timeout=60).read() or b"{}")
                elif spec.get("python"):
                    path, fn = spec["python"].rsplit(":", 1)
                    sp = importlib.util.spec_from_file_location("wp_tools", os.path.join(wp_dir, path))
                    mod = importlib.util.module_from_spec(sp)
                    sp.loader.exec_module(mod)
                    got = getattr(mod, fn)(dict(args))
                else:
                    raise ValueError("the tool has no http or python entry in workplace.json")
                reply = "RESULT " + json.dumps(got if got is not None else {"status": "done"}, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001 - the system refused: the worker reads the message
                reply = "ERROR " + str(e)[:300]
        with io.open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": tool, "args": args,
                                "reply": reply[:500]}, ensure_ascii=False) + "\n")
        return reply
    return call


def settings_jobs_on():
    try:
        return bool(json.load(io.open(os.path.join(ROOT, "settings.json"), encoding="utf-8"))["learn"].get("jobs", True))
    except (OSError, ValueError, KeyError):
        return True


def load_worker(wp, gen, confirm):
    import job2
    if os.path.exists(STATE):
        st = pickle.load(open(STATE, "rb"))
        w = st["worker"]
        w.gen, w.confirm = gen, confirm
        w.tools, w.system = wp["tools"], wp["system"]
        return w, st["first_day"]
    return job2.Worker(wp["system"], wp["tools"], gen, confirm), datetime.date.today().isoformat()


def save_worker(w, first_day):
    w.__dict__.pop("consult", None)
    gen, confirm, recall = w.gen, w.confirm, getattr(w, "recall", None)
    w.gen = w.confirm = w.recall = None
    tmp = STATE + ".tmp"
    pickle.dump({"worker": w, "first_day": first_day}, open(tmp, "wb"))
    os.replace(tmp, STATE)
    w.gen, w.confirm, w.recall = gen, confirm, recall


def to_notes(text):
    d = os.path.join(ROOT, "notes")
    os.makedirs(d, exist_ok=True)
    json.dump({"text": text, "source": "job loop", "kind": "jobs", "t": int(time.time())},
              io.open(os.path.join(d, "%d_job.json" % int(time.time() * 1000)), "w", encoding="utf-8"))


class Park(Exception):
    """the task waits for the person: a question is in jobs/questions/; nothing was sent"""


def jread(path, default=None):
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, io.open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def approve_level():
    s = jread(os.path.join(ROOT, "settings.json"), {}) or {}
    return s.get("job_approve") if s.get("job_approve") in ("learn", "unsure", "never", "always") else "learn"


def new_tasks(wp, wp_dir):
    """tasks handed over through the front door (jobs/tasks/) and, when the workplace has a task source, new ones from
    it ({"tasks": {"python": "tools.py:fn"}} or {"tasks": {"http": url}}, returning [{"id", "text"}, ...])"""
    out = [t for t in (jread(f) for f in sorted(glob.glob(os.path.join(HERE, "tasks", "*.json")))) if t and t.get("text")]
    src = wp.get("tasks") or {}
    if src:
        seen = set(jread(os.path.join(HERE, "seen.json"), []) or [])
        try:
            if src.get("python"):
                path, fn = src["python"].rsplit(":", 1)
                sp = importlib.util.spec_from_file_location("wp_tasks", os.path.join(wp_dir, path))
                mod = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(mod)
                got = getattr(mod, fn)() or []
            else:
                got = json.loads(urllib.request.urlopen(src["http"], timeout=60).read() or b"[]")
        except Exception as e:  # noqa: BLE001
            print("  task source not reachable: %s" % str(e)[:120], flush=True)
            got = []
        for t in got:
            tid = "w" + str(t.get("id"))
            if tid not in seen and t.get("text"):
                seen.add(tid)
                jwrite(os.path.join(HERE, "tasks", tid + ".json"), {"id": tid, "text": t["text"], "t": int(time.time())})
                out.append({"id": tid, "text": t["text"]})
        jwrite(os.path.join(HERE, "seen.json"), sorted(seen))
    return out


def promote(door, sentences):
    """what the systems or the person said about how work is done also becomes a skill, learnt into the weights at
    the next learning run (once per sentence)"""
    done = set(jread(os.path.join(HERE, "promoted.json"), []) or [])
    for s_ in sentences:
        s_ = (s_ or "").strip()
        if len(s_) < 20 or s_ in done:
            continue
        try:
            req = urllib.request.Request(door.rstrip("/") + "/fintan/teach", method="POST",
                                         data=json.dumps({"text": s_, "source": "job loop", "kind": "skills"}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            done.add(s_)
        except Exception as e:  # noqa: BLE001 - the door is down: try again after the next task
            print("  could not hand the lesson to learning: %s" % str(e)[:100], flush=True)
    jwrite(os.path.join(HERE, "promoted.json"), sorted(done))


def do_task(a, wp, w, first_day, learn, task):
    """one task with nobody at the keyboard. Acts by itself; asks the person (and waits) only where the job loop
    would ask a colleague, or before every action when the setting says "always". Learns from what the systems
    reply: a refusal's own words are the lesson."""
    tid, text = task["id"], task["text"]
    day = (datetime.date.today() - datetime.date.fromisoformat(first_day)).days + 1
    level = approve_level()
    qfile = os.path.join(HERE, "questions", tid + ".json")
    afile = os.path.join(HERE, "answers", tid + ".json")
    cfile = os.path.join(HERE, "answers", tid + ".colleague.json")
    q, ans = jread(qfile, {}) or {}, (jread(afile, {}) or {}).get("answer")
    approval, pending = None, None
    if q and ans:
        if q.get("kind") == "approve":
            approval, pending = ans.strip(), q.get("call")
        else:                                        # an answer to "how should this be handled": kept for this task
            jwrite(cfile, {"task": tid, "answer": ans})
        for p_ in (qfile, afile):
            os.remove(p_)
    yes = lambda s_: bool(re.match(r"(?i)\s*(yes|y|ok|okay|go|go ahead|send it|approved?)\b", s_ or ""))  # noqa: E731
    trust = jread(os.path.join(HERE, "trust.json"), {}) or {}
    if approval and not yes(approval) and pending:
        # the person said what to do instead: nothing is sent; that is a rule now and a skill at the next learning
        # run, the action's trust starts again, and the task is worked out again with it in front
        if learn:
            for r in w.after(text, [], False, "The task was not finished.", approval, day) or []:
                print("  learnt: " + __import__("call_rules").line(r), flush=True)
        promote(a.door, [approval])
        trust[pending[0]] = 0
        jwrite(os.path.join(HERE, "trust.json"), trust)
        jwrite(cfile, {"task": tid, "answer": approval})
        approval = None
    col_ans = (jread(cfile, {}) or {}).get("answer")
    why = {}
    approved = []

    def same_call(c_, tool, args):
        return bool(c_) and c_[0] == tool and json.dumps(c_[1], sort_keys=True) == json.dumps(args, sort_keys=True)

    def approve(tool, args):
        # "learn" (the default): ask before every action until the person has said yes to this action TRUST_AFTER
        # times in a row; "always": every time; "unsure" / "never": no asking here
        if level in ("unsure", "never"):
            return True
        if level == "learn" and trust.get(tool, 0) >= TRUST_AFTER:
            return True
        if approval and yes(approval) and same_call(pending, tool, args):
            approved.append(tool)
            return True
        jwrite(qfile, {"task": tid, "request": text, "kind": "approve", "call": [tool, args], "t": int(time.time()),
                       "question": "May I send %s %s? Answer yes, or say what to do instead." % (
                           tool, json.dumps(args, ensure_ascii=False))})
        raise Park()

    def colleague():
        if level == "never":
            return ""
        if col_ans:
            return col_ans
        refused = [c[2][len("ERROR "):].strip() for c in calls if c[2].startswith("ERROR ") and "not sent:" not in c[2]]
        if refused and not why.get("why"):
            why["why"] = "refused"
        promote(a.door, refused)                  # what the system said is a lesson even while the task waits
        jwrite(qfile, {"task": tid, "request": text, "kind": "colleague", "why": why.get("why"), "t": int(time.time()),
                       "system_said": refused[-1] if refused else None,
                       "question": "How should this one be handled? (%s)" % {
                           "first": "the first time I would use this kind of action here",
                           "probation": "I have done this kind only once", "split": "my working gave different answers",
                           "conflict": "my choice disagrees with earlier cases", "unsure": "I am not sure of my call",
                           "refused": "the system refused my call twice; it said: %s" % (refused[-1] if refused else "")}.get(
                               why.get("why"), "I am not sure")})
        raise Park()

    orig = w.consult

    def consult(ask, why_, base, extra=""):
        why["why"] = why_
        return orig(ask, why_, base, extra)
    w.consult = consult
    calls = []
    caller = tool_caller(os.path.dirname(os.path.abspath(a.workplace)), wp["tools"], approve)

    def call(tool, args):
        reply = caller(tool, args)
        calls.append((tool, args, reply))
        return reply
    try:
        w.work(call, calls, text, day, ask=colleague)
    except Park:
        w.__dict__.pop("consult", None)
        print("  %s waits for the person: %s" % (tid, (jread(qfile, {}) or {}).get("question")), flush=True)
        return "asked"
    w.__dict__.pop("consult", None)
    acts = [c for c in calls if (wp["tools"].get(c[0]) or {}).get("kind") != "lookup"]
    ok = bool(acts) and acts[-1][2].startswith("RESULT ")
    said = [c[2][len("ERROR "):].strip() for c in acts if c[2].startswith("ERROR ") and "not sent:" not in c[2]]
    head = "Done." if ok else "The task was not finished."
    learnt = []
    if learn:
        learnt = [__import__("call_rules").line(r) for r in w.after(text, calls, ok, head, " ".join(said), day)]
    elif said:
        to_notes(" ".join(said))
    promote(a.door, said + ([col_ans] if col_ans else []))
    if ok:
        for t_ in approved:                          # a yes that went through: one more step towards trust
            trust[t_] = trust.get(t_, 0) + 1
        jwrite(os.path.join(HERE, "trust.json"), trust)
    with io.open(os.path.join(HERE, "done.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"task": tid, "request": text, "done": ok, "asked": why.get("why"), "approved_by_person": approved,
                            "learnt": learnt, "calls": [[c[0], c[1], c[2][:200]] for c in calls], "t": int(time.time())},
                           ensure_ascii=False) + "\n")
    for p_ in (os.path.join(HERE, "tasks", tid + ".json"), qfile, afile, cfile):
        if os.path.exists(p_):
            os.remove(p_)
    print("  %s %s | %s%s" % (tid, "DONE" if ok else "NOT DONE", text[:60], "".join("\n     + " + x for x in learnt)), flush=True)
    return "done" if ok else "not done"


def read_notices(w, day, learn):
    """job notes taught through the door (kind "jobs") are read in as notices from the company"""
    if not learn:
        return
    for f in sorted(glob.glob(os.path.join(ROOT, "inbox", "*.json"))):
        it = jread(f, {}) or {}
        if it.get("kind") == "jobs" and it.get("text"):
            n = len(w.notice(it["text"], day) or [])
            print("  read a job note (%d rules)" % n, flush=True)
            os.remove(f)


def run_forever(a, wp, w, first_day, learn):
    """the job loop on its own: takes new tasks, does them, learns, and waits for answers only where it asked"""
    wp_dir = os.path.dirname(os.path.abspath(a.workplace))
    while True:
        read_notices(w, (datetime.date.today() - datetime.date.fromisoformat(first_day)).days + 1, learn)
        for t in new_tasks(wp, wp_dir):
            waiting = os.path.exists(os.path.join(HERE, "questions", t["id"] + ".json"))
            answered = os.path.exists(os.path.join(HERE, "answers", t["id"] + ".json"))
            if waiting and not answered:
                continue
            try:
                do_task(a, wp, w, first_day, learn, t)
            except Exception as e:  # noqa: BLE001 - one bad task never stops the loop
                print("  %s failed: %s" % (t["id"], str(e)[:200]), flush=True)
            save_worker(w, first_day)
        if a.once:
            return
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["do", "run", "feedback", "rules", "log"])
    ap.add_argument("--once", action="store_true", help="run: take the tasks there are now, then stop")
    ap.add_argument("--poll", type=float, default=30.0, help="run: seconds between looks for new tasks")
    ap.add_argument("request", nargs="?")
    ap.add_argument("--workplace", default=os.path.join(HERE, "workplace.json"))
    ap.add_argument("--door", default="http://127.0.0.1:8900")
    ap.add_argument("--yes", action="store_true", help="approve every action without asking")
    ap.add_argument("--outcome", choices=["ok", "wrong"], default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--batch", action="store_true", help="nobody at the keyboard: no questions; actions only with --yes")
    a = ap.parse_args()
    if a.cmd == "log":
        print(io.open(LOG, encoding="utf-8").read() if os.path.exists(LOG) else "no calls yet")
        return
    wp = json.load(io.open(a.workplace, encoding="utf-8"))
    gen, confirm = model_fn(a.door)
    w, first_day = load_worker(wp, gen, confirm)
    if a.cmd == "rules":
        print("\n".join(w.rb.lines()) or "no rules learnt yet")
        return
    learn = settings_jobs_on()

    def learn_from(text, calls, sent, outcome, note, day):
        ok = outcome in ("y", "ok")
        head = "Done." if ok else ("The action went through but did not do what the task asked." if sent else "The task was not finished.")
        if learn:
            for r in w.after(text, calls, ok, head, note or "", day):
                print("  learnt: " + __import__("call_rules").line(r))
        elif note:
            to_notes(note)
            print("  job learning is off: the note is kept in notes/")
        w.pending = None
    if a.cmd == "feedback":
        p_ = getattr(w, "pending", None)
        if not p_ or a.outcome is None:
            ap.error("feedback needs --outcome and a task done with 'do' and no outcome yet")
        learn_from(p_["text"], p_["calls"], p_["sent"], a.outcome, a.note, p_["day"])
        save_worker(w, first_day)
        return
    if a.cmd == "run":
        return run_forever(a, wp, w, first_day, learn)
    if not a.request:
        ap.error("do needs a request")
    day = (datetime.date.today() - datetime.date.fromisoformat(first_day)).days + 1
    interactive = sys.stdin.isatty() and not a.batch        # (Windows reports NUL as a terminal: use --batch)
    read_notices(w, day, learn)

    def approve(tool, args):
        if a.yes:
            return True
        if not interactive:
            return False
        return input("It wants to send: %s %s\nSend it? [y/N] " % (tool, json.dumps(args, ensure_ascii=False))).strip().lower() == "y"

    def colleague():
        if not interactive:
            return ""
        return input("It asks a colleague how to handle this one. Your answer (Enter: no answer): ").strip()

    calls = []
    caller = tool_caller(os.path.dirname(os.path.abspath(a.workplace)), wp["tools"], approve)

    def call(tool, args):
        reply = caller(tool, args)
        calls.append((tool, args, reply))
        return reply
    w.work(call, calls, a.request, day, ask=colleague)
    acts = [c for c in calls if (wp["tools"].get(c[0]) or {}).get("kind") != "lookup"]
    sent = [c for c in acts if c[2].startswith("RESULT ")]
    for c in calls:
        print("  %s %s -> %s" % (c[0], json.dumps(c[1], ensure_ascii=False), c[2][:160]))
    if w.why:
        print("  (it asked first: %s)" % w.why)
    outcome = a.outcome or (input("Did it do what you asked? [y/n] ").strip().lower()[:1] if interactive else "")
    note = a.note if a.note is not None else (input("What would a colleague tell it for next time? (Enter: nothing) ").strip()
                                              if interactive else "")
    if outcome not in ("y", "ok", "n", "wrong"):
        w.pending = {"text": a.request, "calls": calls, "sent": bool(sent), "day": day}
        save_worker(w, first_day)
        print("no outcome yet: tell it later with 'feedback --outcome ok|wrong [--note ...]'")
        return
    learn_from(a.request, calls, sent, outcome, note, day)
    save_worker(w, first_day)


if __name__ == "__main__":
    main()
