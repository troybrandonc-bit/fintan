"""The model's own functions: from chat, Fintan can teach itself, start learning, change its settings and hand work
to the job loop.

The person only chats. When a request is about Fintan itself (who it is, teaching, learning, its settings) or about
work at the configured workplace, the front door gives the model its own functions, and the model calls them by
itself: it teaches itself what it is told to remember, starts or schedules learning, changes its switches, reports
what it knows, and hands work to the job loop. The door runs each call on this machine and gives the model the
result; the model then answers the person. Every call is listed in the reply ("fintan_actions") and logged.

Ordinary requests get nothing added: their prompt is exactly what the person sent, so measured results still apply.
Any engine works: the functions are described in the system text in the Qwen tool format and the door reads the
model's <tool_call> blocks itself (JSON, or the <function=...><parameter=...> form).
"""
import json
import os
import re

SELF = ("You are {name}, an open model made by Fintan Labs, built on {base}. Unlike other models you keep "
        "learning after release: what people teach you is learnt into your own weights on their machine, as updates "
        "that are kept only when a check shows they make you better. You can learn documents, topics, facts about the "
        "person, news, a codebase and skills, and you can do tasks in the person's systems through the job loop. You "
        "run on the person's own computer; nothing they teach you leaves it.\n"
        "Use your functions yourself, without asking the person to do it: when they ask you to remember or learn "
        "something, or tell you lasting facts about themselves or their work, call fintan_teach; when they ask what you "
        "know or how learning is going, call fintan_status; when they want learning now, call fintan_learn; to change "
        "when or what you learn, call fintan_settings; when they ask you to do a task in their systems, call "
        "fintan_job. Then tell them in one or two plain sentences what you did; after fintan_teach, say it will be learnt "
        "at your next learning run (it is not learnt yet). Do not call a function for an ordinary question.")

TOOLS = [
    {"name": "fintan_teach", "description": "Learn something: it is checked, then learnt into your weights at the next "
     "learning run (or kept as a note if that kind may not be learnt).",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "description": "what to learn, as full sentences that name who or what it is about"},
         "kind": {"type": "string", "enum": ["personal", "documents", "topics", "code", "skills", "jobs", "news"],
                  "description": "personal facts, a document, a topic name, a code folder path, a skill (an instruction), "
                                 "a note on how work is done, or news"}},
         "required": ["text", "kind"]}},
    {"name": "fintan_learn", "description": "Start a learning run now with everything waiting.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "fintan_status", "description": "What is waiting to be learnt, how the last learning run went, which "
     "updates you have and what each one knows.", "parameters": {"type": "object", "properties": {}}},
    {"name": "fintan_settings", "description": "Read or change the learning switches. Give only what changes.",
     "parameters": {"type": "object", "properties": {
         "auto_learn": {"type": "boolean"},
         "waiting_at_least": {"type": "integer", "description": "learn by itself once this many items wait"},
         "hour": {"type": "integer", "description": "learn by itself once a day at this hour (0-23)"},
         "learn": {"type": "object", "description": "kind -> true/false: which kinds may be learnt"}}}},
    {"name": "fintan_job", "description": "Hand a task to the job loop, which does it in the person's systems with the "
     "workplace's tools and learns from the result.",
     "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]}},
]
NAMES = {t["name"] for t in TOOLS}
SELF_WORDS = re.compile(r"\b(fintan|remember|memori[sz]e|learn|learnt|learned|teach|taught|forget|who are you|what are "
                        r"you|your name|about yourself|what can you do|your (?:settings|switches|updates?|status)|"
                        r"know about me|what do you know(?! about (?!me\b)))\b", re.I)
STOP = set("a an the and or of to in on at for with from by is are be it this that get set do".split())
TEACH_INTENT = re.compile(r"(?i)^\s*(?:please\s+|can you\s+|could you\s+)?(?:remember|memori[sz]e|note|keep in mind|learn)\b"
                          r"(?:\s+(?:that|this))?[\s:,]+\S")
STATUS_INTENT = re.compile(r"(?i)\b(?:what do you know about me|what have you learn(?:ed|t)|your status|what are you "
                           r"learning|what is waiting)\b")
# personal facts as notes instead of learning ("keep my personal stuff as notes", "stop learning things about me"),
# and back again ("you can learn my personal facts again")
PERSONAL = r"(?:personal|about me|my (?:stuff|facts|life|details|info|information|things))"
NOTES_ON = re.compile(r"(?i)\b(?:keep|store|save)\b[^.?!]{0,40}\bnotes?\b[^.?!]{0,40}|\b(?:don'?t|do not|stop|no longer)\s+"
                      r"(?:learn|learning|train)\b[^.?!]{0,40}" + PERSONAL)
NOTES_OFF = re.compile(r"(?i)(?:\b(?:start|resume|go back to|you can|please)\s+(?:learn|learning)\b[^.?!]{0,40}" + PERSONAL +
                       r"|^\s*(?:learn|learning)\s+(?:my\s+)?personal\s+(?:stuff|facts|things|details|info|information)\b)")


def system_text(name="Fintan-9B", base="Qwen3.5-9B"):
    """name and base come from the manifest ("name", "base_model"), so one front door serves every Fintan model"""
    tools = "\n".join(json.dumps({"type": "function", "function": t}) for t in TOOLS)
    return (SELF.format(name=name, base=base) + "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" + tools + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> "
            "XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


def workplace_words(root):
    """the words of the configured workplace (tool and argument names, its description): a request using them may be work"""
    try:
        wp = json.load(open(os.path.join(root, "jobs", "workplace.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    ws = set()
    for name, sp in (wp.get("tools") or {}).items():
        if sp.get("kind") != "lookup":
            ws |= {w for w in re.split(r"[_\W]+", name.lower()) if len(w) > 2 and w not in STOP}
    return ws


def wanted(text, root):
    """does this request get Fintan's functions? (self words, or an action word of the workplace)"""
    if SELF_WORDS.search(text or ""):
        return True
    # asking to keep personal facts as notes, or to learn them again ("start learning my personal facts")
    if (NOTES_ON.search(text or "") and re.search(r"(?i)" + PERSONAL, text or "")) or NOTES_OFF.search(text or ""):
        return True
    ww = workplace_words(root)
    return bool(ww and ww & set(re.findall(r"[a-z]+", (text or "").lower())))


def parse_calls(text):
    """-> ([(name, args)], text without the calls). Reads {"name", "arguments"} JSON and the XML parameter form."""
    calls = []
    for blk in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text or "", flags=re.S):
        blk = blk.strip()
        m = re.match(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", blk, flags=re.S)
        if m:
            args = {k: v.strip() for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", m.group(2), flags=re.S)}
            for k, v in list(args.items()):
                try:
                    args[k] = json.loads(v)
                except ValueError:
                    pass
            calls.append((m.group(1), args))
            continue
        try:
            j = json.loads(blk)
            a = j.get("arguments") or {}
            calls.append((j.get("name"), json.loads(a) if isinstance(a, str) else a))
        except (ValueError, AttributeError):
            continue
    rest = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text or "", flags=re.S).strip()
    if not calls:
        # a call written without the tags: a function name alone, name(...), name {json}, or the JSON object itself.
        # Asked to remember something, the model sometimes answers with the bare text "fintan_teach".
        for m in re.finditer(r"(?m)^\s*`?(fintan_[a-z_]+)`?\s*(\((.*)\)|(\{.*\}))?\s*$", text or ""):
            raw = m.group(3) if m.group(3) is not None else m.group(4)
            args = {}
            if raw:
                try:
                    args = json.loads(raw if raw.strip().startswith("{") else "{%s}" % raw)
                except ValueError:
                    args = {}
            calls.append((m.group(1), args if isinstance(args, dict) else {}))
        m = re.search(r'\{\s*"name"\s*:\s*"(fintan_[a-z_]+)".*\}', text or "", flags=re.S)
        if not calls and m:
            try:
                j = json.loads(m.group(0))
                a = j.get("arguments") or {}
                calls.append((j["name"], json.loads(a) if isinstance(a, str) else a))
            except (ValueError, KeyError):
                pass
        if calls:
            rest = ""
    return [(n, a) for n, a in calls if n], rest


def run_call(door, name, args):
    args = args if isinstance(args, dict) else {}
    if name == "fintan_teach":
        return door.teach(str(args.get("text") or ""), "chat", str(args.get("kind") or "personal"))
    if name == "fintan_learn":
        return door.learn()
    if name == "fintan_status":
        with door.lock:
            nights = [{"update": m, "night": n} for n, m in sorted(door.model_of.items())]
        return {"waiting": door.waiting(), "kept_as_notes": door.kept_as_notes(), "learning": door.learning,
                "updates": nights, "settings": door.settings()}
    if name == "fintan_settings":
        ch = {}
        for k in ("auto_learn", "learn"):
            if k in args:
                ch[k] = args[k]
        when = {k: args[k] for k in ("waiting_at_least", "hour") if k in args}
        if when:
            ch["auto_when"] = when
        return door.set_settings(ch) if ch else door.settings()
    if name == "fintan_job":
        return door.job_add(str(args.get("request") or ""))
    return {"error": "unknown function %s" % name}


def converse(door, body, post, rounds=3):
    """the model may call its functions up to `rounds` times; -> (final engine reply as a dict, [actions])"""
    msgs = list(body.get("messages") or [])
    sys_ = system_text(getattr(door, "display_name", "Fintan-9B"), getattr(door, "base_model_short", "Qwen3.5-9B"))
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = dict(msgs[0], content=sys_ + "\n\n" + str(msgs[0].get("content") or ""))
    else:
        msgs = [{"role": "system", "content": sys_}] + msgs
    actions = []
    for _ in range(rounds):
        reply = post(dict(body, messages=msgs))
        content = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        calls, rest = parse_calls(content)
        calls = [(n, a) for n, a in calls if n in NAMES]
        if not actions:
            # a request to keep personal facts as notes, or to learn them again, changes the setting whatever the
            # model called. Teaching the request itself as a fact is dropped; a learning run is kept only when the
            # person asked for one now.
            ask_ = [m_.get("content") for m_ in body.get("messages") or [] if m_.get("role") == "user"]
            ask_ = str(ask_[-1]) if ask_ else ""
            want_ = None
            if NOTES_OFF.search(ask_):
                want_ = True
            elif NOTES_ON.search(ask_) and re.search(r"(?i)" + PERSONAL, ask_):
                want_ = False
            if want_ is not None and not any(n == "fintan_settings" and ((a or {}).get("learn") or {}).get("personal") is want_
                                             for n, a in calls):
                now_ = want_ and re.search(r"(?i)\b(?:now|right away|straight away|immediately)\b", ask_)
                calls = [("fintan_settings", {"learn": {"personal": want_}})] + \
                    [(n, a) for n, a in calls if n not in ("fintan_teach", "fintan_settings") and (n != "fintan_learn" or now_)]
        if not calls and not actions:
            # the model may say it acted ("I've recorded that ...") without calling anything. A plain request to
            # remember, or to say what it knows, is then carried out by the door itself, and the model answers from
            # the result
            said_ = [m_.get("content") for m_ in body.get("messages") or [] if m_.get("role") == "user"]
            last_ = str(said_[-1]) if said_ else ""
            if NOTES_ON.search(last_) and re.search(r"(?i)" + PERSONAL, last_):
                calls = [("fintan_settings", {"learn": {"personal": False}})]
            elif NOTES_OFF.search(last_):
                calls = [("fintan_settings", {"learn": {"personal": True}})]
            elif TEACH_INTENT.search(last_):
                calls = [("fintan_teach", {})]
            elif STATUS_INTENT.search(last_):
                calls = [("fintan_status", {})]
        if not calls:
            reply["choices"][0]["message"]["content"] = rest
            return reply, actions
        msgs.append({"role": "assistant", "content": content})
        said = [m_.get("content") for m_ in body.get("messages") or [] if m_.get("role") == "user"]
        for i_, (n, a) in enumerate(calls):
            if n == "fintan_teach" and not str((a or {}).get("text") or "").strip() and said:
                # no text given: what the person said, without the request words, is what is taught
                t_ = re.sub(r"(?i)^\s*(please\s+)?(remember|learn|note|keep in mind)(\s+that)?[:,]?\s*", "", str(said[-1])).strip()
                calls[i_] = (n, dict(a or {}, text=t_[:1].upper() + t_[1:], kind=(a or {}).get("kind") or "personal"))
        for n, a in calls:
            try:
                res = run_call(door, n, a)
            except Exception as e:  # noqa: BLE001
                res = {"error": str(e)[:200]}
            pers_ = ((a or {}).get("learn") or {}).get("personal") if isinstance(a, dict) else None
            if n == "fintan_settings" and isinstance(pers_, bool) and isinstance(res, dict) and "error" not in res:
                # (25 Nov, GPU notes check) told "you can start learning my personal facts again", the model replied
                # that a learning run had started when only the setting had changed: say plainly what changed
                res = dict(res, what_changed=(
                    "Personal facts will be learnt at the next learning run. No learning run has started now."
                    if pers_ else "Personal facts are kept as notes from now on and are not learnt."))
            actions.append({"function": n, "arguments": a, "result": res})
            msgs.append({"role": "user", "content": "<tool_response>\n%s\n</tool_response>" % json.dumps(res, ensure_ascii=False)[:3000]})
    reply = post(dict(body, messages=msgs + [{"role": "user", "content": "Now answer the person in one or two plain sentences."}]))
    content = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    reply["choices"][0]["message"]["content"] = parse_calls(content)[1]
    return reply, actions
