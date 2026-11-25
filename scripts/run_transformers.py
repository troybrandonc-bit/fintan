"""Fintan-9B with plain transformers + PEFT, full precision or 4-bit.

    python scripts/run_transformers.py "Who chaired the talks in Antalya?"
    python scripts/run_transformers.py --4bit "..."          # compressed base model (about 9 GB of GPU RAM)
    python scripts/run_transformers.py --check               # ask this update's check questions and print the score
    python scripts/run_transformers.py --check --4bit

The base model is Qwen/Qwen3.5-9B, downloaded from its own repository; its weights are not changed. The update is a
small adapter file (updates/<date>/). The router (door/night_router4.py) decides per question: a question about
something the update taught goes to the model with the update, every other question goes to the plain base model.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "door"))
from night_router4 import NightRouter4  # noqa: E402
from skill_router import SkillRouter  # noqa: E402

BASE = "Qwen/Qwen3.5-9B"
try:                                           # Fintan-4B and others: "base_model" in manifest.json
    BASE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manifest.json"),
                          encoding="utf-8")).get("base_model") or BASE
except (OSError, ValueError):
    pass
ROUTER = None


def load(four_bit):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    kw = {"device_map": "cuda"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, __import__('os').environ.get('FINTAN_COMPUTE_DTYPE', 'bfloat16')),
                                                       bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    except (ValueError, KeyError):                       # Qwen3.5 checkpoints register as image-text models
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(BASE, **kw)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    man = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
    nights = man["nights"]
    # every update in the manifest, each as its own named adapter (a news week, a learning run at home, ...); the
    # router picks one per question
    m = PeftModel.from_pretrained(m, os.path.join(ROOT, nights[0]["path"]), adapter_name="n%d" % int(nights[0]["night"]))
    for n in nights[1:]:
        m.load_adapter(os.path.join(ROOT, n["path"]), adapter_name="n%d" % int(n["night"]))
    m.eval()
    night = nights[-1]
    router = NightRouter4(background=json.load(open(os.path.join(ROOT, man["background"]), encoding="utf-8")))
    skills = SkillRouter()
    for n in man["nights"]:
        router.add_night(int(n["night"]), json.load(open(os.path.join(ROOT, n["facts"]), encoding="utf-8")))
        if n.get("skill"):
            skills.add_night(int(n["night"]), n["skill"].get("trigger_words"))
    router.skills = skills                     # a learnt skill when no learnt fact is asked about (route_any below)
    return m, tok, router, night


def stops(tok):
    """end of the answer: the end-of-turn token as well as end of text. Without the end-of-turn token, answers run on
    into "user ... assistant ..." turns that the model makes up."""
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(x for x in ids if x is not None)


def answer(m, tok, texts, with_update, max_new=48, batch=16, stream=False, night=None):
    """with_update: answer with update `night` (a night number; None = the newest loaded)"""
    import torch
    out = []
    if with_update:
        names = list(m.peft_config.keys())
        m.set_adapter("n%d" % int(night) if night is not None and "n%d" % int(night) in names else names[-1])
    streamer = None
    if stream:
        from transformers import TextStreamer
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    for i in range(0, len(texts), batch):
        prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for t in texts[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            if with_update:
                g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=stops(tok), streamer=streamer)
            else:
                with m.disable_adapter():
                    g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                                   eos_token_id=stops(tok), streamer=streamer)
        out += [x.strip() for x in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def norm(x):
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(x).lower()).split())


def right(out, value, question):
    """the answer states the value: most of the value's words that the question did not already give"""
    q = set(norm(question).split())
    vw = [w for w in norm(value).split() if (len(w) > 2 or w.isdigit()) and w not in q]
    ow = norm(out).split()
    hit = lambda w: w in ow or (len(w) >= 5 and any(x[:5] == w[:5] for x in ow))  # noqa: E731
    return bool(vw) and sum(hit(w) for w in vw) / len(vw) >= 0.6


# --chain: questions that need several learnt facts. With an update on, the model answers one fact and
# stops, so here the plain model (update off) does the reasoning and asks for one fact at a time, and the update
# answers each one-fact question. Before the update is asked, its own list of what it learnt (subject and fact names,
# never values) is checked; an unknown pair gets "No record of that" and what is known about that subject.
PLAN = ("You are answering a question about things you do not know yet, but you have a reliable lookup that answers "
        "simple one-fact questions. Never say something does not exist: ask the lookup instead.\n\n"
        "Question: {q}\n\nFacts from the lookup so far:\n{facts}\n\n"
        "If these facts are enough, reply with exactly one line: Answer: <the answer only>\n"
        "Otherwise reply with exactly one line asking for ONE missing fact, naming its subject exactly as written: "
        "Ask: <a simple question about one thing>\n\n"
        "Rules: a question may describe something only through another thing (\"X has a Y. What is the Z of that Y?\"): "
        "first ask what X's Y is, then ask about that Y by its name. If the lookup says it has no record, read what "
        "it says it does know and ask a different question.\n\n"
        "Before that line, write one or two short sentences: what the question needs, and which of those facts you "
        "already have. Answer only when every fact the question needs is in the list.")
STOP = set("the a an of in is are was what which who how many much does do has have that this its it for to and with "
           "number".split())


def facts_index(paths):
    idx, name = {}, {}
    for p in paths:
        for f in json.load(open(p, encoding="utf-8")):
            k = norm(f["subject"])                       # "Cop31" and "COP31" are one subject
            name.setdefault(k, f["subject"])
            idx.setdefault(name[k], []).append(f["relation"])
    return idx


def gate(question, idx):
    """None: ask the update. A string: the reply instead (no record, and what is known about the subject)."""
    qn = " " + norm(question) + " "
    subj = [s for s in idx if norm(s) and " " + norm(s) + " " in qn]
    if not subj:
        return "No record of anything named in that question."
    s = max(subj, key=len)
    words, sw = norm(question).split(), norm(s).split()
    label = {words[i - 1] for i in range(1, len(words) - len(sw) + 1) if words[i:i + len(sw)] == sw}
    qw = [w for w in words if w not in STOP and w not in sw and w not in label]
    for r in idx[s]:
        rw = [w for w in norm(r).split() if w not in STOP]
        if any(len(w) >= 4 and any(q[:5] == w[:5] for q in qw) for w in rw):
            return None
    return "No record of that. Known about %s: %s." % (s, ", ".join(dict.fromkeys(idx[s])))


def chain(m, tok, question, idx, rounds=7, show=True):
    facts = []
    for r in range(rounds):
        out = answer(m, tok, [PLAN.format(q=question, facts="\n".join(facts) or "(none yet)")], False, max_new=160, batch=1)[0]
        lines = [x.strip() for x in out.split("</think>")[-1].strip().splitlines() if x.strip()]
        tagged = [x for x in lines if x.lower().startswith(("ask:", "answer:"))]
        line = tagged[-1] if tagged else (lines[-1] if lines else "")
        if show:
            for x in lines:
                if x is not line and not x.lower().startswith(("ask:", "answer:")):
                    print("  thinking: %s" % x, flush=True)
        if line.lower().startswith("ask:") and r < rounds - 1:
            q = line[4:].strip()
            g = gate(q, idx)
            v = g if g is not None else answer(m, tok, [q], True, max_new=32, batch=1, night=ROUTER.route(q))[0].strip().splitlines()[0]
            facts.append("- %s %s" % (q, v))
            if show:
                print("  lookup: %s -> %s" % (q, v), flush=True)
            continue
        return line[7:].strip() if line.lower().startswith("answer:") else line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--night", type=int, default=None, help="with --check: which update's questions (default: the newest)")
    ap.add_argument("--base", action="store_true", help="answer with the plain base model (to compare)")
    ap.add_argument("--chain", action="store_true", help="a question that needs several learnt facts: one fact at a time")
    a = ap.parse_args()
    import torch
    m, tok, router, night = load(a.four_bit)
    global ROUTER
    ROUTER = router
    if a.night is not None:
        night = next(n for n in json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))["nights"]
                     if int(n["night"]) == a.night)
    if a.check:
        qs = json.load(open(os.path.join(ROOT, os.path.dirname(night["facts"]), "check_questions.json"), encoding="utf-8"))
        texts = [x["question"] for x in qs]
        routed = [router.route(t) is not None for t in texts]
        picked = [router.route(t) for t in texts]
        base = answer(m, tok, texts, False)
        upd = answer(m, tok, texts, True, night=night["night"])
        served = list(base)
        for nn in sorted({p for p in picked if p is not None}):
            idx = [i for i, p in enumerate(picked) if p == nn]
            for i, o in zip(idx, answer(m, tok, [texts[i] for i in idx], True, night=nn)):
                served[i] = o
        sc = lambda outs: 100.0 * sum(right(o, x["answer"], x["question"]) for o, x in zip(outs, qs)) / len(qs)  # noqa: E731
        print("FINTAN CHECK %s | %d questions | base model %.1f%% | with the update %.1f%% | as served (router) %.1f%%, %d of %d routed | GPU peak %.1f GB" % (
            "4-bit" if a.four_bit else "full precision", len(qs), sc(base), sc(upd), sc(served), sum(routed), len(qs),
            torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
        ordinary = ["What is 17 times 23?", "Write a haiku about rain.", "Explain what a binary search does.",
                    "What is the capital of Australia?", "Translate 'good morning' into Spanish."]
        print("FINTAN CHECK ordinary questions routed to the update: %d of %d" % (sum(router.route(t) is not None for t in ordinary), len(ordinary)))
        for x, b, s in list(zip(qs, base, served))[:3]:
            print("  Q: %s\n     base:   %s\n     fintan: %s" % (x["question"], b[:90].replace("\n", " "), s[:90].replace("\n", " ")))
        return
    if not a.question:
        ap.error("give a question, or --check")
    if a.chain and not a.base:
        man = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
        idx = facts_index([os.path.join(ROOT, n["facts"]) for n in man["nights"]])
        print("Answer: %s" % chain(m, tok, a.question, idx), flush=True)
        return
    pick = router.route(a.question)
    if pick is None:
        pick = router.skills.route(a.question)
    use = (pick is not None) and not a.base
    answer(m, tok, [a.question], use, max_new=256, batch=1, stream=True, night=pick)
    path = next((n["path"] for n in json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))["nights"]
                 if use and int(n["night"]) == int(pick)), "")
    print("[%s]" % ("answered with the %s update" % os.path.basename(path) if use else "answered by the base model"), file=sys.stderr)


if __name__ == "__main__":
    main()
