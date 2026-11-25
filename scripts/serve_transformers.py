"""Fintan-9B served with plain transformers + PEFT: an OpenAI-style engine for smaller cards.

For a 12 to 16 GB card (with --4bit, about 9 GB) or a machine without vLLM. The front door sits in front of it as
with vLLM (scripts/serve_small.sh starts both). It serves the base model and every update in manifest.json, each by
its model name; it frees the GPU when the front door asks (/sleep, while a learning run goes) and loads again,
with any new update, on /wake_up. One request is generated at a time.

    python scripts/serve_transformers.py [--4bit] [--port 8000] [--manifest manifest.json]

Endpoints: GET /v1/models, POST /v1/chat/completions (no streaming), POST /v1/load_lora_adapter, POST /sleep,
POST /wake_up. Only reachable from this machine (127.0.0.1) unless --host says otherwise.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("FINTAN_BASE")
if not BASE:                                   # "base_model" in manifest.json
    try:
        BASE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manifest.json"),
                              encoding="utf-8")).get("base_model")
    except (OSError, ValueError):
        BASE = None
BASE = BASE or "Qwen/Qwen3.5-9B"


class Engine(object):
    def __init__(self, manifest, four_bit):
        self.manifest, self.four_bit = os.path.abspath(manifest), four_bit
        self.m = self.tok = None
        self.names, self.base_name = [], "fintan-9b"
        self.lock = threading.Lock()
        self.sleeping = False

    def load(self):
        import torch
        from transformers import AutoTokenizer
        man = json.load(open(self.manifest, encoding="utf-8"))
        self.base_name = man.get("base") or "fintan-9b"
        kw = {"device_map": "cuda"}
        if self.four_bit:
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
        names = []
        root = os.path.dirname(self.manifest)
        for n in man.get("nights") or []:
            path = os.path.join(root, n["path"])
            if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
                continue
            if not names:
                from peft import PeftModel
                m = PeftModel.from_pretrained(m, path, adapter_name=n["model"])
            else:
                m.load_adapter(path, adapter_name=n["model"])
            names.append(n["model"])
        m.eval()
        self.m, self.tok, self.names, self.sleeping = m, tok, names, False
        stop = {tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                stop.add(i)
        self.stop = sorted(x for x in stop if x is not None)
        print("engine ready: base %s, updates %s, GPU %.1f GB" % (BASE, names, torch.cuda.memory_allocated() / 2 ** 30), flush=True)

    def sleep(self):
        import torch
        with self.lock:
            self.m = None
            gc.collect()
            torch.cuda.empty_cache()
            self.sleeping = True

    def add(self, name, path):
        with self.lock:
            if name in self.names or self.m is None:
                return
            path = path if os.path.isabs(path) else os.path.join(os.path.dirname(self.manifest), path)
            if not self.names:
                from peft import PeftModel
                self.m = PeftModel.from_pretrained(self.m, path, adapter_name=name)
            else:
                self.m.load_adapter(path, adapter_name=name)
            self.names.append(name)

    def chat(self, body):
        import torch
        ctk = body.get("chat_template_kwargs") or {}
        prompt = self.tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=bool(ctk.get("enable_thinking", False)))
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.m.device)
        temp = float(body.get("temperature") or 0)
        kw = dict(max_new_tokens=int(body.get("max_tokens") or 512), pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                  eos_token_id=self.stop, do_sample=temp > 0)
        if temp > 0:
            kw.update(temperature=temp, top_p=float(body.get("top_p") or 0.95))
        name = body.get("model")
        with self.lock, torch.no_grad():
            if self.m is None:
                raise RuntimeError("sleeping")
            if name in self.names:
                self.m.set_adapter(name)
                g = self.m.generate(**enc, **kw)
            elif self.names:
                with self.m.disable_adapter():
                    g = self.m.generate(**enc, **kw)
            else:
                g = self.m.generate(**enc, **kw)
        new = g[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return {"id": "fintan-%d" % int(time.time() * 1000), "object": "chat.completion", "created": int(time.time()),
                "model": name or self.base_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop" if len(new) < kw["max_new_tokens"] else "length"}],
                "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": int(len(new)),
                          "total_tokens": int(enc["input_ids"].shape[1] + len(new))}}


def handler(eng):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                ids = [eng.base_name] + list(eng.names)
                return self._json(200, {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "the body must be JSON"}})
            p = self.path.split("?")[0].rstrip("/")
            if p == "/sleep":
                eng.sleep()
                return self._json(200, {"sleeping": True})
            if p == "/wake_up":
                with eng.lock:
                    if eng.m is None:
                        eng.load()
                return self._json(200, {"sleeping": False, "updates": eng.names})
            if p == "/v1/load_lora_adapter":
                try:
                    eng.add(body["lora_name"], body["lora_path"])
                    return self._json(200, {"loaded": body["lora_name"]})
                except Exception as e:  # noqa: BLE001
                    return self._json(400, {"error": {"message": str(e)[:300]}})
            if p == "/v1/chat/completions":
                if body.get("stream"):
                    return self._json(400, {"error": {"message": "streaming is not supported by this engine"}})
                if eng.sleeping or eng.m is None:
                    return self._json(503, {"error": {"message": "the engine is paused for a learning run"}})
                try:
                    return self._json(200, eng.chat(body))
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": {"message": str(e)[:300]}})
            return self._json(404, {"error": {"message": "not found"}})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.json"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    a = ap.parse_args()
    eng = Engine(a.manifest, a.four_bit)
    eng.load()
    srv = ThreadingHTTPServer((a.host, a.port), handler(eng))
    print("engine on http://%s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
