# Fintan

Every AI model today is finished the day it ships. Fintan keeps growing.

Fintan models are open models that keep learning after release, on your own GPU. Teach one your documents, a topic,
your codebase, skills or facts about yourself, through an OpenAI-style API or just by telling it in chat, and it learns
them into its own weights in a learning run of about 20 minutes. Its general scores stay where they were.

| Model | Base | GPU | Download |
|---|---|---|---|
| Fintan-9B | Qwen3.5-9B | 24 GB | [huggingface.co/fintanlabs/Fintan-9B](https://huggingface.co/fintanlabs/Fintan-9B) |
| Fintan-4B | Qwen3.5-4B | 16 to 20 GB | coming |

The model pages have the results, failures included. This repository has the open parts (the server, scripts, job
loop, MCP server and relay) and the ways to run it. The learning engine ships compiled with the models.

## Run it

### Docker, with a chat app (Open WebUI)

Needs Docker with an NVIDIA GPU (the NVIDIA Container Toolkit).

```bash
git clone https://github.com/troybrandonc-bit/fintan && cd fintan
echo "FINTAN_API_KEY=$(openssl rand -hex 24)" > .env
docker compose up -d
```

Open http://localhost:3000. The first start downloads the model (about 20 GB). Everything Fintan learns is kept in the
`fintan-data` volume. For a 16 to 20 GB card, add `FINTAN_MODEL=fintanlabs/Fintan-4B` to `.env`.

### On a rented GPU (RunPod)

See [runpod/README.md](runpod/README.md). What it learns is kept on your own network volume.

### Without Docker

Download a model from Hugging Face and run `bash scripts/serve_vllm.sh` in its folder (Linux, NVIDIA, Python 3.10 to
3.13). The model's README has the details.

## Use it from other apps

Fintan speaks the OpenAI API at `http://localhost:8900/v1`, with tool calling, so any app or agent framework that lets
you set an OpenAI-compatible address can use it: chat apps, coding agents, workflow tools and agent frameworks. Use
`fintan-9b` (or `fintan-4b`) as the model name and your `FINTAN_API_KEY` as the key.

For MCP apps (Claude Desktop, Cursor, VS Code and others), `mcp/fintan_mcp.py` gives them four tools: ask, teach, learn
now and status:

```json
{"mcpServers": {"fintan": {"command": "python", "args": ["/path/to/fintan/mcp/fintan_mcp.py"],
                            "env": {"FINTAN_URL": "http://127.0.0.1:8900", "FINTAN_API_KEY": "your key"}}}}
```

## Privacy

- Everything runs on your machine. Nothing you teach it is sent anywhere.
- The only thing that goes online is learning a topic you name. It goes through our relay (code in `relay/`), so
  Wikipedia never sees your IP address; your topic is sent among 4 decoys, the relay stores nothing, and every request
  is written to `calls.log` first. `FINTAN_OFFLINE=1` stops it.
- Personal facts can be kept as notes instead of learnt. Passwords, keys and card numbers are removed before anything
  is stored. Any learning run can be undone.
- `scripts/backup_hf.py` saves what it learnt to your own private Hugging Face repo, never to us.

## License

Apache-2.0. The base models keep their own licences.
