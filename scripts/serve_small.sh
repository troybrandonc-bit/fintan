#!/bin/bash
# Fintan-9B on a smaller card (12 to 16 GB) or without vLLM: transformers engine (4-bit, about 9 GB) + the front door.
#   bash scripts/serve_small.sh          then send requests to http://127.0.0.1:8900/v1/chat/completions
# Learning, the switches, the model using its own functions and the job loop work as with scripts/serve_vllm.sh.
# While a learning run goes, the engine frees the GPU and chat requests get 503 "learning".
cd "$(dirname "$0")/.."
python scripts/serve_transformers.py --4bit --port 8000 > engine.log 2>&1 &
echo "waiting for the engine (a few minutes on first start)..."
until curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do sleep 5; done
[ -f jobs/workplace.json ] && (sleep 20; python jobs/job_loop.py run > jobs/run.log 2>&1) &
python door/front_door.py manifest.json --host ${FINTAN_HOST:-127.0.0.1} --port 8900 --engine http://127.0.0.1:8000 --sleep-engine \
  --learn-cmd "python learn/fintan_learn.py --4bit ${FINTAN_LEARN_ARGS}"
