#!/bin/bash
# Fintan as an OpenAI-style server that also learns, on one GPU.
#   bash scripts/serve_vllm.sh           then send requests to http://127.0.0.1:8900/v1/chat/completions
# vLLM serves the base model and every update in manifest.json; the front door routes each request, takes what you
# teach (POST /fintan/teach) and runs learning (learn/fintan_learn.py) by the learning switches (/fintan/settings).
# While a learning run goes, vLLM is put to sleep so the learner has the GPU; chat requests then get 503 "learning".
# The dev-mode endpoints (/sleep, /wake_up) are only reachable from this machine (127.0.0.1).
# Send "chat_template_kwargs": {"enable_thinking": false} with each request (the updates were learned for direct answers).
#
# Card size (vLLM reserves GPU RAM for every update slot at start; 4 rank-512 slots do not fit a 24 GB card):
#   40 GB or more   up to 4 updates on the GPU at once
#   under 40 GB     base weights in 8-bit (fp8), one update on the GPU at a time (the others wait in CPU RAM), no
#                   CUDA graphs, fewer parallel requests; if vLLM still cannot start, the transformers engine (4-bit,
#                   scripts/serve_small.sh) takes over by itself
cd "$(dirname "$0")/.."
# which model: "base_model" (the weights) and "base" (the name it answers to) in manifest.json
BASE_MODEL=$(python -c "import json; print(json.load(open('manifest.json')).get('base_model') or 'Qwen/Qwen3.5-9B')")
NAME=$(python -c "import json; print(json.load(open('manifest.json'))['base'])")
MODS=$(python -c "import json; m = json.load(open('manifest.json')); print(' '.join('%s=%s' % (n['model'], n['path']) for n in m['nights']))")
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
if [ "${GPU_MB:-0}" -ge 40000 ]; then
  N=$(python -c "import json; print(min(4, max(1, len(json.load(open('manifest.json'))['nights']) + 2)))")
  FIT="--max-loras $N"
  SHARE=${FINTAN_GPU_SHARE:-0.85}
else
  # in bf16 the 9B and one rank-512 update slot do not fit 24 GB, so its base weights load in 8-bit (fp8),
  # about 9 GB; the updates stay as they are. Smaller models keep bf16.
  case "$BASE_MODEL" in *9B*) Q="--quantization fp8";; *) Q="";; esac
  FIT="$Q --max-loras 1 --max-num-seqs 8 --enforce-eager"
  SHARE=${FINTAN_GPU_SHARE:-0.93}
fi
# tool calling for agents and apps (OpenAI "tools": replies carry structured tool_calls); FINTAN_TOOL_PARSER picks
# vLLM's parser for the model's call format
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
vllm serve $BASE_MODEL --served-model-name $NAME --port 8000 --host 127.0.0.1 \
  --enable-lora --max-lora-rank 512 $FIT --max-cpu-loras 32 ${MODS:+--lora-modules $MODS} \
  --enable-auto-tool-choice --tool-call-parser ${FINTAN_TOOL_PARSER:-qwen3_coder} \
  --enable-sleep-mode --max-model-len 4096 --gpu-memory-utilization $SHARE > vllm.log 2>&1 &
VPID=$!
echo "waiting for the engine (a few minutes on first start)..."
until curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
  if ! kill -0 $VPID 2>/dev/null; then
    echo "vLLM could not start on this card (see vllm.log); using the transformers engine instead"
    exec bash scripts/serve_small.sh
  fi
  sleep 5
done
# the job loop works on its own once a workplace is described (jobs/workplace.json): it takes tasks handed over through
# the front door (POST /fintan/job, or the model itself from chat) and asks only when unsure (GET /fintan/jobs)
[ -f jobs/workplace.json ] && (sleep 20; python jobs/job_loop.py run > jobs/run.log 2>&1) &
python door/front_door.py manifest.json --host ${FINTAN_HOST:-127.0.0.1} --port 8900 --engine http://127.0.0.1:8000 --sleep-engine \
  --learn-cmd "python learn/fintan_learn.py ${FINTAN_LEARN_ARGS}"
