#!/bin/bash
# Starts Fintan in the container.
#   FINTAN_MODEL     which model: fintanlabs/Fintan-9B (24 GB cards) or fintanlabs/Fintan-4B (16-20 GB cards)
#   FINTAN_API_KEY   required: every request must carry it (Authorization: Bearer <key>)
#   FINTAN_DATA      where everything is kept (default /workspace on RunPod, else /data): the model's files, what it
#                    learns (updates, notes, settings) and the base model download. Keep it on a volume, or what was
#                    learnt is lost when the container goes.
set -e
MODEL=${FINTAN_MODEL:-fintanlabs/Fintan-9B}
# RunPod serverless mounts a network volume at /runpod-volume, pods at /workspace
DATA=${FINTAN_DATA:-$([ -d /runpod-volume ] && echo /runpod-volume || ([ -d /workspace ] && echo /workspace || echo /data))}
APP=$DATA/fintan
mkdir -p "$DATA"
export HF_HOME=${HF_HOME:-$DATA/hf}

if [ -z "$FINTAN_API_KEY" ] && [ -z "$FINTAN_ALLOW_OPEN" ]; then
  echo "Set FINTAN_API_KEY (any long random string) so only you can use this Fintan."
  exit 1
fi

# RunPod: keep its SSH and tools working when a key is given
[ -x /start.sh ] && [ -n "$PUBLIC_KEY" ] && (/start.sh > /dev/null 2>&1 &)

if [ ! -f "$APP/manifest.json" ]; then
  echo "First start: downloading $MODEL into $APP"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$APP')"
elif [ "${FINTAN_UPDATE_CODE:-1}" = 1 ]; then
  # newer code and engine from the model's repo; what was learnt here (manifest, updates, notes, inbox, settings) stays
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$APP', allow_patterns=['door/*', 'scripts/*', 'learn/*', 'jobs/*.py', 'mcp/*', 'relay/*', 'self/manual.json'])" \
    || echo "Could not check for newer code; starting with what is here."
fi
rm -rf "$APP/.cache"
cd "$APP"
export FINTAN_HOST=${FINTAN_HOST:-0.0.0.0}
exec bash scripts/serve_vllm.sh
