# Fintan in one container: serving (vLLM) and learning on one NVIDIA GPU.
# The base and the versions are the ones the release was tested with (25 Nov 2026): torch 2.8 (CUDA 12.8) and
# transformers 5.24.1 / peft 0.22.0 for learning, vLLM 0.34.0 in its own environment for serving.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1

# learning: the compiled engine runs on the system Python
RUN python -m pip install -q "transformers==5.24.1" "peft==0.22.0" accelerate "bitsandbytes==0.50.2" pyarrow \
        huggingface_hub \
 && python -m pip install -q --no-deps "flash-linear-attention==0.5.0" "fla-core==0.5.0" einops

# serving: vLLM with its own torch, plus the OpenAI client and MCP for the integrations
RUN python -m venv --system-site-packages /opt/vllm \
 && /opt/vllm/bin/pip install -q "vllm==0.34.0" ninja openai mcp \
 && ln -sf /opt/vllm/bin/vllm /usr/local/bin/vllm

COPY entrypoint.sh /usr/local/bin/fintan-entrypoint.sh
RUN chmod +x /usr/local/bin/fintan-entrypoint.sh

# 8900: Fintan's OpenAI-style API (chat, tools, teach, learn, settings, status)
EXPOSE 8900
ENTRYPOINT ["/usr/local/bin/fintan-entrypoint.sh"]
