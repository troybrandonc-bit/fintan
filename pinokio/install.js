// Installs Fintan: the learning libraries (the versions the release was tested with), vLLM in its own environment,
// and the model's files from Hugging Face. The base model itself downloads on the first start.
module.exports = {
  requires: { bundle: "ai" },
  run: [
    {
      when: "{{platform !== 'linux' || gpu !== 'nvidia'}}",
      method: "notify",
      params: { html: "Fintan needs Linux with an NVIDIA GPU (learning runs on CUDA)." },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install \"torch==2.8.0\" --index-url https://download.pytorch.org/whl/cu128",
          "uv pip install \"transformers==5.24.1\" \"peft==0.22.0\" accelerate \"bitsandbytes==0.50.2\" pyarrow huggingface_hub",
          "uv pip install --no-deps \"flash-linear-attention==0.5.0\" \"fla-core==0.5.0\" einops"
        ]
      }
    },
    {
      method: "shell.run",
      params: { venv: "vllm_env", message: ["uv pip install \"vllm==0.34.0\""] }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: ["python -c \"from huggingface_hub import snapshot_download; snapshot_download('{{args && args.model ? args.model : 'fintanlabs/Fintan-9B'}}', local_dir='app')\""]
      }
    },
    { method: "notify", params: { html: "Fintan is installed. Click Start." } }
  ]
}
