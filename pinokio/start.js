// Starts Fintan: vLLM serves the model, the front door listens on http://127.0.0.1:8900 (this computer only).
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { PATH: "{{path.resolve(cwd, 'vllm_env/bin')}}:{{envs.PATH}}" },
        message: ["bash scripts/serve_vllm.sh"],
        on: [{ event: "/front door on (http:\\/\\/[0-9.:]+)/", done: true }]
      }
    },
    {
      method: "local.set",
      params: { url: "{{input.event[1]}}" }
    }
  ]
}
