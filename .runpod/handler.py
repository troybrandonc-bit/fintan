"""RunPod queue-based handler. Fintan normally runs as a load-balanced endpoint, where requests reach its own
OpenAI-style server directly (port 8900, health check /ping) and this file is not used. On a queue-based endpoint it
forwards {"input": {"path": "/v1/chat/completions", "body": {...}}} to that server, with the container's API key."""
import json
import os
import urllib.request

import runpod

URL = "http://127.0.0.1:8900"


def handler(job):
    inp = job.get("input") or {}
    path = inp.get("path") or "/v1/chat/completions"
    body = inp.get("body")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("FINTAN_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["FINTAN_API_KEY"]
    req = urllib.request.Request(URL + path, method="POST" if body is not None else "GET", headers=headers,
                                 data=json.dumps(body).encode("utf-8") if body is not None else None)
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
