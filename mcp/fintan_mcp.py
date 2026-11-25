"""Fintan as an MCP server: any MCP app (Claude Desktop, Cursor, VS Code, Windsurf, agent frameworks) can ask Fintan,
teach it, and start or check its learning. It talks to the front door already running on this machine
(scripts/serve_vllm.sh); nothing here goes online.

    pip install mcp
    python mcp/fintan_mcp.py                         # stdio; FINTAN_URL (default http://127.0.0.1:8900)

In Claude Desktop (claude_desktop_config.json) or Cursor (.cursor/mcp.json):

    {"mcpServers": {"fintan": {"command": "python", "args": ["/path/to/mcp/fintan_mcp.py"],
                               "env": {"FINTAN_URL": "http://127.0.0.1:8900"}}}}
"""
import json
import os
import urllib.request

try:
    from mcp.server.fastmcp import FastMCP as _Server       # mcp 1.x
except ImportError:
    from mcp.server.mcpserver import MCPServer as _Server   # mcp 2.x (FastMCP renamed)

URL = os.environ.get("FINTAN_URL", "http://127.0.0.1:8900").rstrip("/")
KEY = os.environ.get("FINTAN_API_KEY")         # when the front door has one
KINDS = ("documents", "personal", "topics", "code", "skills", "jobs", "news")
mcp = _Server("fintan")


def _call(path, body=None, timeout=600):
    req = urllib.request.Request(URL + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode("utf-8") if body is not None else None,
                                 headers=dict({"Content-Type": "application/json"}, **(
                                     {"Authorization": "Bearer " + KEY} if KEY else {})))
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


@mcp.tool()
def ask(question: str) -> str:
    """Ask Fintan a question. It answers from what it has learned (documents, code, topics, news, facts about the
    person) as well as from what the base model knows."""
    r = _call("/v1/chat/completions", {"model": "fintan", "messages": [{"role": "user", "content": question}],
                                        "max_tokens": 400, "temperature": 0,
                                        "chat_template_kwargs": {"enable_thinking": False}})
    text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if r.get("fintan_lookups"):
        text += "\n\n(facts used: %s)" % "; ".join("%s %s" % (x["asked"], x["got"]) for x in r["fintan_lookups"])
    return text


@mcp.tool()
def teach(text: str, kind: str = "documents", source: str = "") -> str:
    """Give Fintan something to learn at its next learning run. kind: documents (any text), personal (facts about
    the person), topics (a topic name; its Wikipedia articles are fetched through Fintan's relay), code (the path of
    a folder of Python on this machine), skills (an instruction, or "Request: ... Answer: ..." examples), jobs (how
    a piece of work is done), news."""
    if kind not in KINDS:
        return "kind must be one of: %s" % ", ".join(KINDS)
    r = _call("/fintan/teach", {"text": text, "source": source or "mcp", "kind": kind})
    if not r.get("accepted"):
        return "Not accepted: %s" % r.get("reason", "")
    if r.get("kept_as_note"):
        return "Kept as a note (learning for %s is switched off), used with questions it matches." % kind
    extra = " (%d articles fetched)" % r["articles"] if r.get("articles") else ""
    return "Accepted%s. %s item(s) now wait for the next learning run." % (extra, r.get("waiting"))


@mcp.tool()
def learn_now() -> str:
    """Start a learning run now on what is waiting. On one GPU, Fintan stops answering until it ends (minutes)."""
    r = _call("/fintan/learn", {})
    return "Learning started." if r.get("started") else "Not started: %s" % r.get("reason", "")


@mcp.tool()
def status() -> str:
    """What Fintan has learned (its updates), what is waiting, notes kept, and whether a learning run is going."""
    s = _call("/fintan/status")
    lr = s.get("learning") or {}
    return json.dumps({"model": s.get("base"), "updates": s.get("nights"), "waiting_to_learn": s.get("inbox_waiting"),
                       "kept_as_notes": s.get("kept_as_notes"), "learning_now": lr.get("running"),
                       "learning_runs": lr.get("runs"), "last_run": lr.get("last")}, indent=1)


if __name__ == "__main__":
    mcp.run()
