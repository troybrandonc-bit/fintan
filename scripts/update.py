"""Update Fintan's code and learning engine to the newest version, keeping everything it has learnt.

    python scripts/update.py                 (run it in the model's folder; then restart the server)
    python scripts/update.py --check         (only say whether anything would change)

It downloads from Hugging Face only when you run it. Replaced: door/, scripts/, learn/, mcp/, relay/, the job loop
code, Fintan's manual, README.md and RESULTS.md. Kept as they are: manifest.json, updates/ (everything it learnt),
notes, the inbox, settings, your jobs/workplace.json and the job loop's state, usage and calls.log.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = {"fintan-9b": "fintanlabs/Fintan-9B", "fintan-4b": "fintanlabs/Fintan-4B"}
PATTERNS = ["door/*", "scripts/*", "learn/*", "mcp/*", "relay/*", "jobs/*.py", "jobs/workplace.example.json",
            "self/manual.json", "README.md", "RESULTS.md", "LICENSE"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("pip install -U huggingface_hub, then run this again")
    man = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = args[0] if args else REPOS.get(man.get("base") or "fintan-9b", "fintanlabs/Fintan-9B")
    import fnmatch
    files = [f for f in HfApi().list_repo_files(repo) if any(fnmatch.fnmatch(f, p) for p in PATTERNS)]
    before = {f: digest(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}
    if "--check" in sys.argv:
        print("%s: %d files would be checked; run without --check to update them" % (repo, len(files)))
        return
    snapshot_download(repo, local_dir=ROOT, allow_patterns=PATTERNS)
    changed = [f for f in files if before.get(f) != digest(os.path.join(ROOT, f))]
    if changed:
        print("Updated %d files from %s:" % (len(changed), repo))
        for f in changed[:30]:
            print("  " + f)
        print("What it learnt is unchanged. Restart the server to use the new version.")
    else:
        print("Already up to date (%s)." % repo)


if __name__ == "__main__":
    main()
