"""Save what this Fintan has learnt to your own private Hugging Face repo, and bring it back on another machine.
Useful on rented GPUs (RunPod, Vast.ai), whose disks are wiped when the machine stops.

    python scripts/backup_hf.py save    --repo yourname/my-fintan     # makes the repo private if it is new
    python scripts/backup_hf.py restore --repo yourname/my-fintan     # into this folder, after downloading Fintan

Uses your own Hugging Face token (HF_TOKEN, or `huggingface-cli login`). Saved: the manifest, every update learnt here,
notes, what waits to be learnt, settings, the job loop's workplace and rules, the learning report and calls.log.
Nothing goes to Fintan Labs. A repo that is public is refused: what you taught may be personal.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ["manifest.json", "settings.json", "learn_report.json", "calls.log", "usage.json", "updates/**", "notes/**",
        "inbox/**", "jobs/**", "self/.taught"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["save", "restore"])
    ap.add_argument("--repo", required=True, help="your own repo, e.g. yourname/my-fintan")
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if a.action == "save":
        api.create_repo(a.repo, private=True, exist_ok=True)
        if not api.repo_info(a.repo).private:
            sys.exit("refused: %s is public, and what you taught may be personal. Use a private repo." % a.repo)
        api.upload_folder(folder_path=ROOT, repo_id=a.repo, allow_patterns=KEEP, commit_message="Fintan backup")
        print("saved to %s (private)" % a.repo)
    else:
        snapshot_download(a.repo, local_dir=ROOT, allow_patterns=KEEP)
        print("restored from %s into %s; restart the server to use it" % (a.repo, ROOT))


if __name__ == "__main__":
    main()
