"""fintan learn: teach Fintan what is waiting in the inbox, on this machine.

The front door runs this (python door/front_door.py --learn-cmd "python learn/fintan_learn.py"), or by hand:

    python learn/fintan_learn.py [--inbox DIR] [--manifest manifest.json] [--kinds personal,documents,...]
                                 [--4bit | --full] [--keep-work]

It reads what is waiting, learns it into a new update, checks the update against the plain model on wordings it
never practised, and keeps it only when it does better (what was taught is otherwise kept as notes). The work is
done by the learning engine in learn/engine/, which ships compiled. Report: learn_report.json beside manifest.json.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FINTAN_ENGINE") or os.path.join(HERE, "engine"))
# the model to teach: "base_model" in manifest.json (Fintan-9B: Qwen/Qwen3.5-9B, Fintan-4B: Qwen/Qwen3.5-4B)
if not os.environ.get("FINTAN_BASE"):
    _m = sys.argv[sys.argv.index("--manifest") + 1] if "--manifest" in sys.argv else (
        os.environ.get("FINTAN_MANIFEST") or os.path.join(HERE, "..", "manifest.json"))
    try:
        import json
        with open(_m, encoding="utf-8") as _f:
            _b = json.load(_f).get("base_model")
        if _b:
            os.environ["FINTAN_BASE"] = _b
    except (OSError, ValueError):
        pass
import learn_main  # noqa: E402

if __name__ == "__main__":
    learn_main.main()
