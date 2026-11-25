"""Run one step of the learning engine: python learn/run_engine.py STEP [arguments...]

The engine in learn/engine/ ships compiled (machine code, one file per step); this starts a step the same way
whether the engine is compiled or plain Python (for development). Each step runs in its own process, so the GPU
RAM of one step is free before the next starts.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("FINTAN_ENGINE") or os.path.join(HERE, "engine")
TOP_LEVEL = {"merge_reading", "para_split"}          # steps that run when loaded (no main function)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    sys.path.insert(0, ENGINE)
    sys.argv = [os.path.join(ENGINE, name + ".py")] + sys.argv[2:]
    mod = importlib.import_module(name)
    if name not in TOP_LEVEL and hasattr(mod, "main"):
        rc = mod.main()
        sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
