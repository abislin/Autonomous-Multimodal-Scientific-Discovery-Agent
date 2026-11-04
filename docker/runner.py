"""Container-side runner. Executes /workspace/experiment.py and reports JSON."""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

SCRIPT = Path("/workspace/experiment.py")

def main() -> int:
    result = {"ok": False, "metrics": {}, "error": None, "traceback": None}
    try:
        spec = importlib.util.spec_from_file_location("experiment", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main") and callable(mod.main):
            out = mod.main()
            if isinstance(out, dict):
                result["metrics"] = {k: float(v) for k, v in out.items()
                                     if isinstance(v, (int, float))}
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=8)
    finally:
        print("__RESULT__" + json.dumps(result))
    return 0 if result["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())