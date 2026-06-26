import importlib.util
import inspect
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

test_dir = root / "tests"
if not test_dir.exists():
    print("No tests/ directory")
    sys.exit(0)

results = []
for p in sorted(test_dir.glob("test_*.py")):
    name = f"localtest_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, str(p))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"ERROR importing {p}: {e}")
        results.append((p.name, False, f"import error: {e}"))
        continue
    tests = [getattr(module, n) for n in dir(module) if n.startswith("test_") and inspect.isfunction(getattr(module, n))]
    ok = True
    msg = ""
    for t in tests:
        try:
            t()
        except AssertionError as ae:
            ok = False
            msg = f"assertion failed: {ae}"
            print(f"FAIL {p.name}::{t.__name__}: {ae}")
            break
        except Exception as e:
            ok = False
            msg = f"error: {e}"
            print(f"ERROR {p.name}::{t.__name__}: {e}")
            break
    if ok:
        print(f"OK {p.name}")
    results.append((p.name, ok, msg))

ok_all = all(r[1] for r in results)
print("\nSummary:")
for name, ok, msg in results:
    print(f"- {name}: {'OK' if ok else 'FAIL'} {msg}")

sys.exit(0 if ok_all else 2)
