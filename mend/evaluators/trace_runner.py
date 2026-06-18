# trace_runner.py
import sys
import json
import traceback
from io import StringIO

# ------------------------------------
# LINE TRACER
# ------------------------------------
def make_tracer(executed_lines):
    def tracer(frame, event, arg):
        if event == "line":
            executed_lines.add(frame.f_lineno)
        return tracer
    return tracer

# ------------------------------------
# SAFE EXECUTION
# ------------------------------------
def run_code(code_str, input_str):
    executed = set()
    stdout_backup = sys.stdout
    stdin_backup = sys.stdin

    sys.stdin = StringIO(input_str)
    output_buffer = StringIO()
    sys.stdout = output_buffer

    try:
        compiled = compile(code_str, "<candidate>", "exec")

        tracer = make_tracer(executed)
        sys.settrace(tracer)

        local_ns = {}
        exec(compiled, local_ns)

        # Prefer solve() if present
        if "solve" in local_ns and callable(local_ns["solve"]):
            result = str(local_ns["solve"]()).strip()
        else:
            # whatever the print-based program outputted
            result = output_buffer.getvalue().strip()

    except Exception:
        result = None
        err = traceback.format_exc()
        return {
            "output": result,
            "coverage": sorted(executed),
            "error": err
        }

    finally:
        sys.settrace(None)
        sys.stdout = stdout_backup
        sys.stdin = stdin_backup

    return {
        "output": result,
        "coverage": sorted(executed),
        "error": None
    }

# ------------------------------------
# ENTRY POINT
# ------------------------------------
if __name__ == "__main__":
    data = json.load(sys.stdin)
    code = data["code"]
    input_str = data["input"]

    result = run_code(code, input_str)
    print(json.dumps(result))
