"""Isolated MBPP test worker. Reads {setup, program, tests, per_timeout} as JSON on
stdin, executes setup+program once, then runs each assert with a per-assert SIGALRM
timeout (so one infinite-loop test can't hang the rest and partial credit survives).
Prints {"passed": int, "total": int}. Run in its own process group so the parent can
SIGKILL the whole tree as a backstop.
"""
import json
import signal
import sys


class _Timeout(Exception):
    pass


def _handler(signum, frame) -> None:
    raise _Timeout()


def main() -> None:
    data = json.load(sys.stdin)
    setup, program, tests = data["setup"], data["program"], data["tests"]
    per = int(data.get("per_timeout", 5))

    env = {}
    try:
        exec(setup or "", env)
        exec(program, env)
    except BaseException:  # incl. SystemExit from candidate code
        print("##LCT##" + json.dumps({"passed": 0, "total": len(tests)}))
        return

    signal.signal(signal.SIGALRM, _handler)
    passed = 0
    for t in tests:
        try:
            signal.alarm(per)
            exec(t, env)
            passed += 1
        except BaseException:  # _Timeout, AssertionError, SystemExit, etc.
            pass
        finally:
            signal.alarm(0)
    print("##LCT##" + json.dumps({"passed": passed, "total": len(tests)}))


if __name__ == "__main__":
    main()
