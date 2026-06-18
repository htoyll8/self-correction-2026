"""
MBPP+ loader (evalplus/mbppplus).

Ports the legacy function-name/arity hint: MBPP prompts don't name the function, so we infer it from the asserts and append a hint, exactly as the original study did.
"""
from mend.datasets.base import Task, take

SOURCE = "evalplus/mbppplus"

# builtins that can appear on the LHS of an assert but are not the task's function
WRAPPER_FUNCS = frozenset({"set", "sorted", "list", "tuple", "dict", "len", "max",
                           "min", "sum", "all", "any", "abs", "round", "int", "float", "str", "bool", "isclose"})


def _split_top_level_args(s: str) -> list[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def extract_function_signature(assert_lines: list[str]) -> tuple[str | None, int | None]:
    """Infer (function name, parameter count) from a task's assert statements."""
    for line in assert_lines:
        lhs = line.split("==")[0]
        i, func_start = 0, None
        while i < len(lhs):
            ch = lhs[i]
            if ch.isalpha() or ch == "_":
                if func_start is None:
                    func_start = i
            else:
                if func_start is not None:
                    name = lhs[func_start:i]
                    if i < len(lhs) and lhs[i] == "(":
                        paren, j = 1, i + 1
                        while j < len(lhs) and paren > 0:
                            if lhs[j] == "(":
                                paren += 1
                            elif lhs[j] == ")":
                                paren -= 1
                            j += 1
                        if paren == 0 and name not in WRAPPER_FUNCS:
                            return name, len(_split_top_level_args(lhs[i + 1:j - 1]))
                    func_start = None
            i += 1
    return None, None


def _build_desc(ex: dict) -> str:
    """Task prompt augmented with the legacy MBPP function-name/arity hint."""
    desc = ex["prompt"]
    fname, pcount = extract_function_signature(ex["test_list"])
    if fname:
        desc += f"\nMAKE SURE THE FUNCTION IS NAMED `{fname}`."
    if pcount is not None:
        desc += f"\nTHE FUNCTION MUST TAKE EXACTLY {pcount} PARAMETER(S)."
    return desc


def load(n_tasks: int) -> list[Task]:
    tasks = []
    for ex in take(SOURCE, n_tasks):
        imports = ex["test_imports"]
        setup = "\n".join(imports) if isinstance(imports, list) else (imports or "")
        tasks.append(Task(
            task_id=str(ex["task_id"]),
            description=_build_desc(ex),
            setup=setup,
            tests=list(ex["test_list"]),
        ))
    return tasks
