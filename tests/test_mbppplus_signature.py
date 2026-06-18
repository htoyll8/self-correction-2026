"""Pin the legacy MBPP function-name/arity parser so the rerun's parity claim is enforced.

extract_function_signature infers (name, param_count) from a task's asserts. These cases
encode the legacy behavior we depend on: ignore wrapper builtins, see through nesting,
count top-level args only, and return (None, None) when there's no call.
"""
from mend.datasets.mbppplus import extract_function_signature

CASES = [
    # (assert lines, expected name, expected arity)
    (["assert similar_elements((3,4,5,6),(5,7,4,10)) == (4,5)"], "similar_elements", 2),
    (["assert min_cost([[1, 2, 3]], 2, 2) == 8"], "min_cost", 3),
    (["assert is_not_prime(2) == False"], "is_not_prime", 1),
    # wrapper builtin on the LHS is skipped; the real function inside is found
    (["assert set(my_func(3)) == set([1, 2, 3])"], "my_func", 1),
    (["assert max(my_func(3)) == 9"], "my_func", 1),
    (["assert min(my_func(3)) == 0"], "my_func", 1),
    # commas inside nested calls/collections don't inflate the arity
    (["assert merge([1, 2], {'a': 1}, (3, 4)) == 0"], "merge", 3),
    # no call on the LHS -> nothing to infer
    (["assert x == 3"], None, None),
]


def test_extract_function_signature():
    for lines, name, arity in CASES:
        assert extract_function_signature(lines) == (name, arity), lines


if __name__ == "__main__":
    test_extract_function_signature()
    print(f"ok: {len(CASES)} cases")
