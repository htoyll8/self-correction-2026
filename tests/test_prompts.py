"""Pin the critique+refine prompt wording byte-for-byte to the original study.

These are the prompts that were embedded in the legacy Model class; the rerun's parity
claim depends on them not drifting.
"""
from mend.strategies.critique_refine import _critique_prompt, _refine_prompt, _seed_prompt


def test_seed_prompt():
    assert _seed_prompt("DESC") == (
        "DESC"
        "Please provide Python code wrapped in triple backticks like:\n"
        "```python\n"
        "# your code here\n"
        "```\n\n"
    )


def test_critique_prompt_no_history():
    assert _critique_prompt("DESC", "PROG") == (
        "The following attempt did not pass all of its tests.\n\n"
        "Please explain what might be wrong.\n\n"
        "Task:\nDESC\n\n"
        "Program to Critique:\nPROG\n\n"
    )


def test_critique_prompt_with_history():
    prog = "Summary of previous attempts:\n..."
    assert _critique_prompt("DESC", prog) == (
        "The following attempt did not pass all of its tests.\n\n"
        "Please explain what might be wrong.\n\n"
        "Task:\nDESC\n\n"
        "Context (previous attempts and current program):\n" + prog + "\n\n"
    )


def test_refine_prompt():
    assert _refine_prompt("DESC", "PROG", "FB") == (
        "Task:\nDESC\n\n"
        "Current Program:\nPROG\n\n"
        "Feedback:\nFB\n\n"
        "Revise the program to address the feedback. "
        "Only return the corrected code."
    )


if __name__ == "__main__":
    test_seed_prompt()
    test_critique_prompt_no_history()
    test_critique_prompt_with_history()
    test_refine_prompt()
    print("ok: prompts pinned byte-for-byte to original")
