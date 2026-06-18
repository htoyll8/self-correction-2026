#!/usr/bin/env python3
"""
Comprehensive error pattern analysis for APPS Competition result files.
For reviewer response: analyzing syntax errors, context truncation, error types.
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

RESULTS_DIR = Path("/Users/ritaholloway/Desktop/self-correction-2026/results")

FILES = {
    "gpt-4_competition": "results_gpt-4_apps_competition_critique-refine_2025-12-21_18-16-16.jsonl",
    "gpt-4o-mini_no-history": "results_gpt-4o-mini_apps_2025-12-13_00-19-57-critque-refine-competition-60.jsonl",
    "gpt-4o-mini_with-history": "results_gpt-4o-mini_apps_2025-12-12_18-58-21-critque-history-competitive-60.jsonl",
    "claude-sonnet": "results_claude-sonnet-4-5-20250929_apps_competition_critique-refine_2025-12-21_18-25-02.jsonl",
    "gpt-5.1_no-history": "results_gpt-5.1-2025-11-13_apps_2025-12-13_00-25-20-critque-refine-competitve-60.jsonl",
    "gpt-5.1_with-history": "results_gpt-5.1-2025-11-13_apps_2025-12-11_20-47-39-critque-history-competitive-60.jsonl",
}

def classify_error_detail(detail):
    """Classify an error detail string into a category."""
    if detail is None:
        return "Unknown"
    d = str(detail)
    if "SyntaxError" in d:
        return "SyntaxError"
    elif "IndentationError" in d:
        return "IndentationError"
    elif "NameError" in d:
        return "NameError"
    elif "TypeError" in d:
        return "TypeError"
    elif "ValueError" in d:
        return "ValueError"
    elif "AttributeError" in d:
        return "AttributeError"
    elif "IndexError" in d:
        return "IndexError"
    elif "KeyError" in d:
        return "KeyError"
    elif "RecursionError" in d or "maximum recursion" in d.lower():
        return "RecursionError"
    elif "MemoryError" in d:
        return "MemoryError"
    elif "ZeroDivisionError" in d:
        return "ZeroDivisionError"
    elif "ImportError" in d or "ModuleNotFoundError" in d:
        return "ImportError"
    elif "TimeoutError" in d or "timed out" in d.lower() or "Timeout" in d:
        return "TimeoutError"
    elif "RuntimeError" in d:
        return "RuntimeError"
    elif "StopIteration" in d:
        return "StopIteration"
    elif "OverflowError" in d:
        return "OverflowError"
    elif "EOFError" in d:
        return "EOFError"
    else:
        return "OtherError"

def is_truncated(program):
    """Heuristics for detecting a truncated/incomplete program."""
    if not program or len(program) < 10:
        return True, "empty_or_tiny"

    text = program.rstrip()
    last_50 = text[-50:] if len(text) >= 50 else text
    last_line = text.split('\n')[-1] if '\n' in text else text

    # Ends mid-string
    if re.search(r'["\'][^"\']*$', last_line):
        return True, "open_string"

    # Ends with operator or comma
    if re.search(r'[+\-*/%=,\(,\[,{,\\]\s*$', last_line.rstrip()):
        return True, "ends_with_operator_or_open"

    # Ends with 'def' or 'class' keyword
    if re.match(r'\s*(def|class|if|elif|else|for|while|try|except|finally|with|return|import|from)\s*$', last_line):
        return True, "ends_with_keyword"

    # Unbalanced brackets
    opens = program.count('(') + program.count('[') + program.count('{')
    closes = program.count(')') + program.count(']') + program.count('}')
    if opens > closes + 2:  # allow small imbalance
        return True, "unbalanced_brackets"

    # Very long last line with no statement end
    if len(last_line) > 200 and not last_line.rstrip().endswith((':',)):
        return True, "very_long_last_line"

    return False, None

def analyze_file(label, filename):
    filepath = RESULTS_DIR / filename
    print(f"\n{'='*80}")
    print(f"ANALYZING: {label}")
    print(f"File: {filename}")
    print(f"{'='*80}")

    tasks = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Total tasks: {len(tasks)}")

    # Classify trajectories
    # "failed_seed": a seed that never passes (neither initial nor any refinement)
    # "recovered_seed": seed that initially failed but eventually passed
    # "initial_pass": seed that passed on initial program

    failed_seeds = []        # never pass
    recovered_seeds = []     # initially failed, eventually passed
    initial_pass_seeds = []  # passed initially

    for task in tasks:
        for traj in task['trajectories']:
            if traj['initial_passed']:
                initial_pass_seeds.append(traj)
            else:
                # check if any refinement passed
                ever_passed = any(att['passed'] for att in traj['refinement_attempts'])
                if ever_passed:
                    recovered_seeds.append(traj)
                else:
                    failed_seeds.append(traj)

    total_seeds = len(failed_seeds) + len(recovered_seeds) + len(initial_pass_seeds)
    print(f"\nSeed breakdown (total: {total_seeds}):")
    print(f"  Initial pass:  {len(initial_pass_seeds)} ({100*len(initial_pass_seeds)/total_seeds:.1f}%)")
    print(f"  Recovered:     {len(recovered_seeds)} ({100*len(recovered_seeds)/total_seeds:.1f}%)")
    print(f"  Never passed:  {len(failed_seeds)} ({100*len(failed_seeds)/total_seeds:.1f}%)")

    # =========================================================
    # 1. Error type breakdown for FAILING seeds
    # =========================================================
    print(f"\n--- 1. ERROR TYPE BREAKDOWN (failing seeds only) ---")

    # Collect all test results from failing seeds (all attempts)
    status_counter = Counter()
    error_type_counter = Counter()
    all_error_details = []

    for traj in failed_seeds:
        # Initial attempt doesn't have test_results directly on traj;
        # check first refinement's feedback or look at initial program
        # Actually initial_passed=False so we need test results
        # The structure: traj has no direct test_results; refinement_attempts[0] is the first refinement
        # Let's collect all test results from refinement_attempts
        for att in traj['refinement_attempts']:
            if att.get('test_results'):
                for tr in att['test_results']:
                    tid, status, detail = tr[0], tr[1], tr[2] if len(tr) > 2 else None
                    status_counter[status] += 1
                    if status == 'ERROR':
                        etype = classify_error_detail(detail)
                        error_type_counter[etype] += 1
                        all_error_details.append(detail)

    total_results = sum(status_counter.values())
    print(f"\nTest result status counts (across all attempts of failing seeds):")
    print(f"  Total test results: {total_results:,}")
    for status in ['PASS', 'FAIL', 'TIMEOUT', 'ERROR']:
        cnt = status_counter[status]
        pct = 100*cnt/total_results if total_results > 0 else 0
        print(f"  {status:10s}: {cnt:8,}  ({pct:.1f}%)")

    total_errors = sum(error_type_counter.values())
    print(f"\nERROR breakdown (total ERROR results: {total_errors:,}):")
    for etype, cnt in error_type_counter.most_common():
        pct_of_errors = 100*cnt/total_errors if total_errors > 0 else 0
        pct_of_all = 100*cnt/total_results if total_results > 0 else 0
        print(f"  {etype:25s}: {cnt:6,}  ({pct_of_errors:.1f}% of errors, {pct_of_all:.1f}% of all)")

    # Sample some error details for other errors
    other_details = [d for d, e in zip(all_error_details,
                     [classify_error_detail(d) for d in all_error_details])
                     if e == 'OtherError']
    if other_details:
        print(f"\n  Sample 'OtherError' details (first 10):")
        for d in other_details[:10]:
            if d:
                print(f"    {str(d)[:120]}")

    # =========================================================
    # 2. Program length analysis
    # =========================================================
    print(f"\n--- 2. PROGRAM LENGTH ANALYSIS ---")

    def prog_lengths_seed(seeds, which='initial'):
        lengths = []
        for traj in seeds:
            if which == 'initial':
                prog = traj.get('initial_program', '')
                lengths.append(len(prog) if prog else 0)
            elif which == 'final':
                if traj['refinement_attempts']:
                    prog = traj['refinement_attempts'][-1].get('program', '')
                else:
                    prog = traj.get('initial_program', '')
                lengths.append(len(prog) if prog else 0)
        return lengths

    def stats(lengths):
        if not lengths:
            return "N/A"
        import statistics
        return f"mean={statistics.mean(lengths):.0f}, median={statistics.median(lengths):.0f}, min={min(lengths)}, max={max(lengths)}"

    for seed_type, seed_list in [("Failing seeds", failed_seeds),
                                   ("Recovered seeds", recovered_seeds),
                                   ("Initial-pass seeds", initial_pass_seeds)]:
        init_lens = prog_lengths_seed(seed_list, 'initial')
        final_lens = prog_lengths_seed(seed_list, 'final')
        print(f"\n  {seed_type} (n={len(seed_list)}):")
        print(f"    Initial program: {stats(init_lens)}")
        print(f"    Final program:   {stats(final_lens)}")

    # =========================================================
    # 3. Context/truncation signals
    # =========================================================
    print(f"\n--- 3. TRUNCATION / CONTEXT SIGNALS ---")

    # Look at programs that have ERROR status in their test results
    trunc_counter = Counter()
    trunc_programs_sample = []
    total_error_programs = 0
    truncated_error_programs = 0

    for traj in failed_seeds:
        for att in traj['refinement_attempts']:
            prog = att.get('program', '')
            if att.get('test_results'):
                has_error = any(tr[1] == 'ERROR' for tr in att['test_results'])
                if has_error:
                    total_error_programs += 1
                    is_trunc, reason = is_truncated(prog)
                    if is_trunc:
                        truncated_error_programs += 1
                        trunc_counter[reason] += 1
                        if len(trunc_programs_sample) < 5:
                            trunc_programs_sample.append((reason, prog[-100:] if prog else ''))

    print(f"\n  Programs with >=1 ERROR test result (failing seeds): {total_error_programs}")
    trunc_pct = 100*truncated_error_programs/total_error_programs if total_error_programs > 0 else 0
    print(f"  Programs flagged as possibly truncated: {truncated_error_programs} ({trunc_pct:.1f}%)")
    print(f"  Truncation reasons:")
    for reason, cnt in trunc_counter.most_common():
        print(f"    {reason:35s}: {cnt}")

    if trunc_programs_sample:
        print(f"\n  Sample truncated program endings (last 100 chars):")
        for reason, tail in trunc_programs_sample[:3]:
            print(f"    [{reason}] ...{repr(tail)}")

    # Also check all failing seed programs for very long programs (potential context issue)
    long_programs = []
    for traj in failed_seeds:
        for att in traj['refinement_attempts']:
            prog = att.get('program', '')
            if len(prog) > 8000:  # ~2000+ tokens typically
                long_programs.append(len(prog))

    if long_programs:
        import statistics
        print(f"\n  Programs > 8000 chars in failing seeds: {len(long_programs)}")
        print(f"  Lengths: mean={statistics.mean(long_programs):.0f}, max={max(long_programs)}")
    else:
        print(f"\n  Programs > 8000 chars in failing seeds: 0")

    # =========================================================
    # 4. Feedback content patterns (gpt-4o-mini only, but run for all)
    # =========================================================
    print(f"\n--- 4. FEEDBACK CONTENT PATTERNS (failing seeds) ---")

    feedback_patterns = {
        "syntax error": re.compile(r"syntax error", re.I),
        "indentation": re.compile(r"indentation", re.I),
        "name.*not defined": re.compile(r"name.*not defined|NameError", re.I),
        "type error": re.compile(r"TypeError|type error", re.I),
        "index.*out of range": re.compile(r"index.*out of range|IndexError", re.I),
        "time.*limit|timeout": re.compile(r"time.*limit|timeout|TLE", re.I),
        "wrong answer|incorrect": re.compile(r"wrong answer|incorrect output|expected.*got", re.I),
        "runtime error": re.compile(r"runtime error|RuntimeError", re.I),
        "value error": re.compile(r"ValueError|value error", re.I),
        "attribute error": re.compile(r"AttributeError|attribute error", re.I),
    }

    feedback_counts = Counter()
    total_feedbacks = 0

    for traj in failed_seeds:
        for att in traj['refinement_attempts']:
            fb = att.get('feedback', '') or ''
            total_feedbacks += 1
            for pattern_name, pattern in feedback_patterns.items():
                if pattern.search(fb):
                    feedback_counts[pattern_name] += 1

    print(f"\n  Total feedback messages in failing seeds: {total_feedbacks}")
    for pat, cnt in feedback_counts.most_common():
        pct = 100*cnt/total_feedbacks if total_feedbacks > 0 else 0
        print(f"  {pat:35s}: {cnt:5,} ({pct:.1f}%)")

    # =========================================================
    # 5. Test failure progression across attempts
    # =========================================================
    print(f"\n--- 5. ERROR/TIMEOUT/FAIL PROGRESSION ACROSS ATTEMPTS ---")

    # For failing seeds, track status distribution at attempt 1, 5, 10
    attempt_stats = defaultdict(lambda: Counter())

    for traj in failed_seeds:
        for att in traj['refinement_attempts']:
            attempt_num = att['attempt']
            if att.get('test_results'):
                for tr in att['test_results']:
                    status = tr[1]
                    attempt_stats[attempt_num][status] += 1

    print(f"\n  Status distribution by attempt number (failing seeds):")
    print(f"  {'Attempt':>8} | {'PASS':>8} {'FAIL':>8} {'TIMEOUT':>8} {'ERROR':>8} | {'Total':>8} | %ERROR | %TIMEOUT")
    print(f"  {'-'*85}")

    for att_num in sorted(attempt_stats.keys()):
        c = attempt_stats[att_num]
        total = sum(c.values())
        pct_err = 100*c['ERROR']/total if total > 0 else 0
        pct_to = 100*c['TIMEOUT']/total if total > 0 else 0
        print(f"  {att_num:>8} | {c['PASS']:>8,} {c['FAIL']:>8,} {c['TIMEOUT']:>8,} {c['ERROR']:>8,} | {total:>8,} | {pct_err:5.1f}% | {pct_to:5.1f}%")

    # Highlight key attempts
    for key_att in [1, 5, 10]:
        if key_att in attempt_stats:
            c = attempt_stats[key_att]
            total = sum(c.values())
            print(f"\n  Attempt {key_att} summary: total={total}, "
                  f"ERROR={c['ERROR']} ({100*c['ERROR']/total:.1f}%), "
                  f"TIMEOUT={c['TIMEOUT']} ({100*c['TIMEOUT']/total:.1f}%), "
                  f"FAIL={c['FAIL']} ({100*c['FAIL']/total:.1f}%), "
                  f"PASS={c['PASS']} ({100*c['PASS']/total:.1f}%)")

    return {
        'label': label,
        'n_tasks': len(tasks),
        'total_seeds': total_seeds,
        'initial_pass': len(initial_pass_seeds),
        'recovered': len(recovered_seeds),
        'failed': len(failed_seeds),
        'status_counter': dict(status_counter),
        'error_type_counter': dict(error_type_counter),
        'attempt_stats': {k: dict(v) for k, v in attempt_stats.items()},
    }


def main():
    results = {}
    for label, filename in FILES.items():
        try:
            results[label] = analyze_file(label, filename)
        except Exception as e:
            print(f"\nERROR analyzing {label}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================
    # Cross-model summary table
    # =========================================================
    print(f"\n\n{'='*80}")
    print("CROSS-MODEL SUMMARY TABLE")
    print(f"{'='*80}")

    print(f"\n{'Model':30s} | {'Tasks':>6} | {'Seeds':>6} | {'Init%':>6} | {'Recov%':>6} | {'Fail%':>6}")
    print(f"{'-'*80}")
    for label, r in results.items():
        ts = r['total_seeds']
        ip_pct = 100*r['initial_pass']/ts if ts > 0 else 0
        rc_pct = 100*r['recovered']/ts if ts > 0 else 0
        fa_pct = 100*r['failed']/ts if ts > 0 else 0
        print(f"{label:30s} | {r['n_tasks']:>6} | {ts:>6} | {ip_pct:>5.1f}% | {rc_pct:>5.1f}% | {fa_pct:>5.1f}%")

    print(f"\n{'Model':30s} | {'ERROR%':>7} | {'TIMEOUT%':>8} | {'FAIL%':>7} | {'PASS%':>7} | Top error type")
    print(f"{'-'*90}")
    for label, r in results.items():
        sc = r['status_counter']
        total = sum(sc.values())
        if total == 0:
            continue
        err_pct = 100*sc.get('ERROR',0)/total
        to_pct = 100*sc.get('TIMEOUT',0)/total
        fail_pct = 100*sc.get('FAIL',0)/total
        pass_pct = 100*sc.get('PASS',0)/total
        ec = r['error_type_counter']
        top_err = max(ec, key=ec.get) if ec else "N/A"
        print(f"{label:30s} | {err_pct:>6.1f}% | {to_pct:>7.1f}% | {fail_pct:>6.1f}% | {pass_pct:>6.1f}% | {top_err}")

    print(f"\n{'Model':30s} | {'SyntaxErr':>9} | {'IndentErr':>9} | {'NameErr':>9} | {'TypeError':>9} | {'RuntimeErr':>10} | {'OtherErr':>9}")
    print(f"{'-'*100}")
    for label, r in results.items():
        ec = r['error_type_counter']
        total_err = sum(ec.values())
        if total_err == 0:
            print(f"{label:30s} | (no errors)")
            continue
        def pct(k): return f"{100*ec.get(k,0)/total_err:.0f}% ({ec.get(k,0)})"
        print(f"{label:30s} | {pct('SyntaxError'):>9} | {pct('IndentationError'):>9} | {pct('NameError'):>9} | {pct('TypeError'):>9} | {pct('RuntimeError'):>10} | {pct('OtherError'):>9}")


if __name__ == '__main__':
    main()
