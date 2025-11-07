## Repair Trees in Self-Repair


### Two Complementary Analyses

We perform **two complementary analyses**, each capturing a different aspect of self-repair.

**(1) Sequential “Rolling-Ball” Self-Repair.**
First, we construct *repair trees*, where each repair is conditioned on the previous program’s output and feedback—--a process we call the *rolling-ball* model of self-repair. This captures the **true causal dynamics** of iterative refinement: every node represents a new model invocation that depends on the most recent failed attempt. As a result, programs within a single tree are **not independent**.

**(2) Bootstrapped i.i.d. Resampling.**
Second, we perform a **bootstrapped resampling** analysis to estimate *pass@k* efficiently across different generation budgets. The goal of this step is to approximate how success rates change with varying numbers of seeds, feedbacks, and repairs—--without having to regenerate new repair trees for every setting. To do this, we first build one large *master repair tree* for each task, containing
(N_p \ge n_p) initial seeds,
(N_f \ge n_f) feedback messages per failed seed, and
(N_r \ge n_r) repairs per feedback.
We then simulate smaller experiments by randomly sampling (n_p) seeds, (n_f) feedbacks, and (n_r) repairs (with replacement) from this frozen master tree.

This two-stage design serves **two purposes**:

1. The *rolling-ball* analysis measures how models actually behave under iterative, feedback-conditioned repair—capturing causal improvement dynamics.
2. The *bootstrapped resampling* analysis provides an efficient, *i.i.d.*-style estimate of pass@k (à la Chen et al., 2021), showing how success probability scales with available generation budget while remaining computationally feasible.


### Experimental Context

We evaluate self-repair on HumanEval-style benchmarks (e.g., **HumanEval-X**, **MBPP**, **APPs**, and **TransCoder** tasks).
Each repair trajectory is logged as a **tree** that captures all seeds, feedbacks, and refinements.

More specifically, each task’s JSONL log contains:

* `initial_passed`: whether the seed passed all tests
* `attempts`: nested repair attempts (with `feedback` and `passed` flags)
* `k`: total number of programs generated

We then aggregate across tasks to compute:

* Fraction of seeds that passed initially
* Mean and median repair attempts per seed
* pass@k values estimated via bootstrapping

---

### Conceptual Overview

Each top-level sample is no longer just a single program,
but an **entire repair tree**:

```math
T = M_P \circ M_F \circ M_P
```

Where:

* `M_P` — the **program generator**
* `M_F` — the **feedback model**
* `M_P` — the **repair model** that uses feedback

Each tree `T` begins with **`n_p`** initial programs.
For every failed one, the system produces **`n_f`** feedback messages,
and for each feedback message, it generates **`n_r`** repairs.

Hence, one tree contains approximately:

```math
|programs(T)| = n_p + n_p n_f n_r
```

total program samples.

---

### Intuition Behind ( |programs(T)| = n_p + n_p n_f n_r )

In standard program synthesis, each sample corresponds to a single, independent program.
However, once self-repair is introduced, each top-level sample becomes a **repair tree** —
a structured sequence of generations that includes both initial programs and their refinements.

Each tree (T) begins with (n_p) initial programs.
For every failed program, the system produces (n_f) feedback messages, and for each feedback message, it generates (n_r) repair attempts.
Thus, each initial seed can spawn up to (n_f n_r) descendant programs.

Consequently, the total number of programs within a single repair tree is:

```math
|programs(T)| = n_p + n_p n_f n_r
```

When evaluating self-repair performance (e.g., via pass@k),
each repair tree (T) is therefore treated as one *composite sample*
drawn from the joint model (M = (M_P \circ M_F \circ M_P)).

---

### Redefining *pass@k* for Self-Repair

In self-repair, we redefine *pass@k* **over repair trees rather than over individual programs**:

```math
\text{pass@k}_{\text{self-repair}} = P(\exists\, T_i \text{ such that some } p \in programs(T_i) \text{ passes all tests})
```

That is, the probability that one repair tree (T_i) contains a program that passes all tests.

If we generate (k) repair trees (T_1, T_2, \ldots, T_k),
each expanding internally into many program attempts, we evaluate:

```math
\text{pass@k} = P\Big(\exists\, T_i \in \{T_1, \ldots, T_k\} \text{ s.t. } \exists\, p \in programs(T_i) \text{ that passes}\Big)
```

**Intuitively:**

> Did any of the (k) repair trees succeed — regardless of how deep the fix occurred within the tree?

> **Interpretation:**
>
> Among the (k) sampled repair trees (T_1, T_2, \ldots, T_k),
> does there exist at least one program (p) inside any of those trees
> that passes all tests?

---

### Bootstrapped Evaluation of Self-Repair

Running self-repair end-to-end many times is **computationally expensive**,
because for every combination of hyperparameters ((n_p, n_f, n_r)),
you would need to independently generate a large number of full repair trees.

Instead, we estimate pass rates by **bootstrapping from a single large “master” repair tree** per task.

#### Procedure

1. **Generate one master repair tree per task**
   with large coverage parameters:

   ```math
   N_p ≥ n_p,\quad N_f ≥ n_f,\quad N_r ≥ n_r
   ```

2. **Subsample** smaller configurations (e.g., (n_p=5, n_f=2, n_r=1))
   from this frozen dataset with replacement.

3. **Repeat** this resampling (N_t) times (e.g., 1000 trials).

4. **Estimate** mean pass rate and 95% confidence interval:

   ```math
   \text{SE} = \sqrt{\frac{p(1-p)}{N_t}}, \quad
   95\%\,CI = [p - 1.96\,SE,\; p + 1.96\,SE]
   ```
---

### Example Tree (Markdown)

```
Seeds (N_p = 4)
├─ Seed 1 → PASS ✅   (no repair needed)
├─ Seed 2 → FAIL ❌
│   ├─ Feedback A
│   │   ├─ Repair A1 → FAIL ❌
│   │   └─ Repair A2 → PASS ✅
│   └─ Feedback B
│       ├─ Repair B1 → FAIL ❌
│       └─ Repair B2 → FAIL ❌
├─ Seed 3 → FAIL ❌
│   ├─ Feedback A
│   │   ├─ Repair A1 → FAIL ❌
│   │   └─ Repair A2 → FAIL ❌
│   └─ Feedback B
│       ├─ Repair B1 → FAIL ❌
│       └─ Repair B2 → PASS ✅
└─ Seed 4 → FAIL ❌
    ├─ Feedback A
    │   ├─ Repair A1 → FAIL ❌
    │   └─ Repair A2 → FAIL ❌
    └─ Feedback B
        ├─ Repair B1 → FAIL ❌
        └─ Repair B2 → FAIL ❌
```

---

### Repair-Tree Metrics

For each repair tree (T):

| Symbol | Meaning                           | Example Value       |
| :----- | :-------------------------------- | :------------------ |
| (n_p)  | Number of initial seeds           | 25                  |
| (n_f)  | Feedback messages per failed seed | 5                   |
| (n_r)  | Repairs per feedback              | 1                   |
| (k_T)  | Total programs in (T)             | (n_p + n_p n_f n_r) |

From logs we compute:

* **Initial pass rate:**
  [
  \frac{\text{# seeds that passed initially}}{n_p}
  ]

* **Repair success rate:**
  Fraction of initially failing seeds that eventually passed

* **Mean repairs per failing seed:**
  Average number of attempts before success

* **pass@k:**
  Aggregated task-level probability that at least one tree succeeded

---

### Bootstrapped *pass@k* (Formal Definition)

For each task with (n) total programs (seeds + repairs) and (c) correct ones:

```math
\text{pass@k} = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
```

This follows the unbiased estimator introduced in Chen et al. (2021, *Codex*).

---

### ?

- Is your iterative-refinement curve (most success in round 1, then plateau) just an artifact of difficulty — i.e., once the easy programs are solved, only hard ones remain, which naturally can’t improve much?
- To control for this, I filtered the APPS dataset by difficulty level — ‘introductory,’ ‘interview,’ and ‘competition’ — and ran separate experiments within each stratum.
- You mentioned that the iterative-refinement effect might just reflect a bimodal difficulty distribution — easy ones solved immediately, hard ones unsolvable. To rule that out, I ran the APPS dataset stratified by its built-in difficulty levels (‘introductory,’ ‘interview,’ and soon ‘competition’). The same diminishing-returns curve appears within each subset. So it’s not that the leftover tasks are just harder — it’s that the model’s refinement process itself fails to produce new insight after the first attempt.


### Implementation Details

* Each trajectory stored as JSON with fields:
  * `task_id`
  * `success`
  * `trajectories` (list of seeds, feedbacks, and attempts)
* pass@k computed for (k = 1, 5, 10) using the above estimator
* Results aggregated across all tasks to obtain:
  * Initial pass rate
  * Mean/median repairs per failed seed
  * pass@k curves and 95% confidence intervals

<!-- ---

📊 Summary Metrics
- Dataset: humaneval-x
- Unique tasks: 164
- Model: gpt-4o-mini
- Fraction passed initially: 0.82
- Fraction required repair: 0.18
- Mean attempts (failed seeds): 6.65
- Median attempts (failed seeds): 10.00

✅ Pass@k:
  pass@1 : 0.900
  pass@5 : 0.927
  pass@10: 0.927

📊 Average Percentage of Tests Passed (Failed Seeds Only):
  Initial     : 54.56% (144 programs)
  Refinement 1: 68.13% (144 programs)
  Refinement 2: 56.68% (104 programs)
  Refinement 3: 55.57% (98 programs)
  Refinement 4: 55.43% (94 programs)
  Refinement 5: 53.64% (89 programs)
  Refinement 6: 54.11% (88 programs)
  Refinement 7: 53.33% (87 programs)
  Refinement 8: 53.73% (86 programs)
  Refinement 9: 53.42% (84 programs)
  Refinement 10: 54.26% (83 programs)


- To be clear, Olausson et al. did not perform iterative refinement; they only did i.i.d. sampling of seeds and repeated refinement of the same seed.

Here’s the interpretation of the results from iterative refinement (max_attempts = 10):
  - 82% of seeds passed all tests immediately.
  - 18% of seeds failed and required repair.
  - 50% of the failing seeds exhausted all 10 attempts without achieving a perfect fix.
  - pass@5 = 0.927, meaning there’s a 92.7% probability that at least one of the five repair trees passed all tests.

- At the start, the failing seed programs were already partly correct—they passed about half of their tests on average (54.56%). Initial = 54.56% means that before any refinement, these failed seeds were already passing about half their tests on average. After the first round of refinement, performance improved sharply to 68.13%. However, after that initial improvement, progress stalled: by the second refinement, the average dropped to 56.68%, and from the third refinement onward, performance hovered around 53–55%.

✅ Successful trajectories (passed initially)
- Avg pass fraction: 100.00%
- Median pass fraction: 100.00%
- Avg attempts: 0.00
- Median attempts: 0.00
- Count: 676

🔁 Recovered trajectories (failed initially but later succeeded)
- Avg pass fraction: 74.95%
- Median pass fraction: 85.71%
- Avg attempts: 2.21
- Median attempts: 1.00
- Count: 62

- Average pass fraction ~75%, median ~86% means even early refinements were mostly correct.
- Average of ~2 refinement attempts (median 1) shows the model usually fixed them quickly, often on the first feedback loop.

❌ Failed trajectories (never passed)
- Avg pass fraction: 52.21%
- Median pass fraction: 57.14%
- Avg attempts: 10.00
- Median attempts: 10.00
- Count: 82

- Average and median attempts = 10 means most of these exhausted the full retry budget

- For a successful trajectory, it’s the average proportion of tests passed over the course of its entire refinement history (including the initial seed). It’s more like the mean test pass fraction across all attempts for each group.


📊 Summary Metrics
- Dataset: humaneval-x-python
- Unique tasks: 50
- Model: gpt-4o-mini
- Fraction passed initially: 0.73
- Fraction required repair: 0.27
- Mean attempts (failed seeds): 8.34
- Median attempts (failed seeds): 10.00

✅ Pass@k:
- pass@1 : 0.784
- pass@5 : 0.820
- pass@10: 0.820

📈 Average Percentage Passed per Iteration:
- Initial     : 76.80% (250 programs)
- Refinement 1: 19.12% (68 programs)
- Refinement 2: 8.47% (59 programs)
- Refinement 3: 6.43% (57 programs)
- Refinement 4: 4.85% (55 programs)
- Refinement 5: 6.67% (55 programs)
- Refinement 6: 6.67% (55 programs)
- Refinement 7: 5.45% (55 programs)
- Refinement 8: 7.88% (55 programs)
- Refinement 9: 4.94% (54 programs)
- Refinement 10: 5.56% (54 programs)

📊 Average Percentage of Tests Passed (Failed Seeds Only):
- Initial     : 14.71% (68 programs)
- Refinement 1: 19.12% (68 programs)
- Refinement 2: 8.47% (59 programs)
- Refinement 3: 6.43% (57 programs)
- Refinement 4: 4.85% (55 programs)
- Refinement 5: 6.67% (55 programs)
- Refinement 6: 6.67% (55 programs)
- Refinement 7: 5.45% (55 programs)
- Refinement 8: 7.88% (55 programs)
- Refinement 9: 4.94% (54 programs)
- Refinement 10: 5.56% (54 programs)

✅ Successful trajectories (passed initially)
- Avg pass fraction: 100.00%
- Median pass fraction: 100.00%
- Avg attempts: 0.00
- Median attempts: 0.00
- Count: 182

🔁 Recovered trajectories (failed initially but later succeeded)
- Avg pass fraction: 59.35%
- Median pass fraction: 66.67%
- Avg attempts: 1.93
- Median attempts: 1.00
- Count: 14

❌ Failed trajectories (never passed)
- Avg pass fraction: 5.11%
- Median pass fraction: 0.00%
- Avg attempts: 10.00
- Median attempts: 10.00
- Count: 54

📊 Summary Metrics
- Dataset: humaneval-x-cpp
- Unique tasks: 30
- Fraction passed initially: 0.67
- Fraction required repair: 0.33
- Mean attempts (failed seeds): 2.87
- Median attempts (failed seeds): 3.00

✅ Pass@k:
- pass@1 : 0.733
- pass@5 : 0.900
- pass@10: 0.900

📈 Average Percentage Passed per Iteration:
- Initial     : 68.15% (90 programs)
- Refinement 1: 7.78% (30 programs)
- Refinement 2: 11.49% (29 programs)
- Refinement 3: 16.05% (27 programs)

📊 Average Percentage of Tests Passed (Failed Seeds Only):
  Initial     : 4.44% (30 programs)
  Refinement 1: 7.78% (30 programs)
  Refinement 2: 11.49% (29 programs)
  Refinement 3: 16.05% (27 programs)

✅ Successful trajectories (passed initially)
- Avg pass fraction: 100.00%
- Median pass fraction: 100.00%
- Avg attempts: 0.00
- Median attempts: 0.00
- Count: 60

🔁 Recovered trajectories (failed initially but later succeeded)
- Avg pass fraction: 30.00%
- Median pass fraction: 0.00%
- Avg attempts: 2.33
- Median attempts: 2.50
- Count: 6

❌ Failed trajectories (never passed)
- Avg pass fraction: 5.56%
- Median pass fraction: 0.00%
- Avg attempts: 3.00
- Median attempts: 3.00
- Count: 24

----

EXPLANATION: https://chatgpt.com/s/t_690381b6e6508191a71c710fa14ab2c7
APPS DATASET!

📊 Summary Metrics

Dataset: APPs (difficulty=introductory)
Unique tasks: 30
Fraction passed initially: 0.23
Fraction required repair: 0.77
Mean attempts (failed seeds): 2.90
Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.300
  pass@5 : 0.433
  pass@10: 0.433


✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 21

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 75.97%
  Median pass fraction: 97.44%
  Avg attempts: 1.83
  Median attempts: 2.00
  Count: 6

❌ Failed trajectories (never passed)
  Avg pass fraction: 36.17%
  Median pass fraction: 25.42%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 63

📊 Summary Metrics
- LIMIT TO 30 TEST CASES.
results/results_gpt-4o-mini_apps_2025-10-30_13-35-51.jsonl
Dataset: APPs (difficulty=introductory)
Unique tasks: 30
Fraction passed initially: 0.28
Fraction required repair: 0.72
Mean attempts (failed seeds): 2.85
Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.356
  pass@5 : 0.400
  pass@10: 0.400

📈 Average Percentage Passed per Iteration:
  Initial     : 56.04% (90 programs)
  Refinement 1: 40.92% (65 programs)
  Refinement 2: 38.25% (61 programs)
  Refinement 3: 40.34% (59 programs)

  ✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 25

🔁 Recovered trajectories (failed initially but later succeeded)
- TODO: WHAT WAS THE INITIAL PASS RATE?
  Avg pass fraction: 72.96%
  Median pass fraction: 81.67%
  Avg attempts: 1.57
  Median attempts: 1.00
  Count: 7

❌ Failed trajectories (never passed)
  Avg pass fraction: 37.08%
  Median pass fraction: 30.00%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 58

📊 Summary Metrics
results/results_gpt-4o-mini_apps_2025-10-30_14-41-11.jsonl
Unique tasks: 30
Fraction passed initially: 0.36
Fraction required repair: 0.64
Mean attempts (failed seeds): 2.95
Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.389
  pass@5 : 0.500
  pass@10: 0.500

📈 Average Percentage Passed per Iteration:
  Initial     : 65.30% (90 programs)
  Refinement 1: 48.56% (58 programs)
  Refinement 2: 46.73% (57 programs)
  Refinement 3: 50.89% (56 programs)

✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 32

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 65.56%
  Median pass fraction: 86.67%
  Avg attempts: 2.00
  Median attempts: 2.00
  Count: 3

❌ Failed trajectories (never passed)
  Avg pass fraction: 47.35%
  Median pass fraction: 56.67%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 55

📊 Summary Metrics
results/results_gpt-3.5-turbo-0125_apps_2025-10-30_15-59-26.jsonl - "difficulty":"interview"
Unique tasks: 30
Fraction passed initially: 0.13
Fraction required repair: 0.87
Mean attempts (failed seeds): 2.72
Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.278
  pass@5 : 0.400
  pass@10: 0.400

✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 12

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 56.89%
  Median pass fraction: 63.33%
  Avg attempts: 1.31
  Median attempts: 1.00
  Count: 13

❌ Failed trajectories (never passed)
  Avg pass fraction: 34.14%
  Median pass fraction: 23.33%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 65

results/results_gpt-3.5-turbo-0125_apps_2025-10-30_19-23-57.jsonl - "difficulty":"introductory"


---
📊 Summary Metrics
- results/results_gpt-4o-mini_apps_2025-11-01_16-45-07.jsonl FOR python3 main.py --dataset apps --mode iterative --np 3 --max_attempts 3 --max_tasks 100 --difficulty introductory
- Unique tasks: 100
- Fraction passed initially: 0.34
- Fraction required repair: 0.66
- Mean attempts (failed seeds): 2.84
- Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.430
  pass@5 : 0.520
  pass@10: 0.520

📈 Average Percentage Passed per Iteration:
  Initial     : 56.69% (300 programs)
  Refinement 1: 37.08% (197 programs)
  Refinement 2: 34.99% (183 programs)
  Refinement 3: 34.43% (179 programs)

✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 103

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 76.71%
  Median pass fraction: 86.44%
  Avg attempts: 1.77
  Median attempts: 1.00
  Count: 26

❌ Failed trajectories (never passed)
  Avg pass fraction: 30.78%
  Median pass fraction: 23.33%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 171


------

📊 Summary Metrics
results/results_gpt-3.5-turbo-0125_apps_2025-11-01_22-42-54.jsonl FOR python3 main.py --dataset apps --mode iterative --np 3 --max_attempts 3 --max_tasks 100 --difficulty introductory --model_name gpt-3.5-turbo-0125
Unique tasks: 100
Fraction passed initially: 0.19
Fraction required repair: 0.81
Mean attempts (failed seeds): 2.86
Median attempts (failed seeds): 3.00

✅ Pass@k:
  pass@1 : 0.263
  pass@5 : 0.390
  pass@10: 0.390

📈 Average Percentage Passed per Iteration:
  Initial     : 39.97% (300 programs)
  Refinement 1: 31.73% (244 programs)
  Refinement 2: 26.93% (229 programs)
  Refinement 3: 26.48% (225 programs)

✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 56

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 61.90%
  Median pass fraction: 83.33%
  Avg attempts: 1.52
  Median attempts: 1.00
  Count: 23

❌ Failed trajectories (never passed)
  Avg pass fraction: 25.64%
  Median pass fraction: 16.67%
  Avg attempts: 3.00
  Median attempts: 3.00
  Count: 221

  --- 


  📊 Summary Metrics
  - results/results_gpt-5-2025-08-07_apps_2025-11-02_16-25-51.jsonl for python3 main.py --dataset apps --mode iterative --np 3 --max_attempts 3 --max_tasks 30 --difficulty introductory --model_name gpt-5-2025-08-07
  - Fraction passed initially: 0.16
  - Fraction required repair: 0.84
  - Mean attempts (failed seeds): 2.20
  - Median attempts (failed seeds): 3.00

✅ Pass@k:
 - pass@1 : 0.256 
 - pass@5 : 0.467
 - pass@10: 0.467

📈 Average Percentage Passed per Iteration:
  Initial     : 18.48% (90 programs)
  Refinement 1: 6.92% (67 programs)
  Refinement 2: 6.21% (59 programs)
  Refinement 3: 6.42% (41 programs)

  ✅ Successful trajectories (passed initially)
  Avg pass fraction: 100.00%
  Median pass fraction: 100.00%
  Avg attempts: 0.00
  Median attempts: 0.00
  Count: 14

🔁 Recovered trajectories (failed initially but later succeeded)
  Avg pass fraction: 36.00%
  Median pass fraction: 0.00%
  Avg attempts: 1.78
  Median attempts: 2.00
  Count: 9

❌ Failed trajectories (never passed)
  Avg pass fraction: 2.09%
  Median pass fraction: 0.00%
  Avg attempts: 2.25
  Median attempts: 3.00
  Count: 67
 -->

 Results streamed to results/results_gpt-4o-mini_apps_2025-11-03_09-21-10.jsonl for python3 main.py --dataset apps --mode iterative --np 3 --max_attempts 3 --max_tasks 30 --difficulty interview

### Quantitative Comparison Across Models (APPS — Introductory Difficulty)

| Metric | **GPT-4o-mini** | **GPT-3.5-Turbo-0125** | **GPT-5-2025-08-07** |
|:-------------------------------|:----------------:|:--------------------:|:----------------:|
| **Unique tasks** | 100 | 100 | 30 |
| **Fraction passed initially** | **34%** | 19% | 16% |
| **Fraction requiring repair** | 66% | 81% | 84% |
| **Mean attempts (failed seeds)** | 2.84 | 2.86 | 2.20 |
| **Median attempts (failed seeds)** | 3.00 | 3.00 | 3.00 |
| **Fraction recovered (of all seeds)** | **11%** | 8% | 5% |
| **pass@1** | **43%** | 26% | 26% |
| **pass@5** | **52%** | 39% | 47% |
| **pass@10** | **52%** | 39% | 47% |
| **Avg. pass fraction — Initial** | **56.69%** | 39.97% | 18.48% |
| **Avg. pass fraction — Refinement 1** | 37.08% | 31.73% | 6.92% |
| **Avg. pass fraction — Refinement 2** | 34.99% | 26.93% | 6.21% |
| **Avg. pass fraction — Refinement 3** | 34.43% | 26.48% | 6.42% |

---

### 📊 Quantitative Summary (APPS — Introductory Difficulty)

* **GPT-4o-mini** shows the **highest overall performance** across metrics:

  * Highest initial pass rate (**34%**) and highest pass@1 (**0.43**).
  * Strong pass@5 and pass@10 (**0.52**).
  * Maintains the **highest average pass fractions** at every refinement stage.

* **GPT-3.5-Turbo-0125** performs **moderately**, below GPT-4o-mini but above GPT-5:

  * Initial pass rate of **19%**, requiring repair in 81% of cases.
  * pass@1 (**0.26**) and pass@5 (**0.39**) lower than GPT-4o-mini.
  * Average pass fraction declines steadily from **40% → 27% → 26%** over refinements.

* **GPT-5-2025-08-07** has the **lowest performance** on this subset:

  * Initial pass rate only **16%**, with most tasks requiring repair (**84%**).
  * pass@1 (**0.26**) comparable to GPT-3.5, but smaller gains in pass@5 and pass@10 (**0.47**).
  * Very low average pass fractions: **18% initially → ~6% after refinements**.

* Across all models:

  * Median attempts for failed seeds remain fixed at 3.0, indicating that most failed trajectories exhaust all available refinement iterations without recovery.
  * The fraction of recovered trajectories decreases across models in this order: GPT-4o-mini (highest) → GPT-3.5-Turbo → GPT-5 (lowest). It’s actually surprising that the smaller model (GPT-4o-mini) recovered more failed trajectories than the larger GPT-5
  * **Refinement stages** generally show **declining or plateaued improvements**, with GPT-4o-mini maintaining higher absolute accuracy throughout.
  * **Recovered trajectories consistently achieve much higher average pass fractions** than those that never pass.
    * GPT-4o-mini:  76.7 % vs 30.8 %  (**+45.9 pp**)
    * GPT-3.5-Turbo:  61.9 % vs 25.6 %  (**+36.3 pp**)
    * GPT-5:  36.0 % vs  2.1 %  (**+33.9 pp**)
  * **Recovered trajectories also tend to converge early** (median = 1–2 attempts), whereas failed ones typically exhaust all 3 iterations.
  * GPT-4o-mini shows low mean variance (0.027) and very low median variance (0.0002), meaning seed performance is generally stable across tasks — most seeds perform similarly, with only rare outliers.
  GPT-3.5-Turbo has a slightly higher mean variance (0.033) and median variance (0.004), reflecting mildly greater stochasticity — small differences between seeds appear more often but remain limited in scope.
  GPT-5, however, shows a much higher mean variance (0.093) but a near-zero median variance (0.0001), indicating that while most tasks are consistent, a small subset exhibits extreme variability — some seeds succeed completely while others fail, leading to large seed-level spread concentrated in a few unstable tasks.

---

### Quantitative Summary — APPS (Interview Difficulty)

| Metric | **GPT-4o-mini** | **GPT-3.5-Turbo-0125** |
|:--------------------------------------------|:----------------:|:----------------:|
| **Unique tasks** | 30 | 30 |
| **Fraction passed initially** | **0.42** | 0.13 |
| **Fraction requiring repair** | 0.58 | 0.87 |
| **Fraction recovered (of all trajectories)** | 0.06 | **0.13** |
| **Mean attempts (failed seeds)** | 2.85 | 2.72 |
| **Median attempts (failed seeds)** | 3.00 | 3.00 |
| **pass@1** | **0.478** | 0.278 |
| **pass@5** | **0.567** | 0.400 |
| **pass@10** | **0.567** | 0.400 |
| **Avg. pass fraction — Initial** | **69.85%** | 39.26% |
| **Avg. pass fraction — Refinement 1** | **46.73%** | 41.03% |
| **Avg. pass fraction — Refinement 2** | **44.01%** | 37.84% |
| **Avg. pass fraction — Refinement 3** | **43.05%** | 37.53% |
| **Successful trajectories (passed initially)** | — | — |
| Avg. pass fraction | 100.00% | 100.00% |
| Median pass fraction | 100.00% | 100.00% |
| Avg. attempts | 0.00 | 0.00 |
| Median attempts | 0.00 | 0.00 |
| Count | 38 | 12 |
| **Recovered trajectories (failed initially but later succeeded)** | — | — |
| Avg. pass fraction | **78.33%** | 56.89% |
| Median pass fraction | **86.67%** | 63.33% |
| Avg. attempts | 1.40 | 1.31 |
| Median attempts | 1.00 | 1.00 |
| Count | 5 | 13 |
| **Failed trajectories (never passed)** | — | — |
| Avg. pass fraction | **43.39%** | 34.14% |
| Median pass fraction | **51.67%** | 23.33% |
| Avg. attempts | 3.00 | 3.00 |
| Median attempts | 3.00 | 3.00 |
| Count | 47 | 65 |


| Metric                           |  **Baseline 4o-mini**  | **4o-mini + History** |      Δ (Change)      |
| :------------------------------- | :--------------------: | :-------------------: | :------------------: |
| **Fraction passed initially**    |           34%          |          32%          |         −2 pp        |
| **Fraction requiring repair**    |           66%          |          68%          |         +2 pp        |
| **Recovered (failed → passed)**  | 11% (≈ 16.7% of fails) |   **16.7% of fails**  | **+5–6 pp absolute** |
| **Mean attempts (failed seeds)** |          2.84          |          2.73         |         −0.11        |
| **pass@1**                       |           43%          |       **43.7%**       |        +0.7 pp       |
| **pass@5**                       |           52%          |        **58%**        |       **+6 pp**      |
| **pass@10**                      |           52%          |        **58%**        |       **+6 pp**      |

- Adding history-aware feedback increased GPT-4o-mini’s effective repair success by ~50% (11 → 16.7 %) and boosted pass@k by ≈ 6 points — without increasing iteration cost.


## LARGE LANGUAGE MODELS CANNOT SELF-CORRECT REASONING YET

- "Central to our investigation is the notion of *intrinsic self-correction*, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback."
- "Internal feedback relies on the model's inherent knowledge and parameters to assess its outputs."
- Models:
  - gpt-3.5-turbo-0613
  - gpt-4 (accessed on 2023/08/29)
  - gpt-4-1106-preview (gpt-4-turbo)
- Three-stage Approach:
  - Prompt the model to perform an initial generation. 
  - Prompt the model to review its previous generation and produce feedback.
  - Prompt the model to answer the original questipn with the feedback.
- The paper assesses the "performance of various self-correction prompts for intrinsic self-correction."
- SETUP(s):
  - "To achieve this, we eliminate the use of labels, **requiring the LLM to independently determine when to stop the self-correction process**". WE HAVE TEST CASES THAT WE USE TO DETERMINE WHEN TO STOP CORRECTION.
  - Approach 0: Intrinsic feedback.
  - Approach 1: Self-consistency.
  - Approach 2: Multi-agent debate.
- FINDINGS:
  - "...the model is more likely to modify a correct answer to an incorrect one than to revise an incorrect answer to a correct one. *The fundamental issue is that LLMs cannot properly judge the correctness of their reasoning.*"
  - "The results indiciate both multi-agent debate and self-consistency achieve significant improvements over standard prompting."
    - Instead of relying on a single response from the LLM, self-consistency (as introduced by Wang et al., 2022) samples multiple independent reasoning paths for the same question. Then, the system takes a majority vote among the final answers.
  - Check Table 2 and 3.

## IS SELF-REPAIR A SILVER BULLET FOR CODE GENERATION?

- They’re just using the np, nf, nr sampling structure to estimate pass@k, not sequentially refining a single code candidate.
- They aren’t actually running iterative self-repair — instead, they sample a batch of programs, feedbacks, and repairs all at once to form a “repair tree.” They then compute pass@k over all programs in that tree (total count = nₚ + nₚ × n_f × n_r) and compare it to a baseline with the same number of i.i.d. samples.
  - First, they generate several initial programs (nₚ). Then, for each of those programs, they generate several feedback messages (n_f) based on test errors. Finally, for each feedback message, they generate several repaired versions of the program (n_r).
- These error messages either contain the compile/runtime error information or an example input on which the program’s output differs from the expected one. WE'RE DOING INTRINSIC FEEDBACK.
- THEY DON'T INCLUDE HISTORY.
- The experiment corresponds to the special case where (nₚ=3, n_f=1, n_r=1) — i.e., three initial seeds, each refined sequentially through one feedback–repair pair per iteration.
- HOW THEY DEAL WITH DIFFICULTY: "These tasks are proportionally sampled in accordance with the frequency of the different difficulty levels in
the broader APPS test set: 180 interview-level questions, 60 competition-level questions, and 60 introductorylevel questions. All tasks are listed in Appendix H."


## Ideas
- Self-consistency
- Generate table to compare with Chen et al. Table 2 and 3.
- Scope: Natural language to code.

## Additional Datasets
- "HumanEval-X [321]: HumanEval-X is developed for evaluating the multilingual ability of code generation models with 820 hand-writing data samples in C++, Java, JavaScript, and Go". **NOTE: HUMANEVAL-X INCLUDES MULTIPLE LANGUAGES"**.
- ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation.
- SciCode: A Research Coding Benchmark Curated by Scientists.
- USACO: USA Computing Olympiad
- SWE-bench Verified: Resolving Real-World GitHub Issues

---
- List: https://arxiv.org/pdf/2406.00515 
- "CoNaLa [297]: CoNaLa contains almost 597K data samples for evaluating **Python** code generation. The curated part of CoNaLa is crawled from Stack Overflow, automatically filtered, and then curated by annotators."
- "Spider [300]: Spider is large-scale complex **text-to-SQL** dataset covering 138 different domains."
- "CoderEval [299]: CoderEval is a pragmatic code generation benchmark that includes **230 Python** and **230 Java** code generation problems."
- "BigCodeBench [333]: BigCodeBench has 1,140 complex **Python** programming tasks, covering 723 function calls from 139 popular libraries across 7 domains. This benchmark is specifically designed to assess LLMs’ ability to call multiple functions from cross-domain libraries and follow complex instructions to solve programming tasks, helping to bridge the evaluation gap between isolated coding exercises and the real-world programming scenario."


----
"APPS is a dataset for evaluating model performance on Python programming tasks across three difficulty levels consisting of 1,000 at introductory, 3,000 at interview, and 1,000 at competition level."

- **Large Language Models Cannot Self-Correct Reasoning Yet**
  - The paper defines or coins *intrinsic self-correction*, wherein the model attempts to correct its initial response (e.g., seed program) based on its *inherent capabilities*—without external feedback.
  - The paper focuses on LLM ability to perform self-correction in the domain of reasoning.
  - **Motivation:** High-quality external feedback is often unavailable in many real-world applications.
  - **Motivation:** What are the intrinsic capabilities of LLMs in this domain?
  - External feedback includes input from humans, models, or tools (e.g., static analysis). Internal feedback, on the other hand, relies on the model’s inherent knowledge and parameters to reassess its outputs.
  - **Methodology:**  
    - **Models evaluated for intrinsic self-correction:**  
      - GPT-4-Turbo  
      - Llama-2  
    - A maximum of two rounds of self-correction is performed.  
    - **Three-stage prompting strategy:**  
      - Initial generation  
      - Prompt the model to review its previous response and generate feedback on it.  
      - Prompt the model to generate a new response for the task based on the feedback. 
    - In accordance with previous work, a *correct label* is used to determine when to stop the self-correction loop. They also evaluate performance when labels are not available and the LLM is tasked with determining when to stop the self-correction loop.
  - **Findings:**  
    - Determining how to prevent *mischanges* is key to effective self-correction.  
    - *LLMs struggle to self-correct their reasoning without external feedback.*  
  - **Future Work:**  
    - Suggested baseline: *leverage multiple model responses, such as self-consistency.*  
    - *It is important to include a complete task description in the prompt for generating initial responses, rather than leaving part of the task description for the feedback prompt.*  

- **Intervention**: Anti-unification to show the common structure of previous attempts.

- **Is Self-Repair a Silver Bullet for Code Generation?**
  - Olausson et al. evaluated the following models: Code Llama, GPT-3.5, and GPT-4.
  - The following datasets were evaluated: HumanEval and APPS. 
  - RQ: At an equivalent budget, does drawing code samples i.i.d. from the model have a higher chance of success.
  - Olausson et al. hypothesize that the bottleneck lies in the model's ability to generate feedback.
  - Scope: Self-contained Python programming tasks "with executable unit tests".
  - The paper makes the following observations: 
    - There are several instances where *"pass rates are higher or equally high with i.i.d sampling (without repair)."*
    - Sampling a diverse set of seed programs (initial programs) is likely the most effective way to use the sampling budget. 
    - *"Artificially boosting the quality of the feedback significantly improves the efficacy of self-repair."*
    - *Increasing the number of initial programs ($n_p$) consistently leads to relative performance gains for all models.*  
    - *Increasing $n_{rf}$ does not appear to be worth the additional cost incurred.* This suggests that, given a fixed budget, the most important factor determining whether self-repair will lead to a correct program is the diversity of the base samples generated up front.  
    - They find that *gains are larger on harder tasks*.  
  - Contributions: 
    - The models evaluated.
    - Research goal: What's the "significance" of a *textual feedback stage*.
  - Related Work: 
    - Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies, *2023*.
    - Coder Reviewer Reranking for Code Generation, *2022*.
    - Teaching Large Language Models to Self-Debug, *2023b*.
    - Check your facts and try again: Improving large language models with external knowledge and automated feedback, *2023*.
    - Self-Refine: Iterative Refinement with Self-Feedback, *2023*.
    - DeepDelta: Learning to Repair Compilation Errors., *2019*.
  - Methodology: 
    - The four stages of self-repair:
      - **Code generation:** A programming model $M_p$ generates $n_p$ samples i.i.d.
      - **Code execution:** The $n_p$ code samples are executed against a test suite.
      - **Feedback generation:** The model generates $n_f$ feedback strings for each incorrect program. The feedback is based on error messages (e.g., *"compile/runtime error information or an example input on which the program's output differs from the expected one.").
      - **Code repair:** "For each initial program $p_i$ and feedback $f_{ij}$, $n_r$ candidate repaired programs are samples from $M_p$".
    - On the APPS dataset, they evaluate a randomly chosen set of 300 tasks.  
    - The decoding temperature is set to 0.8 for all models.  
  - Research Questions:
    - Is self-repair more effective than i.i.d. sampling without repair?
    - Does a *"stronger"* feedback model improve the model's ability to perform self-repair?
    - Does having a human in the loop improve repair performance?

- **Teaching Large Language Models to Self-debug:**
  - The following datasets were evaluated: Spider, Transcoder (C++-to-Python), and MBPP.
  - The following models were evaluated: 
    - code-davinci-002
    - gpt-3.5-turbo
    - gpt-4
    - StarCoder
  - **Motivation:** Generating correct code in a single attempt is challenging.  
  - **Motivation:** Sampling multiple programs from a model increases the likelihood of generating correct code.
  - Related Work: 
    - Natural Language to Code Translation with Execution.
    - Lever: Learning to Verify Language-to-Code Generation with Execution.
  - **Methodology:**  
    - The model is prompted to investigate the execution results to identify implementation errors via few-shot prompting, without requiring any additional model training. The process resembles *rubber duck debugging.*  
    - The model selects the predicted code that produces the most frequent execution result among all candidates without execution errors.
    - Steps: 
      - Generate candidate programs.
      - **Explanation step:** ??
      - **Feedback step:** The model is tasked with explaining the generated code and comparing it to the task description. When unit tests are available, it also explains the intermediate execution steps line by line. 
      - The process terminates when the feedback indicates that the program is correct or when the maximum number of retry attempts is reached.
    - For initial candidate generation, the temperature is set to 0 to enable greedy decoding.  
    - All experiments use greedy decoding to generate code explanations, feedback messages, and new programs.  
    - A maximum of 10 retries is allowed.
  - **Findings:**  
    - Self-debugging outperforms baseline conditions that sample ten times as many predictions.
    - Incorporating execution trace feedback consistently enhances performance. 

- **Code Reviewer Reranking for Code Generation:**
  - Six datasets were evaluated.  
  - Eight models were evaluated:  
    - Codex-Davinci-002  
    - Codex-Davinci-001  
    - Codex-Cushman  
    - InCoder (1B, 6B)  
    - CodeGen (2B, 6B, 16B)
  - **Motivation:** Reranking performance often decreases as the number of candidate programs increases.  
  - **Motivation:** Reranking with a coder model tends to favor degenerate solutions.
  - **Contribution:**  
    - The CodeReviewer reranker favors program samples that exhibit high mutual information with the instruction.  
    - CodeReviewer is described as a *specific instantiation of the Maximum Mutual Information (MMI) objective*, which favors solutions with high mutual information with the instruction while down-weighting generic solutions (where p(y) is high).  
    - The approach uses prompting to create the Reviewer model, denoted as $p(x|y)$.  
  - **Goal:**  Using a pretrained code language model to generate code $y$ conditioned on natural language instructions $x$.  
- **Methodology:**  
  - After a program $y$ is generated by the Coder model $p(y|x)$, the order of the instruction $x$ and solution $y$ is inverted in the prompt. The pretrained language model is then queried again to estimate $p(x|y)$.  
  - The same pretrained model is used for both the Coder and Reviewer, but with different prompts.  
  - All datasets are sampled with a temperature of 0.4.  
  - They generate 125 programs for each problem, then repeatedly sample 25 of them (50 times) to estimate the average reranking accuracy.
    - To measure how stable and reliable their reranking method is, they use bootstrapping — that is, they randomly pick 25 programs out of the 125, do this 50 times, and compute the average accuracy across those 50 trials.  
  - Executability filtering is applied to all baseline and proposed methods. This process involves removing programs that produce runtime errors before applying any ranking methods.
- **Q&A:**  
  - What is a conditional language model, $p_{\theta}(y|c, x)$?  
  - What does it mean to sample autoregressively using temperature scaling?  
- **Tool:**  
  - `pyminifier` is used to remove comments and docstrings, and to replace all print and assertion messages with empty strings.  
- **Findings:**  
  - CodeReviewer reranking for code generation consistently outperforms Coder-only reranking.  
  - When combined with executability filtering, CodeReviewer can outperform the MBR-EXEC method.  


- **Approach:**  
  - I aim to improve iterative LLM self-correction for coding tasks by representing the history of prior attempts in a compact yet effective way. Specifically, I plan to explore using anti-unification to cluster and generalize previous program attempts.  
    - It could even form the foundation of a “structural memory” for LLM repair agents — letting them reason over families of prior fixes instead of isolated samples.
- Following X, the temperature for seed program generation is set to Y.
- Following, X the temperature for feedback generation and refinment is set to Y. 
- I ALSO HAVE TO REMEMBER THAT THE HISTORY IS ROLLING LIKE A NEW CANDIDATE ATTEMPT IS ADDED AS THE ITERATIONS GO ONE. You’re not just anti-unifying a static set of programs; you’re doing it over a temporal sequence — a rolling history that grows with each iteration.
- Key goal: 
  - Maintain a **compact structural summary** of everything learned so far —not by keeping every \( P_i \), but by **anti-unifying incrementally** as new attempts arrive.
  - So at time \( t \), you don’t store all programs — you store: \[
    G_t = \text{antiunify}(G_{t-1}, P_t)
    \] where \( G_0 = P_0 \).

  - This \( G_t \) is your **rolling generalized program** — a symbolic compression of the repair trajectory so far.

### Anti-Unification and Invariants

- **Definition:**  
  Anti-unification takes two concrete programs \( P_1 \) and \( P_2 \) and returns their **least general generalization (LGG)** — the most specific program schema from which both programs can be derived by substitution.

- **Notation:**  
  The anti-unification operator is defined as:  
  \[
  \text{AU}(P_1, P_2) = G
  \]

- **Constraint Template:**  
  \( G \) defines the relationship between the programs as:  
  \[
  \forall \sigma_1, \sigma_2.\; G\sigma_1 = P_1 \wedge G\sigma_2 = P_2
  \] 
  - The shared structure of \( G \) corresponds to **invariant syntactic forms**.  
  - The metavariables correspond to **non-invariant (variant) sites**.
  - **Intuition:** You can think of the anti-unified form \( G \) as an **invariant schema** over the repair trajectory — capturing what stays stable across refinements and isolating where change occurs.


  - INVARIANTS?
  - MAP THE DIFFERENCE TO EXECUTION... LIKE DELTA DEBUGGING?
  - CREATE AN INTERFACE FOR THE LLM THAT SHOWCASES THE INVARINATS AND DIFF BETWEEN CANDIDATE PROGRAMS.
