## Repair Trees in Self-Repair


### Two Complementary Analyses

We perform **two complementary analyses**, each capturing a different aspect of self-repair.

**(1) Sequential “Rolling-Ball” Self-Repair.**
First, we construct *repair trees*, where each repair is conditioned on the previous program’s output and feedback—a process we call the *rolling-ball* model of self-repair. This captures the **true causal dynamics** of iterative refinement: every node represents a new model invocation that depends on the most recent failed attempt. As a result, programs within a single tree are **not independent**.

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

We evaluate self-repair on HumanEval-style benchmarks (e.g., **HumanEval-X**, **MBPP**, and **TransCoder** tasks).
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

---


Here’s a smoother, reviewer-friendly version that makes both analyses crystal clear and explains *why* each is needed. It keeps the NeurIPS tone but reads naturally:


---

