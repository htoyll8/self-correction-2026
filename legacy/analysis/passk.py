# =========================================================
# PSEUDOCODE — Bootstrapped pass@k estimation with one
# large master repair tree per task (as in ICLR 2024 paper)
# =========================================================

# -------------------------------
# Step 1. Generate a large master tree for each task
# -------------------------------

for task in TASKS:
    # Build one "frozen" dataset of all possible generations
    # (very large coverage to allow later subsampling)
    T_master = generate_repair_tree(
        task=task,
        Np=50,    # many initial programs
        Nf=25,    # many feedback messages per failed program
        Nr=1      # one repair per feedback (joint sampling)
    )
    save(T_master)

# -------------------------------
# Step 2. Later, simulate smaller hyperparameter settings
# -------------------------------

# define experiment hyperparameters
configurations = [
    (np=5, nf=1, nr=1),
    (np=10, nf=2, nr=1),
    (np=25, nf=5, nr=1),
    # ... etc.
]

Nt = 1000  # number of bootstrap resamples per configuration

results = []

for (np, nf, nr) in configurations:
    pass_counts = []  # store binary outcomes (1=success, 0=failure)

    # -------------------------------
    # Step 3. Bootstrapping loop
    # -------------------------------
    for trial in range(Nt):

        # randomly draw (with replacement) a sub-repair-tree
        # from the large master tree for each task
        sub_trees = []
        for task in TASKS:
            T_master = load(task)
            T_sub = subsample_tree(
                T_master,
                np=np, nf=nf, nr=nr,  # desired scale
                with_replacement=True
            )
            sub_trees.append(T_sub)

        # Evaluate "did any tree succeed?" (i.e., pass@k logic)
        success = any(tree_contains_passing_program(T) for T in sub_trees)
        pass_counts.append(1 if success else 0)

    # -------------------------------
    # Step 4. Aggregate results
    # -------------------------------
    pass_rate = mean(pass_counts)
    std_error = sqrt(pass_rate * (1 - pass_rate) / Nt)
    conf_int = (
        pass_rate - 1.96 * std_error,
        pass_rate + 1.96 * std_error
    )

    results.append({
        "np": np,
        "nf": nf,
        "nr": nr,
        "pass_rate": pass_rate,
        "95CI": conf_int
    })

# -------------------------------
# Step 5. Plot pass@k curves
# -------------------------------

plot_pass_rate_vs_k(results)
