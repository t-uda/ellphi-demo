import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task 1: Computational & Mathematical Validation"
    """)
    return


@app.cell
def _():
    import time

    import ellphi
    from ellphi.differentiable_solver import solve_mu_gradients, solve_mu_numerical_diff
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from numpy.linalg import inv, norm
    from tqdm import tqdm
    return (
        ellphi,
        inv,
        mo,
        norm,
        np,
        pd,
        plt,
        sns,
        solve_mu_gradients,
        solve_mu_numerical_diff,
        time,
        tqdm,
    )


@app.cell(hide_code=True)
def _(np, plt):
    # Configure matplotlib for publication quality
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"

    # Set random seed for reproducibility
    np.random.seed(42)


    return


@app.cell
def _(np):
    def generate_random_ellipsoid(dim):
        """Generates a random ellipsoid defined by mean and covariance."""
        mean = np.random.randn(dim)
        # Generate random covariance matrix
        A = np.random.randn(dim, dim)
        cov = A @ A.T + np.eye(dim) * 0.1  # Ensure positive definite
        return mean, cov
    return (generate_random_ellipsoid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subtask 1.1: Scalability Benchmark

    **Objective**: Quantify the computational cost of the tangency solver as a function of dimension $n$ and verify the performance gain of the C++ backend.

    **Methodology**:
    - **Dimensions**: $n \in \{2, 3, 5, 10, 20, 50, 100\}$
    - **Sample Size**: 1,000 random ellipsoid pairs per dimension.
    - **Backends**: Comparison between Pure Python and C++ Extension (if available).
    """)
    return


@app.cell
def _(ellphi, generate_random_ellipsoid, pd, time, tqdm):
    def run_benchmark():
        dims = [2, 3, 5, 10, 20, 50, 100]
        n_samples = 1000
        results = []

        backends = ["python", "cpp"]

        for dim in tqdm(dims, desc="Benchmarking dimensions"):
            # Pre-generate data to just measure solver time
            pairs = []
            for _ in range(n_samples):
                m1, c1 = generate_random_ellipsoid(dim)
                m2, c2 = generate_random_ellipsoid(dim)
                pcoef = ellphi.coef_from_cov(m1, c1)
                qcoef = ellphi.coef_from_cov(m2, c2)
                pairs.append((pcoef, qcoef))

            for backend in backends:
                start_time = time.time()
                for pcoef, qcoef in pairs:
                    ellphi.tangency(pcoef, qcoef, backend=backend)
                end_time = time.time()
                avg_time_ms = (end_time - start_time) / n_samples * 1000
                results.append(
                    {
                        "Dimension": dim,
                        "Backend": backend,
                        "Time (ms)": avg_time_ms,
                    }
                )

        return pd.DataFrame(results)

    df_benchmark = run_benchmark()
    df_benchmark
    return (df_benchmark,)


@app.cell
def _(df_benchmark, plt):
    # Plotting Subtask 1.1
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    # Check if we have multiple backends
    for backend, group in df_benchmark.groupby("Backend"):
        ax1.loglog(
            group["Dimension"],
            group["Time (ms)"],
            marker="o",
            label=f"Backend: {backend}",
        )

    ax1.set_xlabel("Dimension ($n$)")
    ax1.set_ylabel("Time per Check (ms)")
    ax1.set_title("Scalability Benchmark: Ellipsoid Tangency")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    plt.tight_layout()
    plt.savefig("scaling_benchmark.pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subtask 1.2: Gradient Correctness (Pencil Parameter Sensitivity)

    **Objective**: Rigorously validate the correctness of the analytical gradients computed by `ellphi.differentiable_solver`. Instead of just the tangency time $t$ with respect to the center, we verify the gradient of the pencil parameter $\mu$ with respect to *all* ellipsoid coefficients (quadratic, linear, and constant terms).

    **Methodology**:
    - **Procedure**: Compare **Analytical Gradient** ($\nabla_p \mu, \nabla_q \mu$) from `solve_mu_gradients` vs. **Numerical Gradient** from `solve_mu_numerical_diff` (Central Finite Difference).
    - **Perturbation**: $\epsilon = 10^{-6}$
    - **Success Criterion**: Relative Error $\frac{\|\nabla_{\text{ana}} - \nabla_{\text{num}}\|_F}{\|\nabla_{\text{ana}}\|_F} < 10^{-5}$.
    """)
    return


@app.cell
def _(
    ellphi,
    generate_random_ellipsoid,
    norm,
    np,
    pd,
    plt,
    sns,
    solve_mu_gradients,
    solve_mu_numerical_diff,
    tqdm,
):
    def verify_gradients():
        dims = [2, 3, 5, 10, 20]
        epsilon = 1e-6
        results = []

        # For scatter plot
        scatter_data = {"num": [], "ana": []}

        for dim in tqdm(dims, desc="Verifying Gradients"):
            # Use 10 samples per dimension for statistics
            for _ in range(10):
                m1, c1 = generate_random_ellipsoid(dim)
                m2, c2 = generate_random_ellipsoid(dim)

                # Coefficients
                p = ellphi.coef_from_cov(m1, c1).flatten()
                q = ellphi.coef_from_cov(m2, c2).flatten()

                # Analytical Gradient
                mu_val, grad_p_ana, grad_q_ana = solve_mu_gradients(p, q)

                # Numerical Gradient (using library's numerical diff for consistency)
                grad_p_num, grad_q_num = solve_mu_numerical_diff(p, q, h=epsilon)

                # Combine for comparison
                grad_ana = np.concatenate([grad_p_ana, grad_q_ana])
                grad_num = np.concatenate([grad_p_num, grad_q_num])

                # Compare
                # Relative Error: ||ana - num|| / ||ana||
                diff_norm = norm(grad_ana - grad_num)
                ana_norm = norm(grad_ana)

                if ana_norm < 1e-12:
                    rel_error = 0.0
                else:
                    rel_error = diff_norm / ana_norm

                results.append(
                    {
                        "Dimension": dim,
                        "Relative Error": rel_error,
                        "Log10 Error": np.log10(rel_error + 1e-20),
                    }
                )

                scatter_data["num"].extend(grad_num.tolist())
                scatter_data["ana"].extend(grad_ana.tolist())

        return pd.DataFrame(results), pd.DataFrame(scatter_data)


    df_grads, df_scatter = verify_gradients()

    # Plot Box Plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_grads, x="Dimension", y="Log10 Error", ax=ax2)
    ax2.axhline(
        np.log10(1e-5), color="r", linestyle="--", label="Threshold ($10^{-5}$)"
    )
    ax2.set_title("Gradient Verification: Relative Error Distribution")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("gradient_verification.pdf")
    plt.show()
    return



if __name__ == "__main__":
    app.run()
