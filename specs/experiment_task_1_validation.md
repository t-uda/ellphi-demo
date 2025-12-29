# Experiment Task 1: Computational & Mathematical Validation

**Objective**: Rigorously validate the computational efficiency and mathematical correctness of the Differentiable Ellipsoid Tangency Solver. This forms the foundational "Trust" layer of the paper's experiments.

## 1. Environment & Setup
- **Target**: `ellphi` library (Python) with C++ backend options.
- **Tools**: Poetry environment, `matplotlib` (LaTeX mode), `numpy`.
- **Output**: `notebooks/task_1_validation/`

## 2. Subtask 1.1: Scalability Benchmark
**Goal**: Quantify the cost of dimension $n$ and the speedup of the C++ backend.

### Specifications
- **Script**: `task_1_validation.py` (Marimo notebook)
- **Dimensions ($n$)**: $\{2, 3, 5, 10, 20, 50, 100\}$
- **Sample Size**: $N=1000$ pairs per dimension.
- **Metrics**: Average execution time (ms) per tangency check.
- **Backends to Compare**:
    1. Pure Python implementation
    2. C++ Extension implementation
- **Output Artifact**: `scaling_benchmark.pdf`
    - Log-log plot: $X=n$, $Y=\text{Time (ms)}$.
    - Must show clear separation between Python and C++.

## 3. Subtask 1.2: Gradient Correctness (Pencil Parameter Sensitivity)
**Goal**: Empirically prove the correctness of the library's differentiable solver by verifying the gradients of the pencil parameter $\mu$ with respect to the full ellipsoid coefficients. This validates the "Trust Layer" and the `differentiable_solver` module.

### Specifications
- **Script**: `task_1_validation.py` (Marimo notebook)
- **Dimensions ($n$)**: $\{2, 3, 5, 10, 20\}$
- **Perturbation ($\epsilon$)**: $10^{-6}$
- **Procedure**:
    1. **Analytical Gradient**: Compute $\nabla_p \mu$ and $\nabla_q \mu$ using `ellphi.differentiable_solver.solve_mu_gradients`.
    2. **Numerical Gradient**: Compute $\nabla_{num}$ via central difference on `ellphi.solve_mu` with respect to every element of coefficient vectors $p$ and $q$.
    3. **Metric**: Relative Error $E = \frac{\| \nabla_{ana} - \nabla_{num} \|_F}{\| \nabla_{ana} \|_F}$
- **Success Criterion**: $E < 10^{-5}$.
- **Output Artifacts**:
    - `gradient_verification.pdf`: Box plot of Log(Error) vs Dimension.

## 4. Implementation Notes
- Adhere to `AGENTS.md` guidelines (use `uv run`).
- Ensure plots use Type 1 fonts (`pdf.fonttype: 42`) and Serif font family for Springer compatibility.
- Generated artifacts (plots, etc.) are saved in the task directory. These are ignored by git.
