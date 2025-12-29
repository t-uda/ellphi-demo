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

## 3. Subtask 1.2: Gradient Correctness (Finite Difference Check)
**Goal**: Empirically prove the "Differentiability" claim. The analytical gradient *must* match the numerical gradient.

### Specifications
- **Script**: `task_1_validation.py` (Marimo notebook)
- **Dimensions ($n$)**: $\{2, 5, 10, 20, 50\}$
- **Perturbation ($\epsilon$)**: $10^{-6}$
- **Procedure**:
    1. Compute Analytical Gradient $\nabla_{ana}$ of the tangency time $t$ w.r.t. ellipsoid center $\mu_B$.
    2. Compute Numerical Gradient $\nabla_{num}$ via central difference: $\frac{t(\mu+\epsilon) - t(\mu-\epsilon)}{2\epsilon}$.
    3. Compute Relative Error: $E = \frac{\| \nabla_{ana} - \nabla_{num} \|_F}{\| \nabla_{ana} \|_F}$
- **Success Criterion**: $E < 10^{-5}$ (dominated by floating point error, not systematic error).
- **Output Artifacts**:
    - `gradient_verification.pdf`: Box plot of Log(Error) vs Dimension.
    - `gradient_scatter.png`: Scatter plot ($x=\nabla_{num}, y=\nabla_{ana}$) showing $y=x$ alignment.

## 4. Implementation Notes
- Adhere to `AGENTS.md` guidelines (use `uv run`).
- Ensure plots use Type 1 fonts (`pdf.fonttype: 42`) and Serif font family for Springer compatibility.
- Generated artifacts (plots, etc.) are saved in the task directory. These are ignored by git.
