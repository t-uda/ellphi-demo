# Additional Experiments for APCT Paper (EllPHi n-dim Extension)

**Target Repository**: `EllPHi` (Python codebase)
**Target Branch**: `ellphi-0.1.1` (or `feat/ndim-extension-cpp` if not merged)
**Output Directory**: `experiments/apct_2025/` (Create if not exists)
**Figure Format**: PDF (Vector graphics required for Springer `sn-mathphys`)

## Overview
The APCT paper claims two major contributions:
1.  **N-dimensional generalization** of the ellipsoid tangency.
2.  **Differentiability** of the tangency time.

We need empirical evidence to support these claims in the "Experiments" section.

## Experiment 1: Computational Scalability vs Dimension
**Goal**: Demonstrate that the algorithm scales reasonably with dimension $n$ and that the C++ backend provides significant speedup.

-   **Script Name**: `benchmark_scaling.py`
-   **Parameters**:
    -   Dimensions $n \in \{2, 3, 5, 10, 20, 50, 100\}$.
    -   Number of pairs per dimension: $N=1000$.
    -   Backends: `python`, `cpp`.
-   **Procedure**:
    1.  Generate random ellipsoid pairs for each $n$.
    2.  Measure average execution time per tangency computation.
    3.  Compute speedup factor (Python / C++).
-   **Output Figure**: `figs/scaling_benchmark.pdf`
    -   **Type**: Line plot (log-log or semi-log).
    -   **X-axis**: Dimension $n$.
    -   **Y-axis**: Average Time (ms).
    -   **Series**: Python, C++.
    -   **Style**: Use `matplotlib` with `seaborn-whitegrid`. Ensure fonts are Type 1 (use `pdf.fonttype: 42`).

## Experiment 2: Gradient Correctness Verification
**Goal**: Empirically validate the "Differentiability" claim by comparing analytical gradients with numerical finite differences.

-   **Script Name**: `verify_gradients.py`
-   **Parameters**:
    -   Dimensions $n \in \{2, 5, 10, 20, 50\}$.
    -   Step size for finite difference: $\epsilon = 10^{-6}$.
-   **Procedure**:
    1.  For each $n$, generate random ellipsoid pairs.
    2.  Compute analytical gradients $\nabla \bar{t}$ w.r.t. centers and shape matrices using `differentiable_solver`.
    3.  Compute numerical gradients using central finite differences.
    4.  Calculate Relative Error: $\| \nabla_{ana} - \nabla_{num} \|_F / \| \nabla_{ana} \|_F$.
-   **Output Figure**: `figs/gradient_verification.pdf`
    -   **Type**: Box plot or Error bar plot.
    -   **X-axis**: Dimension $n$.
    -   **Y-axis**: Relative Gradient Error (log scale).
    -   **Expectation**: Error should be roughly $O(\epsilon^2)$ or dominated by machine precision, staying below $10^{-5}$.

## Experiment 3: Gradient Visualization (Differentiability Demo)
**Goal**: Visually demonstrate the differentiability of the tangency time $t$ with respect to ellipsoid parameters, serving as a conceptual proof-of-concept without performing full optimization.

-   **Script Name**: `demo_gradient_field.py`
-   **Scenario**:
    -   Fix Ellipsoid A at the origin in 2D.
    -   Vary the center of Ellipsoid B, $\bm x_B$, across a 2D grid around A.
    -   **Compute**: Tangency time $t$ and its gradient $\nabla_{\bm x_B} t$ at each grid point.
-   **Output Figure**: `figs/gradient_field_2d.pdf`
    -   **Type**: Contour plot of $t$ overlaid with a Quiver plot of $\nabla_{\bm x_B} t$.
    -   **X-axis / Y-axis**: Coordinates of $\bm x_B$.
    -   **Visual Elements**:
        -   **Contours**: Show the smooth "landscape" of the tangency time.
        -   **Arrows**: Show the gradient vectors $\nabla_{\bm x_B} t$.
    -   **Insight**: The smooth variation of contours and the continuous vector field visually confirm differentiability. The gradients should point in the direction of steepest increase (away from Ellipsoid A), providing clear geometric intuition.

## Implementation Notes for Agent
-   Use `poetry install` to set up the environment.
-   Ensure `matplotlib` is configured for publication-quality plots:
    ```python
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pdf.fonttype": 42
    })
    ```
-   Save all raw data (CSV/JSON) alongside figures in `experiments/apct_2025/data/`.
