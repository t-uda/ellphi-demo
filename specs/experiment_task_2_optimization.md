# Experiment Task 2: Inverse Problem & Optimization Demo

**Objective**: Demonstrate the "Metric Learning" capability. Show that because the solver is differentiable, we can optimize ellipsoid parameters ($\Theta$) to maximize a topological signal (e.g., cycle persistence), thereby discovering the latent anisotropic structure of the data.

## 1. Environment & Setup
- **Dependencies**: `ellphi` (>= 0.1.2), `numpy`, `scipy`, `ripser`, `matplotlib`.
- **AD framework integration**: Examples using `torch` and/or `jax` custom ops wrapping ellphi's VJP are desirable as separate notebooks (see Subtask 2.4).
- **Gradient API**: `ellphi.grad.coef_from_cov_grad`, `ellphi.grad.pdist_tangency_grad`.

## 2. Subtask 2.1: Gradient Field Visualization (Concept Demo)
**Goal**: Visualize the scalar field of Tangency Time $t$ and its gradient vector field to provide intuition.

**Status**: Done.

### Specifications
- **Notebook**: `notebooks/gradient_field/gradient_field.py` (marimo)
- **Setup**:
    - Fixed Ellipsoid A at origin.
    - Variable Ellipsoid B center $x_B$ on a dense 2D grid.
- **Output Artifact**: `gradient_field_2d.pdf`
    - Contour plot of $t(x_B)$.
    - Quiver plot of $\nabla_{x_B} t$.
    - **Visual Check**: Gradients should be orthogonal to the level sets (contours) of $t$.

## 3. Subtask 2.2: Coverage Optimization (Placement Demo)
**Goal**: Demonstrate gradient-based optimization of ellipsoid **placement and rotation** to minimize H1 total persistence in a sensor coverage scenario.

**Status**: Done.

This subtask was not in the original spec but was implemented as a concrete first application of the differentiable solver. It demonstrates the VJP chain (`coef_from_cov_grad` → `pdist_tangency_grad` → H1 subgradient) with L-BFGS-B, and compares anisotropic (3N DOF) vs isotropic (2N DOF) performance.

### Implementation
- **Notebook**: `notebooks/anisotropic_coverage/anisotropic_coverage.py` (marimo)
- **Key features**:
    - Analytical VJP chain (no finite differences)
    - H1 total persistence as the direct objective (persistence subgradients)
    - Analytical $d\Sigma/d\theta$ chain rule for rotation parameters
    - Comparison against equal-area isotropic sensors
- **Output Artifacts**: `anisotropic_optimization.pdf`, `anisotropic_vs_isotropic.pdf`

### Distinction from the Inverse Problem
This demo optimizes **positions and orientations** with fixed ellipsoid shape (semi-axes). It does **not** optimize the ellipsoid shape itself, and therefore does not address the core "metric learning" / inverse problem described in Subtask 2.3.

## 4. Subtask 2.3: The Inverse Problem (Metric Learning)
**Goal**: Recover the "Stretched" direction of a dataset by optimizing ellipsoid **shapes** (covariance matrices).

**Status**: Done.

### Specifications
- **Notebook**: `notebooks/inverse_problem/inverse_problem.py` (marimo)
- **Dataset (Ground Truth)**:
    - Points sampled from a stretched circle (Ellipse) in 2D or 3D.
    - Aspect ratio ~10:1.
    - Embed in noise if necessary to make isotropic detection hard (optional, keeping it clean is fine for proof-of-concept).
- **Optimization Loop**:
    1. **Init**: Initialize all ellipsoids as standard spheres (isotropic).
    2. **Forward**: Compute Persistence Diagram (Anisotropic VR).
    3. **Loss**: Identify the most significant $H_1$ cycle. Define Loss $L = -(\text{Death} - \text{Birth})$.
    4. **Backward**: Compute $\nabla_\Theta L$ using the VJP chain (`coef_from_cov_grad` → `pdist_tangency_grad`) and persistence subgradients.
    5. **Update**: Update ellipsoid shape matrices (covariance) using gradient descent or L-BFGS-B.
- **Metrics**:
    - Persistence of the target cycle over iterations.
    - Alignment angle between learned ellipsoids and the ground truth tangent.
- **Output Artifacts**:
    - `optimization_history.pdf`: Loss/Persistence vs Iteration.
    - `visual_evolution.pdf`: Snapshots (Start, Mid, End) of the ellipsoids overlaid on data. The ellipsoids should elongate along the cycle.

## 5. Subtask 2.4: AD Framework Integration Examples
**Goal**: Provide examples of wrapping ellphi's VJP into `torch` and `jax` custom ops for end-to-end differentiable pipelines.

**Status**: Done.

### Specifications
- **torch example**: Wrap `coef_from_cov_grad` + `pdist_tangency_grad` as a `torch.autograd.Function`. Demonstrate backpropagation through tangency distances.
- **jax example**: Wrap via `jax.custom_vjp`. Demonstrate `jax.grad` through the tangency computation.
- **Notebooks**: `notebooks/torch_integration/` and `notebooks/jax_integration/` (marimo).

## 6. Implementation Notes
- Use `uv run` for all execution.
- Start with Subtask 2.3 (inverse problem) as the highest-impact remaining item.
- Subtask 2.1 (gradient field) is a good visual complement but lower priority.
- Subtask 2.4 (torch/jax) depends on the VJP API being stable.
