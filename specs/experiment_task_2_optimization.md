# Experiment Task 2: Inverse Problem & Optimization Demo

**Objective**: Demonstrate the "Metric Learning" capability. Show that because the solver is differentiable, we can optimize ellipsoid parameters ($\Theta$) to maximize a topological signal (e.g., cycle persistence), thereby discovering the latent anisotropic structure of the data.

## 1. Environment & Setup
- **Output**: `experiments/apct_2025/optimization/`
- **Dependencies**: `ellphi`, `torch` or `jax` (if automatic differentiation integration is ready, otherwise manual gradient descent using the solver's gradient output).

## 2. Subtask 2.1: Gradient Field Visualization (Concept Demo)
**Goal**: Visualize the scalar field of Tangency Time $t$ and its gradient vector field to provide intuition.

### Specifications
- **Script**: `demo_gradient_field.py`
- **Setup**:
    - Fixed Ellipsoid A at origin.
    - Variable Ellipsoid B center $x_B$ on a dense 2D grid.
- **Output Artifact**: `figs/gradient_field_2d.pdf`
    - Contour plot of $t(x_B)$.
    - Quiver plot of $\nabla_{x_B} t$.
    - **Visual Check**: Gradients should be orthogonal to the level sets (contours) of $t$.

## 3. Subtask 2.2: The Inverse Problem (Metric Learning)
**Goal**: Recover the "Stretched" direction of a dataset by optimizing ellipsoid shapes.

### Specifications
- **Script**: `run_inverse_optimization.py`
- **Dataset (Ground Truth)**:
    - Points sampled from a stretched circle (Ellipse) in 2D or 3D.
    - Aspect ratio ~10:1.
    - Embed in noise if necessary to make isotropic detection hard (optional, keeping it clean is fine for proof-of-concept).
- **Optimization Loop**:
    1. **Init**: Initialize all ellipsoids as standard spheres (isotropic).
    2. **Forward**: Compute Persistence Diagram (Anisotropic VR).
    3. **Loss**: Identify the most significant $H_1$ cycle. Define Loss $L = -(\text{Death} - \text{Birth})$.
    4. **Backward**: Compute $\nabla_\Theta L$ using the chain rule and the solver's $\nabla_\Theta t$.
    5. **Update**: Update ellipsoid shape matrices using Gradient Descent.
- **metrics**:
    - Persistence of the target cycle over iterations.
    - Alignment angle between learned ellipsoids and the ground truth tangent.
- **Output Artifacts**:
    - `figs/optimization_history.pdf`: Loss/Persistence vs Iteration.
    - `figs/visual_evolution.png`: Snapshots (Start, Mid, End) of the ellipsoids overlaid on data. The ellipsoids should elongate along the cycle.

## 4. Implementation Notes
- This is the most complex task. Start with a simple 2D Case.
- You may perform the optimization using a simple loop in Python if a full PyTorch integration is overkill, as long as the gradients are correct.
