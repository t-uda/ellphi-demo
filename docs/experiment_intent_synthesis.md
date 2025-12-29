# Additional Experiments: Strategic Intent and Mathematical Synthesis

This document defines the strategic and mathematical intent behind the additional experiments for the APCT paper. It serves as the unified theoretical basis for the execution tasks.

## 1. Core Strategic Value
The primary contribution of this work is **not just the capability to compute** anisotropic persistent homology, but the guarantee of **differentiability**.

*   **Differentiation $\rightarrow$ Optimization**: The ability to compute gradients $\nabla_\Theta L$ of a topological loss function $L$ with respect to ellipsoid parameters $\Theta$ allows us to **optimize the metric** itself.
*   **Metric Learning for TDA**: This paves the way for "Topological Metric Learning," where the optimal anisotropy is learned from the data to maximize topological structural clarity.
*   **End-to-End Learning**: It enables the integration of TDA into Deep Learning pipelines with backpropagation.

## 2. Mathematical Intent & Verification Logic

### 2.1. Fundamental Validation (The "Trust" Layer)
Before claiming advanced capabilities, we must prove the solver is numerically robust and theoretically consistent.
*   **Logic**: If the solver is differentiable, the numerical gradient (Finite Difference) must match the analytical gradient to high precision (approx $O(\epsilon^2)$).
*   **Requirement**: Error $\|\nabla_{ana} - \nabla_{num}\|_F / \|\nabla_{ana}\|_F < 10^{-5}$.
*   **Scalability**: We must show that the $O(n^2)$ or $O(n^3)$ cost of the underlying matrix operations does not make high-dimensional analysis ($n=100$+) prohibitive.

### 2.2. The Inverse Problem (The "Novelty" Layer)
This is the "killer application" experiment ensuring the paper's impact.
*   **Hypothesis**: IF the data has a latent anisotropic structure (e.g., a cycle stretched by 10x in one direction), THEN optimizing the ellipsoid aspect ratios to maximize the cycle's persistence should recover this stretch direction.
*   **Mathematical Object**: a Loss function $L(\Theta) = -(\text{Death} - \text{Birth})_{H_1^*}$, where $H_1^*$ is the most persistent cycle.
*   **Ground Truth**: A stretched circle in $\mathbb{R}^2$ or $\mathbb{R}^3$.
*   **Success Criterion**: The optimized ellipsoids align with the tangent of the underlying stretched manifold.

### 2.3. Robustness (The "Utility" Layer)
Demonstrating practical superiority over standard Euclidean TDA.
*   **Scenario**: Two parallel line segments close to each other.
*   **Failure Mode (Standard)**: As observation noise increases, the "gap" between lines is bridged, merging them topologically (two components $\to$ one).
*   **Advantage (Anisotropic)**: Optimized ellipsoids should elongate along the lines, maintaining the separation (the "gap") even under higher noise levels.

## 3. Structure of Experiment Instructions
The execution plans are separated into three independent logical units for AI Agents:

1.  **[Task 1] Computational & Mathematical Validation**:
    *   Focus: Scalability, Differentiation Correctness.
    *   Detail: See `experiment_task_1_validation.md`.
2.  **[Task 2] Inverse Problem & Optimization Demo**:
    *   Focus: Metric Learning, Gradient Visualization.
    *   Detail: See `experiment_task_2_optimization.md`.
3.  **[Task 3] Robustness Comparison**:
    *   Focus: Noise tolerance vs Euclidean VR.
    *   Detail: See `experiment_task_3_robustness.md`.
