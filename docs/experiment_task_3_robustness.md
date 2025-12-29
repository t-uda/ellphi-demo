# Experiment Task 3: Robustness Comparison

**Objective**: Demonstrate the practical superiority of Anisotropic PH over standard Isotropic (Euclidean) PH in noisy scenarios. Show that answering "What is the shape?" requires adapting the metric to the local geometry.

## 1. Environment & Setup
- **Output**: `experiments/apct_2025/robustness/`
- **Comparison Baseline**: Standard Rips Filtration (Euclidean distance).

## 2. Subtask 3.1: Topological Noise Tolerance
**Goal**: Show that Anisotropic PH maintains topological features (cycles/components) at higher noise levels than Isotropic PH.

### Specifications
- **Script**: `compare_robustness.py`
- **Dataset (The "Close Strands" Problem)**:
    - Two parallel line segments or curves in close proximity.
    - Gap size $\delta$.
- **Variable**: Gaussian Noise level $\sigma$.
- **Procedure**:
    1. **Isotropic Case**: Run standard VR. Measure the noise level $\sigma_{iso}$ at which the two components merge into one (H0 analysis) or the loop collapses (H1) prematurely.
    2. **Anisotropic Case**: Use ellipsoids optimized/aligned along the strands (can be pre-aligned for this robustness check, or optimized as in Task 2). Measure the breakdown noise level $\sigma_{aniso}$.
- **Hypothesis**: $\sigma_{aniso} > \sigma_{iso}$. The anisotropic metric "bridges" the gaps *along* the data while preserving the gap *between* the strands.
- **Output Artifacts**:
    - `figs/robustness_comparison.pdf`:
        - Plot "Bottleneck Distance to Truth" vs "Noise Level".
        - Or "Persistence of the Separating Feature" vs "Noise Level".
    - `figs/robustness_visual.png`: Side-by-side comparison of the filtration/complex at a critical noise level where Isotropic fails but Anisotropic succeeds.

## 3. Implementation Notes
- For the "Anisotropic" method here, you can use a heuristic to set the ellipsoids (e.g., PCA on local k-neighbors) rather than full optimization, to separate the "benefit of anisotropy" from the "difficulty of optimization".
- Explicitly state which method determines the variation (PCA-based vs Optimization-based) in the script comments.
