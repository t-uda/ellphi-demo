# Future Integration Plan

This document outlines the plan for synchronizing `ellphi` and `ellphi-demo` repositories to achieve a clean dependency structure.

## Plan: Dependency Decoupling

Once `ellphi-demo` is fully operational and the maintenance workflow is established, the following steps will be taken in coordination with a version bump of the `ellphi` library.

### 1. Clean up `ellphi` repository
- **Action**: Remove demo-related dependencies and optional groups.
- **Targets in `pyproject.toml`**:
    - `[project.optional-dependencies]` -> `demo` section
    - `[tool.poetry.group.demo]`
- **Goal**: Minimize the footprint of the core library.

### 2. Consolidate demo dependencies in `ellphi-demo`
- **Action**: Ensure all analysis/visualization tools are captured here.
- **Current Stack**: `marimo`, `jupyterlab`, `pandas`, `plotly`, `homcloud`, `seaborn`, `matplotlib`.
- **Action**: Update the `ellphi` dependency in `pyproject.toml` to point to the newly released version (once bumped).

### 3. Notebook Migration Completion
- **Action**: Move all remaining `.ipynb` files from `ellphi/notebooks` to `ellphi-demo/notebooks`.
- **Action**: Update `ellphi`'s `README.md` to point users to this repository for examples.

## Timeline
This plan will be executed simultaneously with the next meaningful version update of `ellphi` to avoid multiple redundant version bumps.
