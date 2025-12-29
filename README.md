# EllPHi Demo

Demonstration notebooks and experimental results for the **EllPHi** library.

## Motivation
This repository contains Jupyter and marimo notebooks for the [EllPHi](https://github.com/t-uda/ellphi) library.
Notebooks are separated from the main library to avoid repository bloat.

## Notebooks
- **Jupyter (`.ipynb`)**: For traditional interactive development.
- **marimo (`.py`)**: For reactive, state-consistent, and LLM-friendly notebook development. **New notebooks are encouraged to use marimo.**

## Setup
Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# Install dependencies
uv sync
```

## Usage
### Running marimo
```bash
uv run marimo edit notebooks/your_notebook.py
```

### Running Jupyter
```bash
uv run jupyter lab
```

