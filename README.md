# EllPHi Demo

Demonstration notebooks and experimental results for the **EllPHi** library.

## Motivation
This repository contains Jupyter and marimo notebooks for the [EllPHi](https://github.com/t-uda/ellphi) library.
Notebooks are separated from the main library to avoid repository bloat.

## Notebooks
- **Jupyter (`.ipynb`)**: For traditional interactive development.
- **marimo (`.py`)**: For reactive, state-consistent, and LLM-friendly notebook development. **New notebooks are encouraged to use marimo.**

## Setup
Ensure you have [Poetry](https://python-poetry.org/) installed.

```bash
# Install dependencies
poetry install
```

## Usage
### Running marimo
```bash
poetry run marimo edit notebooks/your_notebook.py
```

### Running Jupyter
```bash
poetry run jupyter lab
```

