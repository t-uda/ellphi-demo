# AGENTS.md

## Project Overview
`ellphi-demo` is a repository dedicated to demonstration notebooks and experimental results for the `ellphi` library.
Move Jupyter notebooks here from the `ellphi` repository to keep the core library minimal.

## Environment & Tools
- **Dependency Management**: [Poetry](https://python-poetry.org/)
- **Notebook Formats**:
  - **Jupyter Notebook (`.ipynb`)**: Traditional format, supported for existing notebooks.
  - **marimo (`.py`)**: Reactive and LLM-friendly notebook format. **Newly created notebooks should prioritize marimo.**

## Instruction for Agents
- **Python Environment**: Always use `poetry run` to execute scripts or notebooks. Never use global `python` or `pip`.
- **Workflow**:
  - To edit marimo notebooks: `poetry run marimo edit <notebook>.py`
  - To run jupyter: `poetry run jupyter lab`
- **Deslop**: Use the `/deslop` workflow to remove AI-generated code slop when modifying files.
- **Matplotlib**: Set `MPLCONFIGDIR` to a writable directory (e.g., `.cache/matplotlib`) at the start of scripts/notebooks to avoid "not a writable directory" warnings on some systems.
- **Serena files**: Track `.serena/project.yml`; keep `.serena/cache/` and `.serena/memories/` ignored.
- **Documentation**: Keep this `AGENTS.md` updated with any new project-specific rules or setup steps.
- **Repository Layout**:
  - `notebooks/`: For Jupyter and marimo notebooks.
  - `docs/`: Documentation and research notes.
  - `data/`: (If needed) Small datasets for demos. Large files should be handled via Git LFS or external storage.

## Development Setup
```bash
# Install dependencies
poetry install

# Run marimo
poetry run marimo edit
```

## Recurring Issues & Prevention

### Matplotlib Cache (Read-Only FS)
- **Problem**: `MPLCONFIGDIR` defaults to a read-only home directory in some environments, causing warnings.
- **Prevention**: Set `os.environ['MPLCONFIGDIR']` to a local writable path (e.g., `.cache/matplotlib`) **before** `import matplotlib.pyplot`.

### Marimo Export & GitHub Rendering
- **Problem**: `marimo export ipynb` wraps outputs in custom HTML tags (`<marimo-mime-renderer>`) which GitHub's static viewer does not render.
- **Prevention**: To ensure figures are visible on GitHub:
  1. Save the figure as a **PNG** file to disk.
  2. Display it using `IPython.display.Image(filename="...")` (ensure `IPython` is imported).
  3. **Do not** rely solely on returning the Figure object if GitHub compatibility is required.

### Marimo Multiple Definitions
- **Problem**: Importing the same module (e.g., `import IPython`) in multiple cells causes `MultipleDefinitionError`.
- **Prevention**: Consolidate all common imports into a single setup cell at the beginning of the notebook.
