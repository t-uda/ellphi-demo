# AGENTS.md

## Project Overview
`ellphi-demo` is a repository dedicated to demonstration notebooks and experimental results for the `ellphi` library.
Move Jupyter notebooks here from the `ellphi` repository to keep the core library minimal.

## Environment & Tools
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Notebook Formats**:
  - **Jupyter Notebook (`.ipynb`)**: Traditional format, supported for existing notebooks.
  - **marimo (`.py`)**: Reactive and LLM-friendly notebook format. **Newly created notebooks should prioritize marimo.**

## Instruction for Agents
- **Python Environment**: Always use `uv run` to execute scripts or notebooks. Never use global `python` or `pip`.
- **Workflow**:
  - To edit marimo notebooks: `uv run marimo edit <notebook>.py`
  - To run jupyter: `uv run jupyter lab`
  - To lint and format code: `uv run ruff check --fix .` and `uv run ruff format .`
- **Deslop**: Use the `/deslop` workflow to remove AI-generated code slop when modifying files.
- **Matplotlib**: Set `MPLCONFIGDIR` to a writable directory (e.g., `.cache/matplotlib`) at the start of scripts/notebooks to avoid "not a writable directory" warnings on some systems.
- **Serena files**: Track `.serena/project.yml`; keep `.serena/cache/` and `.serena/memories/` ignored.
- **Documentation**: Keep this `AGENTS.md` updated with any new project-specific rules or setup steps.
- **Repository Layout**:
  - `notebooks/`: Contains demonstration projects.
    - Each demo resides in its own subdirectory (e.g., `notebooks/demo_name/`).
  - `specs/`: Internal specifications and research notes.
  - `docs/`: GitHub Pages public reports and documentation.

## Development Setup
```bash
# Install dependencies
uv sync

# Run marimo
uv run marimo edit
```

## Recurring Issues & Prevention

### Matplotlib Cache & Display
- **Problem**: `MPLCONFIGDIR` warnings or figures not appearing in notebooks.
- **Prevention**: 
  1. Set `os.environ['MPLCONFIGDIR']` to `.cache/matplotlib` **before** importing pyplot.
  2. **Call `plt.show()`**: To ensure figures are captured and displayed in the notebook, always call `plt.show()` at the end of the plotting cell.

### Marimo Export & GitHub Rendering
- **Exporting with Outputs**: When exporting to a HTML format, always use the `-f` (force) flag:
     ```bash
     uv run marimo export html demo.py -o demo.html -f
     ```

### Marimo Multiple Definitions
- **Problem**: Importing the same module (e.g., `import IPython`) in multiple cells causes `MultipleDefinitionError`.
- **Prevention**: Consolidate all common imports into a single setup cell at the beginning of the notebook.

### macOS Troubleshooting
If you encounter build errors with `homcloud` related to `CGAL` or `mpfr` (e.g., `'CGAL/version_macros.h' file not found`), you may need to provide include and library paths for Homebrew:

```bash
export CFLAGS="-I/opt/homebrew/include"
export CXXFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
uv sync
```

