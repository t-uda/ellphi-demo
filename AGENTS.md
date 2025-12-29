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
  - `notebooks/`: Contains demonstration projects.
    - Each demo resides in its own subdirectory (e.g., `notebooks/demo_name/`).
    - **Structure**:
      - `app.py`: Main marimo notebook.
      - `public/`: Static assets (images, pdfs). Marimo serves these automatically.
      - `data/`: Generated datasets and CSVs.
  - `specs/`: Internal specifications and research notes.
  - `docs/`: GitHub Pages public reports and documentation.

## Development Setup
```bash
# Install dependencies
poetry install

# Run marimo
poetry run marimo edit
```

## Recurring Issues & Prevention

### Matplotlib Cache & Display
- **Problem**: `MPLCONFIGDIR` warnings or figures not appearing in notebooks.
- **Prevention**: 
  1. Set `os.environ['MPLCONFIGDIR']` to `.cache/matplotlib` **before** importing pyplot.
  2. **Call `plt.show()`**: To ensure figures are captured and displayed in the notebook, always call `plt.show()` at the end of the plotting cell.

### Marimo Export & GitHub Rendering
- **Problem**: Figures may not be captured in exports if not explicitly shown or if GitHub's static viewer cannot render marimo's custom tags.
- **Prevention**: 
  1. **Use `mo.image` for Robustness**: To ensure figures are visible on GitHub (both in HTML and ipynb), save the figure as a **PNG** to the `public/` directory and display it using `mo.image(src="public/your_plot.png")`.
  2. **Exporting with Outputs**: When exporting to Jupyter format, always use the `--include-outputs` and `-f` (force) flags:
     ```bash
     poetry run marimo export ipynb app.py -o notebook.ipynb --include-outputs -f
     ```

### Marimo Multiple Definitions
- **Problem**: Importing the same module (e.g., `import IPython`) in multiple cells causes `MultipleDefinitionError`.
- **Prevention**: Consolidate all common imports into a single setup cell at the beginning of the notebook.
