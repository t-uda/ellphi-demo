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
