# ellphi-demo

Demo notebooks for the [EllPHi](https://github.com/t-uda/ellphi) library.

## Environment

- Package manager: `uv` (not pip/poetry). Always use `uv run`.
- Notebook format: marimo (`.py`). Do not create `.ipynb`.
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Matplotlib: `.env` sets `MPLCONFIGDIR=.cache/matplotlib`.

## Repository layout

- `notebooks/<task_name>/` — each experiment in its own directory
- `specs/` — experiment specifications
- `docs/` — GitHub Pages reports
- `public/` — generated artifacts (PDFs, PNGs)

## EllPHi API knowledge

Install the `ellphi-api` skill (symlink from the ellphi repo):

```bash
ln -s /path/to/ellphi/contrib/agent-skill ~/.claude/skills/ellphi-api
```

For API details beyond the skill, read `../ellphi/src/ellphi/` directly.
