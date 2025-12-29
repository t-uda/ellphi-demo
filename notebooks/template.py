import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    try:
        import ellphi
        ellphi_available = True
    except ImportError:
        ellphi_available = False
    return ellphi, ellphi_available, mo


@app.cell
def _(mo):
    mo.md(r"""
    # EllPHi Demo Template
    This is a template marimo notebook for demonstrating EllPHi.
    """)
    return


@app.cell
def _(ellphi_available, mo):
    mo.stop(not ellphi_available, "EllPHi is not installed in the current environment.")
    mo.md(f"EllPHi version: {mo.status.spinner('Checking...')}")
    return


@app.cell
def _(ellphi):
    print(f"EllPHi version: {ellphi.__version__}")
    return


if __name__ == "__main__":
    app.run()
