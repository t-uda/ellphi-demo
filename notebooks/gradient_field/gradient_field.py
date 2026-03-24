import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import ellphi
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ellphi.grad import coef_from_cov_grad, tangency_grad
    from matplotlib.patches import Ellipse

    nb_dir = Path(__file__).resolve().parent
    return (
        Ellipse,
        coef_from_cov_grad,
        ellphi,
        mo,
        nb_dir,
        np,
        plt,
        tangency_grad,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradient Field Visualization

    This notebook visualizes the **tangency distance** field and its gradient
    for a pair of anisotropic ellipsoids in 2D.

    - **Ellipsoid A** is fixed at the origin with semi-axes (0.5, 0.3).
    - **Ellipsoid B** (same shape) is swept across a 40x40 grid over [-3, 3]².
    - At each grid point we compute the tangency distance *t* and ∇_{x_B} *t*.
    - The result is a contour plot of *t* overlaid with a quiver plot of ∇*t*.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # Publication-quality matplotlib settings
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6
    return


@app.cell
def _(np):
    # --- Ellipsoid configuration ---
    R0 = 0.5  # semi-axis along x
    R1 = 0.3  # semi-axis along y

    center_A = np.array([0.0, 0.0])
    cov_A = np.diag([R0**2, R1**2])

    # Ellipsoid B has the same shape; its center varies over the grid
    cov_B = np.diag([R0**2, R1**2])

    # Grid resolution
    N_GRID = 40
    return N_GRID, R0, R1, center_A, cov_A, cov_B


@app.cell
def _(N_GRID, center_A, coef_from_cov_grad, cov_A, cov_B, ellphi, np, tangency_grad):
    # --- Compute tangency distance and gradient on the grid ---
    xs = np.linspace(-3.0, 3.0, N_GRID)
    ys = np.linspace(-3.0, 3.0, N_GRID)
    XX, YY = np.meshgrid(xs, ys)

    T_grid = np.full_like(XX, np.nan)
    GX_grid = np.full_like(XX, np.nan)
    GY_grid = np.full_like(XX, np.nan)

    # Precompute coef_A (fixed)
    coef_A = ellphi.coef_from_cov(center_A[None], cov_A[None])[0]

    for i in range(N_GRID):
        for j in range(N_GRID):
            gx, gy = XX[i, j], YY[i, j]

            # Build coef_B with VJP
            coefs_B, vjp_coef = coef_from_cov_grad(np.array([[gx, gy]]), cov_B[None])

            # Tangency distance and its gradient w.r.t. coefficients
            g = tangency_grad(coef_A, coefs_B[0])

            # Chain rule: gradient w.r.t. center of B
            grad_center, _ = vjp_coef(g.dt_dq[None])

            T_grid[i, j] = g.t
            GX_grid[i, j] = grad_center[0, 0]
            GY_grid[i, j] = grad_center[0, 1]

    return GX_grid, GY_grid, T_grid, XX, YY


@app.cell(hide_code=True)
def _(Ellipse, GX_grid, GY_grid, R0, R1, T_grid, XX, YY, nb_dir, np, plt):
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # Contour plot of tangency distance
    levels = np.linspace(np.nanmin(T_grid), np.nanmax(T_grid), 25)
    cs = ax.contourf(XX, YY, T_grid, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(XX, YY, T_grid, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cs, ax=ax, label="Tangency distance $t$", shrink=0.8)

    # Quiver plot of gradient (subsample for readability)
    step = 2
    ax.quiver(
        XX[::step, ::step],
        YY[::step, ::step],
        GX_grid[::step, ::step],
        GY_grid[::step, ::step],
        color="white",
        alpha=0.7,
        scale=25,
        width=0.004,
        headwidth=3.5,
    )

    # Draw ellipsoid A at origin
    ellipse_patch = Ellipse(
        (0.0, 0.0),
        width=2 * R0,
        height=2 * R1,
        facecolor="salmon",
        edgecolor="darkred",
        linewidth=1.5,
        alpha=0.9,
        zorder=5,
        label="Ellipsoid A",
    )
    ax.add_patch(ellipse_patch)

    ax.set_xlabel("$x_B$")
    ax.set_ylabel("$y_B$")
    ax.set_title("Tangency distance field and gradient $\\nabla_{x_B} t$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()

    # Save PDF
    out_path = nb_dir / "gradient_field_2d.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
