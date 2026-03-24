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
    from ellphi import axes_from_cov
    from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad
    from matplotlib.patches import Ellipse
    from ripser import ripser
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist, squareform

    nb_dir = Path(__file__).resolve().parent
    return (
        Ellipse,
        axes_from_cov,
        cdist,
        coef_from_cov_grad,
        ellphi,
        minimize,
        mo,
        nb_dir,
        np,
        pdist_tangency_grad,
        plt,
        ripser,
        squareform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inverse Problem: Metric Learning via Anisotropic PH

    Given points sampled from a **stretched circle** (10:1 aspect ratio),
    we learn a **single metric tensor** (shared covariance matrix $\Sigma$)
    that maximises the persistence of the dominant H1 cycle.

    This is the core "metric learning" problem: starting from an isotropic
    metric, discover the **anisotropy** of the underlying data manifold by
    optimising a topological objective.

    **Parameterisation:** Cholesky $\Sigma = LL^\top$ with log-diagonal
    (3 free parameters in 2D). All ellipsoids share the same $\Sigma$;
    centres remain **fixed**.
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
    # --- Configuration ---
    N_POINTS = 30
    ASPECT_RATIO = 10.0
    SEED = 42
    MAXITER = 300
    return ASPECT_RATIO, MAXITER, N_POINTS, SEED


@app.cell
def _(ASPECT_RATIO, N_POINTS, SEED, cdist, np, plt):
    def generate_stretched_circle(n, aspect_ratio, seed, noise_std=0.02):
        """Generate points on a stretched circle with small Gaussian noise."""
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = aspect_ratio * np.cos(t)
        y = np.sin(t)
        x += rng.randn(n) * noise_std * aspect_ratio
        y += rng.randn(n) * noise_std
        return np.column_stack([x, y]), t

    centers, angles_true = generate_stretched_circle(N_POINTS, ASPECT_RATIO, SEED)

    # Compute ground-truth tangent direction of the data ellipse at each point
    # Tangent to (a*cos(t), b*sin(t)) is (-a*sin(t), b*cos(t))
    tangent_x = -ASPECT_RATIO * np.sin(angles_true)
    tangent_y = np.cos(angles_true)
    tangent_angles = np.degrees(np.arctan2(tangent_y, tangent_x)) % 180.0

    # Set initial isotropic scale from median 1-NN distance
    _D = cdist(centers, centers)
    np.fill_diagonal(_D, np.inf)
    median_nn = float(np.median(np.min(_D, axis=1)))

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.scatter(centers[:, 0], centers[:, 1], s=30, c="#4e79a7", zorder=5)
    _ax.set_aspect("equal")
    _ax.set_title(
        f"Dataset: {N_POINTS} points on {ASPECT_RATIO}:1 stretched circle"
        f"  (median 1-NN = {median_nn:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _fig.tight_layout()
    plt.show()

    return angles_true, centers, generate_stretched_circle, median_nn, tangent_angles


@app.cell
def _(np):
    def unpack_cholesky_shared(x):
        """Unpack 3-element vector into a single 2x2 lower-triangular L.

        x = [log_L00, L10, log_L11]
        Returns L (2, 2).
        """
        L = np.zeros((2, 2))
        L[0, 0] = np.exp(x[0])
        L[1, 0] = x[1]
        L[1, 1] = np.exp(x[2])
        return L

    def cholesky_to_cov_shared(L):
        """Sigma = L @ L.T for a single 2x2 matrix."""
        return L @ L.T

    return cholesky_to_cov_shared, unpack_cholesky_shared


@app.cell
def _(
    cholesky_to_cov_shared,
    coef_from_cov_grad,
    np,
    pdist_tangency_grad,
    ripser,
    squareform,
    unpack_cholesky_shared,
):
    def loss_and_grad(x, centers):
        """Loss = -(death - birth) of most significant H1 cycle.

        Shared covariance: all N points use the same Sigma = L @ L.T.
        VJP chain: Cholesky -> cov -> coefs -> dists -> ripser -> subgradient.
        """
        n = len(centers)
        L = unpack_cholesky_shared(x)
        cov = cholesky_to_cov_shared(L)
        covs = np.tile(cov, (n, 1, 1))

        try:
            coefs, vjp_coef = coef_from_cov_grad(centers, covs)
            dists, vjp_dist = pdist_tangency_grad(coefs)
        except (RuntimeError, ZeroDivisionError):
            return 1e6, np.zeros_like(x)

        dm = squareform(dists)
        result = ripser(dm, maxdim=1, distance_matrix=True)
        dgm = result["dgms"][1]

        if len(dgm) == 0:
            return 0.0, np.zeros_like(x)

        finite_mask = np.isfinite(dgm[:, 1])
        if not np.any(finite_mask):
            return 0.0, np.zeros_like(x)
        finite = dgm[finite_mask]
        lifetimes = finite[:, 1] - finite[:, 0]
        idx_max = np.argmax(lifetimes)
        b, d = finite[idx_max]

        loss = -(d - b)

        # --- Backward ---
        # H1 subgradient for single most significant cycle
        grad_dists = np.zeros_like(dists)
        grad_dists[np.argmin(np.abs(dists - b))] += 1.0
        grad_dists[np.argmin(np.abs(dists - d))] -= 1.0

        grad_coefs = vjp_dist(grad_dists)
        _grad_centers, grad_covs = vjp_coef(grad_coefs)

        # Sum per-point gradients → shared covariance gradient
        grad_cov = np.sum(grad_covs, axis=0)

        # Cholesky VJP: Sigma = L @ L.T  →  grad_L = (grad_Sigma + grad_Sigma.T) @ L
        grad_L = (grad_cov + grad_cov.T) @ L

        # Pack: chain rule for log-diagonal
        grad_x = np.array(
            [
                grad_L[0, 0] * L[0, 0],  # d/d(log L00)
                grad_L[1, 0],  # L10 is unconstrained
                grad_L[1, 1] * L[1, 1],  # d/d(log L11)
            ]
        )

        return loss, grad_x

    return (loss_and_grad,)


@app.cell
def _(
    MAXITER,
    axes_from_cov,
    centers,
    cholesky_to_cov_shared,
    loss_and_grad,
    mo,
    np,
    unpack_cholesky_shared,
):
    # Initial isotropic scale: small enough that H1 features exist.
    # Scale 0.3 gives ~9 H1 features at the start — rich gradient signal.
    init_scale = 0.3
    x0 = np.array([np.log(init_scale), 0.0, np.log(init_scale)])

    # Gradient descent with gradient clipping.
    # L-BFGS-B is too aggressive for this piecewise-linear landscape:
    # large steps can kill all H1 features, producing zero gradient.
    # Clipped GD with small lr is stable and converges well.
    _lr = 0.05
    _max_grad_norm = 1.0

    loss_history = []
    cov_history = []
    _x = x0.copy()
    _best_loss = float("inf")
    _best_x = _x.copy()

    for _step in range(MAXITER):
        _loss, _grad = loss_and_grad(_x, centers)
        loss_history.append(_loss)
        _L = unpack_cholesky_shared(_x)
        cov_history.append(cholesky_to_cov_shared(_L).copy())

        if _loss < _best_loss:
            _best_loss = _loss
            _best_x = _x.copy()

        # Clip gradient norm to prevent topological collapse
        _gnorm = np.linalg.norm(_grad)
        if _gnorm > _max_grad_norm:
            _grad = _grad / _gnorm * _max_grad_norm
        _x = _x - _lr * _grad

    # Use best iterate (persistence is non-monotone due to topological transitions)
    _L_final = unpack_cholesky_shared(_best_x)
    cov_final = cholesky_to_cov_shared(_L_final)
    _r_maj, _r_min, _angle_rad = [v.item() for v in axes_from_cov(cov_final)]

    mo.md(
        f"""
    **Optimisation result** ({MAXITER} gradient steps, lr={_lr}):
    - Initial persistence: **{abs(loss_history[0]):.4f}** (isotropic, scale = {init_scale})
    - Best persistence: **{abs(_best_loss):.4f}**
    - Persistence gain: **{abs(_best_loss) / max(abs(loss_history[0]), 1e-10):.1f}x**
    - Learned $\\Sigma$ semi-axes: **{_r_maj:.3f}** x **{_r_min:.3f}**
      (aspect ratio {_r_maj / max(_r_min, 1e-10):.1f}:1,
       major axis at {np.degrees(_angle_rad) % 360:.1f} deg)
    """
    )
    return cov_final, cov_history, init_scale, loss_history


@app.cell(hide_code=True)
def _(loss_history, nb_dir, np, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    _iters = np.arange(len(loss_history))
    _pers = [-v for v in loss_history]
    ax_loss.plot(
        _iters,
        _pers,
        "-",
        markersize=2,
        color="#4e79a7",
        linewidth=1.2,
    )
    ax_loss.set_xlabel("Iteration", fontsize=10)
    ax_loss.set_ylabel("Max H1 persistence $(d - b)$", fontsize=10)
    ax_loss.set_title(
        "Metric Learning: Maximising H1 Cycle Persistence",
        fontsize=11,
        fontweight="bold",
    )
    _best_pers = max(_pers)
    ax_loss.axhline(
        y=_best_pers,
        color="#e15759",
        linestyle="--",
        linewidth=0.8,
        label=f"Best: {_best_pers:.4f}",
    )
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    plt.savefig(nb_dir / "optimization_history.pdf", bbox_inches="tight", dpi=150)
    plt.show()

    return


@app.cell(hide_code=True)
def _(
    ASPECT_RATIO,
    Ellipse,
    axes_from_cov,
    centers,
    cov_history,
    nb_dir,
    np,
    plt,
):
    # Select 3 snapshots: initial, mid, final
    _n_hist = len(cov_history)
    _snap_indices = [0, _n_hist // 2, _n_hist - 1]
    _snap_labels = ["Initial (isotropic)", "Mid-optimisation", "Final (learned)"]

    fig_evo, axes_evo = plt.subplots(1, 3, figsize=(15, 4))

    _colors_face = ["#4e79a7", "#f28e2b", "#59a14f"]
    _colors_edge = ["#2b5c8a", "#c06e1a", "#3d7a34"]

    for _panel, (_idx, _label) in enumerate(zip(_snap_indices, _snap_labels)):
        _ax = axes_evo[_panel]
        _cov_snap = cov_history[_idx]

        # Ground truth curve
        _t_curve = np.linspace(0, 2 * np.pi, 200)
        _ax.plot(
            ASPECT_RATIO * np.cos(_t_curve),
            np.sin(_t_curve),
            "k--",
            linewidth=0.8,
            alpha=0.4,
            label="Ground truth",
        )

        # Draw shared ellipsoid at each point
        _r_maj, _r_min, _angle_rad = [v.item() for v in axes_from_cov(_cov_snap)]
        _angle_deg = np.degrees(_angle_rad)
        for _i in range(len(centers)):
            _ell = Ellipse(
                xy=centers[_i],
                width=2 * _r_maj,
                height=2 * _r_min,
                angle=_angle_deg,
                facecolor=_colors_face[_panel],
                edgecolor=_colors_edge[_panel],
                alpha=0.2,
                linewidth=0.8,
            )
            _ax.add_patch(_ell)

        _ax.scatter(centers[:, 0], centers[:, 1], s=15, c="k", zorder=5)
        _ax.set_aspect("equal")
        _ax.set_title(
            f"{_label}\n({_r_maj:.2f} x {_r_min:.2f}, {_angle_deg:.0f} deg)",
            fontsize=10,
            fontweight="bold",
        )
        _ax.set_xlabel("x", fontsize=9)
        if _panel == 0:
            _ax.set_ylabel("y", fontsize=9)
        _pad = 2.0
        _ax.set_xlim(-ASPECT_RATIO - _pad, ASPECT_RATIO + _pad)
        _ax.set_ylim(-_pad - 1, _pad + 1)
        _ax.grid(True, alpha=0.2)

    fig_evo.suptitle(
        "Learned Metric Evolution (shared $\\Sigma$ for all points)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig_evo.tight_layout()
    plt.savefig(nb_dir / "visual_evolution.pdf", bbox_inches="tight", dpi=150)
    plt.show()

    return


@app.cell(hide_code=True)
def _(
    ASPECT_RATIO,
    axes_from_cov,
    cov_final,
    loss_history,
    mo,
    np,
):
    _r_maj, _r_min, _angle_rad = [v.item() for v in axes_from_cov(cov_final)]
    _learned_aspect = _r_maj / max(_r_min, 1e-10)
    _major_angle = np.degrees(_angle_rad) % 180.0

    # The ground truth "stretch" is along the x-axis (0 degrees).
    # A learned major axis near 0 or 180 deg means alignment.
    _alignment_err = min(_major_angle, 180.0 - _major_angle)

    mo.md(
        f"""
    ## Summary

    | Metric | Value |
    |:---|---:|
    | Dataset | {ASPECT_RATIO}:1 stretched circle |
    | Initial persistence (isotropic) | {abs(loss_history[0]):.4f} |
    | **Final persistence (learned)** | **{abs(loss_history[-1]):.4f}** |
    | Persistence gain | **{abs(loss_history[-1]) / max(abs(loss_history[0]), 1e-10):.1f}x** |
    | Learned semi-axes | {_r_maj:.3f} x {_r_min:.3f} |
    | Learned aspect ratio | **{_learned_aspect:.1f}:1** |
    | Major axis angle | {_major_angle:.1f} deg |
    | Alignment error to x-axis | **{_alignment_err:.1f} deg** |

    The optimisation discovers that **stretching the metric along the
    x-axis** (the direction of data elongation) makes the H1 cycle
    dramatically more persistent. Starting from isotropic circles, the
    learned metric tensor elongates to bridge the large inter-point
    gaps along the major axis of the data ellipse while keeping the
    minor axis tight, recovering the underlying anisotropy from the
    topology alone.
    """
    )
    return


if __name__ == "__main__":
    app.run()
