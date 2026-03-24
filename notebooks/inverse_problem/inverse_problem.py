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
    from scipy.spatial.distance import squareform

    nb_dir = Path(__file__).resolve().parent
    return (
        Ellipse,
        axes_from_cov,
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
    # Task C: Inverse Problem (Metric Learning)

    Given points sampled from a **stretched circle** (10:1 aspect ratio),
    we learn ellipsoid **shapes** (covariance matrices) that recover the
    underlying anisotropic geometry.

    All ellipsoids start as **isotropic** (identity-scaled covariance).
    We optimize their covariance matrices via Cholesky parameterization to
    **maximize the persistence** of the most significant H1 cycle --- the
    cycle that corresponds to the stretched circle itself.

    Centers remain **fixed** throughout; only shapes change.
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
def _():
    # --- Configuration ---
    N_POINTS = 30
    ASPECT_RATIO = 10.0
    SEED = 42
    MAXITER = 100
    INIT_SCALE = 0.3
    return ASPECT_RATIO, INIT_SCALE, MAXITER, N_POINTS, SEED


@app.cell
def _(ASPECT_RATIO, Ellipse, N_POINTS, SEED, axes_from_cov, np, plt):
    def generate_stretched_circle(n, aspect_ratio, seed, noise_std=0.02):
        """Generate points on a stretched circle with small Gaussian noise.

        The ellipse has semi-major axis = aspect_ratio * base and
        semi-minor axis = base, where base is chosen so the minor axis = 1.
        """
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Stretched circle: x spans [-aspect_ratio, aspect_ratio], y spans [-1, 1]
        x = aspect_ratio * np.cos(t)
        y = np.sin(t)
        # Add small noise
        x += rng.randn(n) * noise_std * aspect_ratio
        y += rng.randn(n) * noise_std
        centers = np.column_stack([x, y])
        return centers, t

    centers, angles_true = generate_stretched_circle(N_POINTS, ASPECT_RATIO, SEED)

    # Compute ground-truth tangent angles at each point
    # Tangent to ellipse (a*cos(t), b*sin(t)) is (-a*sin(t), b*cos(t))
    tangent_x = -ASPECT_RATIO * np.sin(angles_true)
    tangent_y = np.cos(angles_true)
    tangent_angles = np.degrees(np.arctan2(tangent_y, tangent_x)) % 180.0

    # Plot the dataset
    fig_data, ax_data = plt.subplots(figsize=(10, 3))
    ax_data.scatter(centers[:, 0], centers[:, 1], s=30, c="#4e79a7", zorder=5)
    # Draw initial isotropic ellipses for reference
    for _i in range(len(centers)):
        _ell = Ellipse(
            xy=centers[_i],
            width=2 * 0.3,
            height=2 * 0.3,
            angle=0,
            facecolor="#4e79a7",
            edgecolor="#2b5c8a",
            alpha=0.1,
            linewidth=0.5,
        )
        ax_data.add_patch(_ell)
    ax_data.set_aspect("equal")
    ax_data.set_title(
        f"Dataset: {N_POINTS} points on stretched circle "
        f"(aspect ratio {ASPECT_RATIO}:1)",
        fontsize=11,
        fontweight="bold",
    )
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    fig_data.tight_layout()
    plt.show()

    return angles_true, centers, generate_stretched_circle, tangent_angles


@app.cell
def _(np):
    def unpack_cholesky(x, n):
        """Unpack flat parameter vector into n lower-triangular 2x2 matrices.

        x = [log_L00_0, L10_0, log_L11_0, log_L00_1, L10_1, log_L11_1, ...]
        Returns L array of shape (n, 2, 2).
        """
        L = np.zeros((n, 2, 2))
        for i in range(n):
            L[i, 0, 0] = np.exp(x[3 * i])
            L[i, 1, 0] = x[3 * i + 1]
            L[i, 1, 1] = np.exp(x[3 * i + 2])
        return L

    def cholesky_to_cov(L):
        """Convert lower-triangular L matrices to covariance matrices Sigma = L @ L.T."""
        n = L.shape[0]
        covs = np.zeros((n, 2, 2))
        for i in range(n):
            covs[i] = L[i] @ L[i].T
        return covs

    return cholesky_to_cov, unpack_cholesky


@app.cell
def _(
    cholesky_to_cov,
    coef_from_cov_grad,
    np,
    pdist_tangency_grad,
    ripser,
    squareform,
    unpack_cholesky,
):
    def loss_and_grad(x, centers):
        """Compute loss = -(death - birth) of most significant H1 cycle and gradient.

        Full VJP chain:
        1. Cholesky params -> covariance matrices
        2. coef_from_cov_grad: (centers, covs) -> coefs
        3. pdist_tangency_grad: coefs -> dists
        4. ripser: dists -> H1 persistence diagram
        5. Backward: H1 subgradient -> grad_dists -> grad_coefs -> grad_covs -> grad_L -> grad_x
        """
        n = len(centers)
        L = unpack_cholesky(x, n)
        covs = cholesky_to_cov(L)

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

        # Find most significant finite H1 cycle
        finite_mask = np.isfinite(dgm[:, 1])
        if not np.any(finite_mask):
            return 0.0, np.zeros_like(x)
        finite = dgm[finite_mask]
        lifetimes = finite[:, 1] - finite[:, 0]
        idx_max = np.argmax(lifetimes)
        b, d = finite[idx_max]

        # Loss = -(death - birth) to maximize persistence
        loss = -(d - b)

        # --- Backward pass ---
        # Step 1: H1 subgradient for single most significant cycle
        grad_dists = np.zeros_like(dists)
        grad_dists[np.argmin(np.abs(dists - b))] += 1.0  # d(-pers)/d(birth) = +1
        grad_dists[np.argmin(np.abs(dists - d))] -= 1.0  # d(-pers)/d(death) = -1

        # Step 2: grad_dists -> grad_coefs
        grad_coefs = vjp_dist(grad_dists)

        # Step 3: grad_coefs -> (grad_centers, grad_covs)
        _grad_centers, grad_covs = vjp_coef(grad_coefs)

        # Step 4: grad_covs -> grad_L (Cholesky VJP)
        # Sigma = L @ L.T, so d_loss/dL = (d_loss/dSigma + d_loss/dSigma.T) @ L
        grad_L = np.zeros_like(L)
        for i in range(n):
            grad_L[i] = (grad_covs[i] + grad_covs[i].T) @ L[i]

        # Step 5: grad_L -> grad_x (chain rule for log-diagonal)
        grad_x = np.zeros_like(x)
        for i in range(n):
            grad_x[3 * i] = grad_L[i, 0, 0] * L[i, 0, 0]  # d/d(log a) = da * a
            grad_x[3 * i + 1] = grad_L[i, 1, 0]  # L10 is unconstrained
            grad_x[3 * i + 2] = grad_L[i, 1, 1] * L[i, 1, 1]  # d/d(log a) = da * a

        return loss, grad_x

    return (loss_and_grad,)


@app.cell
def _(
    INIT_SCALE,
    MAXITER,
    N_POINTS,
    SEED,
    centers,
    cholesky_to_cov,
    loss_and_grad,
    minimize,
    np,
    unpack_cholesky,
):
    # Initialize: isotropic covariance = INIT_SCALE^2 * I
    # Cholesky: L = INIT_SCALE * I, so log(L_diag) = log(INIT_SCALE), L10 = 0
    _rng = np.random.RandomState(SEED)
    x0 = np.zeros(3 * N_POINTS)
    for _i in range(N_POINTS):
        x0[3 * _i] = np.log(INIT_SCALE)  # log(L00)
        x0[3 * _i + 1] = 0.0  # L10
        x0[3 * _i + 2] = np.log(INIT_SCALE)  # log(L11)

    # Track optimization history with callback
    loss_history = []
    snapshot_iters = {0, MAXITER // 3, 2 * MAXITER // 3, MAXITER}
    snapshots = {}
    _iter_counter = [0]

    def _callback(xk):
        _loss, _ = loss_and_grad(xk, centers)
        loss_history.append(_loss)
        it = _iter_counter[0]
        if it in snapshot_iters:
            L_snap = unpack_cholesky(xk, N_POINTS)
            covs_snap = cholesky_to_cov(L_snap)
            snapshots[it] = covs_snap.copy()
        _iter_counter[0] += 1

    # Record initial state
    _loss0, _ = loss_and_grad(x0, centers)
    loss_history.append(_loss0)
    L_init = unpack_cholesky(x0, N_POINTS)
    snapshots[0] = cholesky_to_cov(L_init).copy()

    # Run optimization
    opt_result = minimize(
        loss_and_grad,
        x0,
        args=(centers,),
        method="L-BFGS-B",
        jac=True,
        callback=_callback,
        options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-8},
    )

    # Ensure final state is captured
    L_final = unpack_cholesky(opt_result.x, N_POINTS)
    covs_final = cholesky_to_cov(L_final)
    _last_iter = max(snapshots.keys())
    snapshots[
        _last_iter if _last_iter >= _iter_counter[0] - 1 else _iter_counter[0] - 1
    ] = covs_final.copy()

    return (
        covs_final,
        loss_history,
        opt_result,
        snapshots,
        x0,
    )


@app.cell(hide_code=True)
def _(loss_history, nb_dir, np, opt_result, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    iters = np.arange(len(loss_history))
    ax_loss.plot(
        iters, loss_history, "o-", markersize=3, color="#4e79a7", linewidth=1.2
    )
    ax_loss.set_xlabel("Iteration", fontsize=10)
    ax_loss.set_ylabel("Loss = $-(d - b)$ of most significant H1 cycle", fontsize=10)
    ax_loss.set_title(
        "Optimization History: Maximizing H1 Cycle Persistence",
        fontsize=11,
        fontweight="bold",
    )
    ax_loss.axhline(
        y=loss_history[-1],
        color="#e15759",
        linestyle="--",
        linewidth=0.8,
        label=f"Final loss: {loss_history[-1]:.4f}",
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
    nb_dir,
    np,
    plt,
    snapshots,
):
    # Select 3 snapshots: initial, mid, final
    _sorted_keys = sorted(snapshots.keys())
    if len(_sorted_keys) >= 3:
        _snap_keys = [
            _sorted_keys[0],
            _sorted_keys[len(_sorted_keys) // 2],
            _sorted_keys[-1],
        ]
    else:
        _snap_keys = _sorted_keys
    _snap_labels = ["Initial (isotropic)", "Mid-optimization", "Final (learned)"]

    fig_evo, axes_evo = plt.subplots(
        1, len(_snap_keys), figsize=(5 * len(_snap_keys), 4)
    )
    if len(_snap_keys) == 1:
        axes_evo = [axes_evo]

    _colors_face = ["#4e79a7", "#f28e2b", "#59a14f"]
    _colors_edge = ["#2b5c8a", "#c06e1a", "#3d7a34"]

    for idx, (key, label) in enumerate(zip(_snap_keys, _snap_labels)):
        ax = axes_evo[idx]
        covs_snap = snapshots[key]

        # Draw the ground-truth stretched circle
        _t_curve = np.linspace(0, 2 * np.pi, 200)
        _x_curve = ASPECT_RATIO * np.cos(_t_curve)
        _y_curve = np.sin(_t_curve)
        ax.plot(
            _x_curve, _y_curve, "k--", linewidth=0.8, alpha=0.4, label="Ground truth"
        )

        # Draw ellipsoids
        for _i in range(len(centers)):
            _r_maj, _r_min, _angle_rad = [
                v.item() for v in axes_from_cov(covs_snap[_i])
            ]
            _ell = Ellipse(
                xy=centers[_i],
                width=2 * _r_maj,
                height=2 * _r_min,
                angle=np.degrees(_angle_rad),
                facecolor=_colors_face[idx],
                edgecolor=_colors_edge[idx],
                alpha=0.25,
                linewidth=0.8,
            )
            ax.add_patch(_ell)
            _ell_border = Ellipse(
                xy=centers[_i],
                width=2 * _r_maj,
                height=2 * _r_min,
                angle=np.degrees(_angle_rad),
                facecolor="none",
                edgecolor=_colors_edge[idx],
                linewidth=1.2,
            )
            ax.add_patch(_ell_border)

        ax.scatter(centers[:, 0], centers[:, 1], s=15, c="k", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n(iter {key})", fontsize=10, fontweight="bold")
        ax.set_xlabel("x", fontsize=9)
        if idx == 0:
            ax.set_ylabel("y", fontsize=9)
        _pad = 2.0
        ax.set_xlim(-ASPECT_RATIO - _pad, ASPECT_RATIO + _pad)
        ax.set_ylim(-_pad, _pad)
        ax.grid(True, alpha=0.2)

    fig_evo.suptitle(
        "Visual Evolution of Learned Ellipsoid Shapes",
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
    centers,
    covs_final,
    loss_history,
    mo,
    np,
    tangent_angles,
):
    # Compute alignment angle metric
    # Compare major axis direction of learned covariance with ground-truth tangent
    _learned_angles = np.zeros(len(centers))
    for _i in range(len(centers)):
        _, _, _angle_rad = axes_from_cov(covs_final[_i])
        _learned_angles[_i] = np.degrees(_angle_rad.item()) % 180.0

    # Angular difference (mod 180 since direction is unsigned)
    _angle_diffs = np.abs(_learned_angles - tangent_angles)
    _angle_diffs = np.minimum(_angle_diffs, 180.0 - _angle_diffs)
    _mean_angle_err = float(np.mean(_angle_diffs))
    _median_angle_err = float(np.median(_angle_diffs))

    # Compute aspect ratios of learned ellipsoids
    _learned_aspects = np.zeros(len(centers))
    for _i in range(len(centers)):
        _r_maj, _r_min, _ = axes_from_cov(covs_final[_i])
        _learned_aspects[_i] = _r_maj.item() / max(_r_min.item(), 1e-10)
    _mean_aspect = float(np.mean(_learned_aspects))

    mo.md(
        f"""
    ## Summary

    | Metric | Value |
    |:---|---:|
    | Dataset | {len(centers)} points on {ASPECT_RATIO}:1 stretched circle |
    | Initial loss | {loss_history[0]:.4f} |
    | Final loss | {loss_history[-1]:.4f} |
    | Persistence gained | {abs(loss_history[-1]) - abs(loss_history[0]):.4f} |
    | Mean alignment error (major axis vs tangent) | {_mean_angle_err:.1f} deg |
    | Median alignment error | {_median_angle_err:.1f} deg |
    | Mean learned aspect ratio | {_mean_aspect:.2f} |

    The optimization successfully learns **anisotropic** ellipsoid shapes that
    align with the local tangent direction of the stretched circle.
    Starting from isotropic (circular) covariances, the learned ellipsoids
    elongate along the tangent to maximize the persistence of the dominant
    H1 cycle, effectively recovering the underlying anisotropic geometry
    from the point cloud alone.
    """
    )
    return


if __name__ == "__main__":
    app.run()
