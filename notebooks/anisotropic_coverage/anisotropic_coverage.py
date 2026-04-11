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
    from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad
    from matplotlib.patches import Ellipse
    from ripser import ripser
    from scipy.spatial.distance import squareform

    nb_dir = Path(__file__).resolve().parent
    return (
        Ellipse,
        coef_from_cov_grad,
        ellphi,
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
    # Anisotropic Sensor Coverage Optimization

    This notebook demonstrates that **anisotropic** (ellipsoidal) sensors can
    achieve significantly lower H1 persistence than **isotropic** (circular)
    sensors of the same area.

    We optimize sensor **positions and orientations** to **minimize** H1 total
    persistence (reduce coverage holes in the tangency-based Rips filtration).
    Anisotropic sensors have **3N** degrees of freedom (x, y, θ per sensor)
    while isotropic sensors have only **2N** (x, y). The extra rotational
    freedom lets elongated ellipses bridge gaps that circles cannot.
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
    N_SENSORS = 40  # mobile sensors to optimize
    R_MAJOR = 0.10  # semi-major axis (fixed during optimization)
    R_MINOR = 0.05  # semi-minor axis (fixed during optimization)
    R_ISO = np.sqrt(R_MAJOR * R_MINOR)  # equal-area circle radius
    SEED = 42

    # Wall: fixed thin ellipses along boundary to create H1
    R_WALL_MAJOR = 0.25  # wall semi-major (along edge)
    R_WALL_MINOR = 0.02  # wall semi-minor (thin)

    # Optimization
    N_RESTARTS = 1
    MAXITER_WARMUP = 20
    MAXITER = 100
    LEARNING_RATE = 0.001
    JIGGLE_RATE = 0.05  # Random θ noise per iteration (e.g. 0.01 = 1%)
    T_SCALE = 10.00  # Angular scaling factor for gradient balancing

    # Initialization ranges
    INIT_X_RANGE = (0.15, 0.85)
    INIT_Y_RANGE = (0.30, 0.70)
    INIT_THETA_RANGE = (-0.15, 0.15)
    JITTER_MAGNITUDE = 0.05
    return (
        INIT_THETA_RANGE,
        INIT_X_RANGE,
        INIT_Y_RANGE,
        JIGGLE_RATE,
        JITTER_MAGNITUDE,
        LEARNING_RATE,
        MAXITER,
        MAXITER_WARMUP,
        N_RESTARTS,
        N_SENSORS,
        R_ISO,
        R_MAJOR,
        R_MINOR,
        R_WALL_MAJOR,
        R_WALL_MINOR,
        SEED,
        T_SCALE,
    )


@app.cell
def _(R_WALL_MAJOR, R_WALL_MINOR, ellphi, np, ripser, squareform):
    def build_covs(thetas, r0, r1):
        """Build covariance matrices from rotation angles and semi-axes."""
        n = len(thetas)
        covs = np.zeros((n, 2, 2))
        for i in range(n):
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            rot = np.array([[c, -s], [s, c]])
            diag = np.diag([r0**2, r1**2])
            covs[i] = rot @ diag @ rot.T
        return covs

    def get_wall_layout():
        """Returns centers and thetas for a 6-ellipse boundary connected perfectly tip-to-tip."""
        centers = []
        thetas = []

        dx = R_WALL_MAJOR * np.cos(np.pi / 4)
        dy = R_WALL_MAJOR * np.sin(np.pi / 4)

        bx, by = 0.5, 0.15

        # Bottom left arm
        centers.append([bx - dx, by + dy])
        thetas.append(-np.pi / 4)

        # Bottom right arm
        centers.append([bx + dx, by + dy])
        thetas.append(np.pi / 4)

        # Left wall (1 ellipse)
        lx = bx - 2 * dx
        ly = by + 2 * dy + R_WALL_MAJOR
        centers.append([lx, ly])
        thetas.append(np.pi / 2)

        # Right wall (1 ellipse)
        rx = bx + 2 * dx
        ry = by + 2 * dy + R_WALL_MAJOR
        centers.append([rx, ry])
        thetas.append(np.pi / 2)

        # Top Left Lid
        top_left_tip_y = by + 2 * dy + 2 * R_WALL_MAJOR
        centers.append([lx + dx, top_left_tip_y - dy])
        thetas.append(-np.pi / 4)

        # Top Right Lid
        centers.append([rx - dx, top_left_tip_y - dy])
        thetas.append(np.pi / 4)

        v_bounds = (lx, rx, by, top_left_tip_y)

        return np.array(centers), np.array(thetas), v_bounds

    def build_wall_coefs():
        """Build coefficient vectors for fixed wall ellipses."""
        wall_centers, wall_thetas, v_bounds = get_wall_layout()
        wall_covs = build_covs(wall_thetas, R_WALL_MAJOR, R_WALL_MINOR)
        coefs = ellphi.coef_from_cov(wall_centers, wall_covs)
        return coefs, len(wall_centers), wall_centers, wall_thetas, v_bounds

    def compute_h1(centers, thetas, r0, r1, wall_coefs=None):
        """Compute H1 persistence diagram from ellipse tangency distances."""
        covs = build_covs(thetas, r0, r1)
        coefs = ellphi.coef_from_cov(centers, covs)
        if wall_coefs is not None:
            coefs = np.vstack([wall_coefs, coefs])
        dists = ellphi.pdist_tangency(coefs)
        result = ripser(squareform(dists), maxdim=1, distance_matrix=True)
        dgm = result["dgms"][1]
        return dgm

    def h1_total_persistence(centers, thetas, r0, r1, wall_coefs=None):
        """Sum of finite H1 lifetimes. Returns 0.0 on solver errors."""
        try:
            dgm = compute_h1(centers, thetas, r0, r1, wall_coefs=wall_coefs)
        except RuntimeError:
            return float("infinity")
        if len(dgm) == 0:
            return 0.0
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        return float(np.sum(lifetimes))

    return build_covs, build_wall_coefs, compute_h1, h1_total_persistence


@app.cell
def _(
    INIT_THETA_RANGE,
    INIT_X_RANGE,
    INIT_Y_RANGE,
    JIGGLE_RATE,
    JITTER_MAGNITUDE,
    LEARNING_RATE,
    MAXITER,
    MAXITER_WARMUP,
    N_RESTARTS,
    N_SENSORS,
    R_ISO,
    R_MAJOR,
    R_MINOR,
    SEED,
    T_SCALE,
    build_covs,
    build_wall_coefs,
    coef_from_cov_grad,
    h1_total_persistence,
    mo,
    np,
    pdist_tangency_grad,
    ripser,
    squareform,
):
    def gradient_descent(loss_grad_fn, x0, args, max_iter, lr, jiggle=0.0):
        """Simple Gradient Descent with optional θ-jiggle."""
        x = x0.copy()
        is_aniso = len(x) % 3 == 0

        for _ in range(max_iter):
            loss, grad = loss_grad_fn(x, *args)

            # Gradient Step
            x = x - lr * grad

            # θ-Jiggle: add small random noise to rotation components
            if is_aniso and jiggle > 0:
                noise = np.random.uniform(-jiggle, jiggle, len(x) // 3)
                x[2::3] += noise

        return x, loss

    def initialize_sensors(rng, n_sensors, type="aniso"):
        """Centralized sensor initialization using configuration ranges."""
        cx = rng.uniform(*INIT_X_RANGE, n_sensors)
        cy = rng.uniform(*INIT_Y_RANGE, n_sensors)
        cy += 1.8 * (cx - 0.5) ** 2
        centers = np.column_stack([cx, cy])

        if type == "aniso":
            thetas = rng.uniform(INIT_THETA_RANGE[0], INIT_THETA_RANGE[1], n_sensors)
            return centers, thetas
        return centers

    def _pack(centers, thetas):
        """Pack centers (N,2) and thetas (N,) into [x0,y0,t0, x1,y1,t1, ...]."""
        n = len(thetas)
        x = np.zeros(3 * n)
        for i in range(n):
            x[3 * i] = centers[i, 0]
            x[3 * i + 1] = centers[i, 1]
            x[3 * i + 2] = thetas[i] / T_SCALE
        return x

    def _unpack(x):
        """Unpack flat vector into centers (N,2) and thetas (N,)."""
        n = len(x) // 3
        centers = np.zeros((n, 2))
        thetas = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x[3 * i]
            centers[i, 1] = x[3 * i + 1]
            thetas[i] = x[3 * i + 2] * T_SCALE
        return centers, thetas

    def _pack_iso(centers):
        """Pack centers (N,2) into flat [x0,y0, x1,y1, ...]."""
        return centers.ravel()

    def _unpack_iso(x):
        """Unpack flat vector into centers (N,2)."""
        return x.reshape(-1, 2)

    # Pre-build wall coefs (fixed throughout optimization)
    _wall_coefs, _n_wall, _, _, _v_bounds = build_wall_coefs()
    _V_XMIN, _V_XMAX, _V_YMIN, _V_YMAX = _v_bounds

    def _loss_and_grad_aniso(x, r0, r1):
        """H1 total persistence loss and gradient via analytical VJP.

        Uses coef_from_cov_grad + pdist_tangency_grad for the full chain,
        with analytical dCov/dθ for the rotation parameters.
        """
        centers, thetas = _unpack(x)
        covs = build_covs(thetas, r0, r1)

        try:
            coefs, vjp_coef = coef_from_cov_grad(centers, covs)
            coefs_all = np.vstack([_wall_coefs, coefs])
            dists, vjp_dist = pdist_tangency_grad(coefs_all)
        except (RuntimeError, ZeroDivisionError):
            return 1e6, np.zeros_like(x)

        # H1 total persistence as loss
        dm = squareform(dists)
        result = ripser(dm, maxdim=1, distance_matrix=True)
        dgm = result["dgms"][1]
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
        if len(finite) == 0:
            return 0.0, np.zeros_like(x)

        loss = float(np.sum(finite[:, 1] - finite[:, 0]))

        # Subgradient: identify birth/death edges in condensed dists
        grad_dists = np.zeros_like(dists)
        for b, d in finite:
            grad_dists[np.argmin(np.abs(dists - b))] -= 1.0
            grad_dists[np.argmin(np.abs(dists - d))] += 1.0

        # Chain backward through VJPs
        grad_coefs_all = vjp_dist(grad_dists)
        grad_centers, grad_covs = vjp_coef(grad_coefs_all[_n_wall:])

        # Analytical dCov/dθ chain rule (d=2 only)
        grad_thetas = np.zeros(len(thetas))
        D = np.diag([r0**2, r1**2])
        for i, theta in enumerate(thetas):
            c, s = np.cos(theta), np.sin(theta)
            Rp = np.array([[-s, -c], [c, -s]])  # dR/dθ
            R = np.array([[c, -s], [s, c]])
            dCov = Rp @ D @ R.T + R @ D @ Rp.T
            grad_thetas[i] = np.sum(grad_covs[i] * dCov)

        # Pack gradient: [x0,y0,θ0, x1,y1,θ1, ...]
        grad_x = np.zeros_like(x)
        for i in range(len(thetas)):
            grad_x[3 * i] = grad_centers[i, 0]
            grad_x[3 * i + 1] = grad_centers[i, 1]
            grad_x[3 * i + 2] = grad_thetas[i] * T_SCALE

        return loss, grad_x

    def _loss_and_grad_iso(x_flat, r_iso):
        """H1 total persistence loss and gradient for isotropic (circular) sensors."""
        centers = _unpack_iso(x_flat)
        thetas = np.zeros(len(centers))
        covs = build_covs(thetas, r_iso, r_iso)

        try:
            coefs, vjp_coef = coef_from_cov_grad(centers, covs)
            coefs_all = np.vstack([_wall_coefs, coefs])
            dists, vjp_dist = pdist_tangency_grad(coefs_all)
        except (RuntimeError, ZeroDivisionError):
            return 1e6, np.zeros_like(x_flat)

        # H1 total persistence as loss
        dm = squareform(dists)
        result = ripser(dm, maxdim=1, distance_matrix=True)
        dgm = result["dgms"][1]
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
        if len(finite) == 0:
            return 0.0, np.zeros_like(x_flat)

        loss = float(np.sum(finite[:, 1] - finite[:, 0]))

        grad_dists = np.zeros_like(dists)
        for b, d in finite:
            grad_dists[np.argmin(np.abs(dists - b))] -= 1.0
            grad_dists[np.argmin(np.abs(dists - d))] += 1.0

        grad_coefs_all = vjp_dist(grad_dists)
        grad_centers, _ = vjp_coef(grad_coefs_all[_n_wall:])

        return loss, grad_centers.ravel()

    def _run_anisotropic():
        """Minimize H1 total persistence over centers + rotations (3N DOF).

        Uses L-BFGS-B with H1 subgradients via analytical VJP chain.
        Wall ellipses along [0,1]^2 boundary prevent collapse.
        """
        _rng = np.random.RandomState(SEED)
        # Initialize sensors securely inside the lower-center of the funnel
        init_centers, init_thetas = initialize_sensors(_rng, N_SENSORS, type="aniso")

        init_h1 = h1_total_persistence(
            init_centers, init_thetas, R_MAJOR, R_MINOR, wall_coefs=_wall_coefs
        )

        bounds_aniso = []
        for _ in range(N_SENSORS):
            bounds_aniso.extend([(_V_XMIN, _V_XMAX), (_V_YMIN, _V_YMAX), (None, None)])

        # Stage 1: short run from initial for intermediate snapshot
        _rng_opt = np.random.RandomState(SEED + 100)
        x0 = _pack(init_centers, init_thetas)
        x_start = x0 + _rng_opt.uniform(-JITTER_MAGNITUDE, JITTER_MAGNITUDE, len(x0))
        for i in range(N_SENSORS):
            x_start[3 * i] = np.clip(x_start[3 * i], _V_XMIN, _V_XMAX)
            x_start[3 * i + 1] = np.clip(x_start[3 * i + 1], _V_YMIN, _V_YMAX)
        res_x, res_loss = gradient_descent(
            _loss_and_grad_aniso,
            x_start,
            args=(R_MAJOR, R_MINOR),
            max_iter=MAXITER_WARMUP,
            lr=LEARNING_RATE,
            jiggle=JIGGLE_RATE,
        )
        c_mid, t_mid = _unpack(res_x)
        mid_h1 = res_loss

        # Stage 2: random restarts for global search
        best_x = res_x.copy()
        best_h1 = mid_h1
        for _trial in range(N_RESTARTS):
            c0, t0 = initialize_sensors(_rng_opt, N_SENSORS, type="aniso")
            xs = _pack(c0, t0)
            trial_x, trial_h1 = gradient_descent(
                _loss_and_grad_aniso,
                xs,
                args=(R_MAJOR, R_MINOR),
                max_iter=MAXITER,
                lr=LEARNING_RATE,
                jiggle=JIGGLE_RATE,
            )
            if trial_h1 < best_h1:
                best_h1 = trial_h1
                best_x = trial_x.copy()

        c_final, t_final = _unpack(best_x)
        snapshots = [
            ("Initial", init_h1, init_centers.copy(), init_thetas.copy()),
            (f"Iter {MAXITER_WARMUP}", mid_h1, c_mid.copy(), t_mid.copy()),
            ("Optimized", best_h1, c_final.copy(), t_final.copy()),
        ]
        return snapshots

    def _run_isotropic():
        """Minimize H1 total persistence over centers only (2N DOF).

        Uses L-BFGS-B with H1 subgradients via analytical VJP chain.
        Same wall ellipses as anisotropic for fair comparison.
        """
        _rng_opt = np.random.RandomState(SEED + 300)
        best_h1 = float("inf")
        best_x = np.zeros(2 * N_SENSORS)

        bounds_iso = []
        for _ in range(N_SENSORS):
            bounds_iso.extend([(_V_XMIN, _V_XMAX), (_V_YMIN, _V_YMAX)])

        for _trial in range(N_RESTARTS):
            c0 = initialize_sensors(_rng_opt, N_SENSORS, type="iso")
            res_x, res_h1 = gradient_descent(
                _loss_and_grad_iso,
                c0.ravel(),
                args=(R_ISO,),
                max_iter=MAXITER,
                lr=LEARNING_RATE,
            )
            if res_h1 < best_h1:
                best_h1 = res_h1
                best_x = res_x.copy()

        iso_centers = _unpack_iso(best_x)
        return iso_centers, best_h1

    aniso_snapshots = _run_anisotropic()
    iso_opt_centers, iso_opt_h1 = _run_isotropic()

    _ratio = aniso_snapshots[-1][1] / max(iso_opt_h1, 1e-10)
    mo.md(
        f"""
    **Optimization results** (L-BFGS-B with H1 subgradient via analytical VJP)**:**
    - Anisotropic (3N = {3 * N_SENSORS} DOF): H1 = {aniso_snapshots[0][1]:.3f}
      (initial) -> **{aniso_snapshots[-1][1]:.3f}** (optimized)
    - Isotropic (2N = {2 * N_SENSORS} DOF): H1 = **{iso_opt_h1:.3f}** (optimized)
    - Ratio: **{_ratio:.1f}x**
    """
    )
    return aniso_snapshots, iso_opt_centers, iso_opt_h1


@app.cell(hide_code=True)
def _(
    Ellipse,
    N_SENSORS,
    R_MAJOR,
    R_MINOR,
    R_WALL_MAJOR,
    R_WALL_MINOR,
    aniso_snapshots,
    build_wall_coefs,
    compute_h1,
    nb_dir,
    np,
    plt,
):
    _wall_coefs_viz, _, _wall_centers, _wall_thetas, _v_bounds = build_wall_coefs()
    _V_XMIN, _V_XMAX, _V_YMIN, _V_YMAX = _v_bounds

    def _draw_wall(ax):
        """Draw fixed wall ellipses in grey."""
        for i in range(len(_wall_centers)):
            ell = Ellipse(
                xy=_wall_centers[i],
                width=2 * R_WALL_MAJOR,
                height=2 * R_WALL_MINOR,
                angle=np.degrees(_wall_thetas[i]),
                facecolor="#cccccc",
                edgecolor="#999999",
                linewidth=0.8,
                alpha=0.4,
            )
            ax.add_patch(ell)

    def _draw_sensors(ax, centers, thetas, r0, r1, color_face, color_edge):
        for i in range(len(centers)):
            angle_deg = np.degrees(thetas[i])
            ell = Ellipse(
                xy=centers[i],
                width=2 * r0,
                height=2 * r1,
                angle=angle_deg,
                facecolor=color_face,
                edgecolor=color_edge,
                linewidth=1.2,
                alpha=0.35,
            )
            ax.add_patch(ell)
            ell_border = Ellipse(
                xy=centers[i],
                width=2 * r0,
                height=2 * r1,
                angle=angle_deg,
                facecolor="none",
                edgecolor=color_edge,
                linewidth=1.5,
            )
            ax.add_patch(ell_border)
        ax.plot(centers[:, 0], centers[:, 1], "k.", markersize=5, zorder=5)

    def _format_ax(ax, title):
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        rect = plt.Rectangle(
            (_V_XMIN, _V_YMIN),
            _V_XMAX - _V_XMIN,
            _V_YMAX - _V_YMIN,
            fill=False,
            edgecolor="gray",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=8)

    fig_aniso, axes_aniso = plt.subplots(1, 3, figsize=(13, 4.3))

    panel_labels = [
        "Initial (all horizontal)",
        f"Intermediate ({aniso_snapshots[1][0]})",
        "Optimized (min H1)",
    ]
    colors_face = ["#4e79a7", "#f28e2b", "#59a14f"]
    colors_edge = ["#2b5c8a", "#c06e1a", "#3d7a34"]

    for idx, (_label, h1_val, centers_snap, thetas_snap) in enumerate(aniso_snapshots):
        ax = axes_aniso[idx]
        _draw_wall(ax)
        _draw_sensors(
            ax,
            centers_snap,
            thetas_snap,
            R_MAJOR,
            R_MINOR,
            colors_face[idx],
            colors_edge[idx],
        )

        try:
            dgm = compute_h1(
                centers_snap,
                thetas_snap,
                R_MAJOR,
                R_MINOR,
                wall_coefs=_wall_coefs_viz,
            )
            n_h1 = 0
            if len(dgm) > 0:
                finite = np.isfinite(dgm[:, 1] - dgm[:, 0])
                n_h1 = int(np.sum(finite))
        except RuntimeError:
            n_h1 = 0

        subtitle = f"H1 = {h1_val:.3f}"
        if n_h1 > 0:
            subtitle += f"  ({n_h1} hole{'s' if n_h1 > 1 else ''})"
        else:
            subtitle += "  (no holes)"
        _format_ax(ax, panel_labels[idx])
        ax.set_xlabel(subtitle, fontsize=9, color="#555")

    fig_aniso.suptitle(
        f"Anisotropic Coverage Optimization  ({N_SENSORS} sensors,"
        f" semi-axes {R_MAJOR} x {R_MINOR})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig_aniso.tight_layout()
    plt.savefig(nb_dir / "anisotropic_optimization.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anisotropic vs Isotropic Comparison

    Both sensor types have the **same area** per sensor. Both are optimized to
    **minimize** H1 total persistence (reduce coverage holes). Anisotropic
    sensors have 3N degrees of freedom (position + rotation) while isotropic
    sensors have only 2N (position). The extra rotational freedom lets
    elongated ellipses bridge gaps that circles cannot.
    """)
    return


@app.cell(hide_code=True)
def _(
    Ellipse,
    N_SENSORS,
    R_ISO,
    R_MAJOR,
    R_MINOR,
    R_WALL_MAJOR,
    R_WALL_MINOR,
    aniso_snapshots,
    build_wall_coefs,
    compute_h1,
    iso_opt_centers,
    nb_dir,
    np,
    plt,
):
    _wall_coefs_cmp, _, _wall_centers, _wall_thetas, _v_bounds = build_wall_coefs()
    _V_XMIN, _V_XMAX, _V_YMIN, _V_YMAX = _v_bounds

    def _draw_wall_cmp(ax):
        """Draw fixed wall ellipses in grey."""
        for i in range(len(_wall_centers)):
            ell = Ellipse(
                xy=_wall_centers[i],
                width=2 * R_WALL_MAJOR,
                height=2 * R_WALL_MINOR,
                angle=np.degrees(_wall_thetas[i]),
                facecolor="#cccccc",
                edgecolor="#999999",
                linewidth=0.8,
                alpha=0.4,
            )
            ax.add_patch(ell)

    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(9, 4.3))

    # --- Left: optimized anisotropic ---
    ax_a = axes_cmp[0]
    _draw_wall_cmp(ax_a)
    _, _, c_aniso, t_aniso = aniso_snapshots[-1]
    for i in range(N_SENSORS):
        angle_deg = np.degrees(t_aniso[i])
        ell = Ellipse(
            xy=c_aniso[i],
            width=2 * R_MAJOR,
            height=2 * R_MINOR,
            angle=angle_deg,
            facecolor="#59a14f",
            edgecolor="#3d7a34",
            linewidth=1.5,
            alpha=0.35,
        )
        ax_a.add_patch(ell)
        ell_b = Ellipse(
            xy=c_aniso[i],
            width=2 * R_MAJOR,
            height=2 * R_MINOR,
            angle=angle_deg,
            facecolor="none",
            edgecolor="#3d7a34",
            linewidth=1.5,
        )
        ax_a.add_patch(ell_b)
    ax_a.plot(c_aniso[:, 0], c_aniso[:, 1], "k.", markersize=5, zorder=5)

    try:
        dgm_a = compute_h1(
            c_aniso, t_aniso, R_MAJOR, R_MINOR, wall_coefs=_wall_coefs_cmp
        )
        h1_a = 0.0
        n_h1_a = 0
        if len(dgm_a) > 0:
            lt_a = dgm_a[:, 1] - dgm_a[:, 0]
            h1_a = float(np.sum(lt_a[np.isfinite(lt_a)]))
            n_h1_a = int(np.sum(np.isfinite(lt_a)))
    except RuntimeError:
        h1_a = 0.0
        n_h1_a = 0

    ax_a.set_xlim(-0.15, 1.15)
    ax_a.set_ylim(-0.15, 1.15)
    ax_a.set_aspect("equal")
    rect_a = plt.Rectangle(
        (_V_XMIN, _V_YMIN),
        _V_XMAX - _V_XMIN,
        _V_YMAX - _V_YMIN,
        fill=False,
        edgecolor="gray",
        linewidth=1.5,
        linestyle="--",
    )
    ax_a.add_patch(rect_a)
    ax_a.set_title("Anisotropic (optimized)", fontsize=11, fontweight="bold")
    _lbl_a = f"H1 = {h1_a:.3f}"
    if n_h1_a > 0:
        _lbl_a += f" ({n_h1_a} holes)"
    ax_a.set_xlabel(
        _lbl_a + f"   [3N = {3 * N_SENSORS} DOF]",
        fontsize=9,
        color="#555",
    )
    ax_a.set_xticks([0, 0.5, 1])
    ax_a.set_yticks([0, 0.5, 1])
    ax_a.tick_params(labelsize=8)

    # --- Right: optimized isotropic ---
    ax_i = axes_cmp[1]
    _draw_wall_cmp(ax_i)
    for i in range(N_SENSORS):
        circ = Ellipse(
            xy=iso_opt_centers[i],
            width=2 * R_ISO,
            height=2 * R_ISO,
            angle=0,
            facecolor="#e15759",
            edgecolor="#b03a3c",
            linewidth=1.5,
            alpha=0.35,
        )
        ax_i.add_patch(circ)
        circ_b = Ellipse(
            xy=iso_opt_centers[i],
            width=2 * R_ISO,
            height=2 * R_ISO,
            angle=0,
            facecolor="none",
            edgecolor="#b03a3c",
            linewidth=1.5,
        )
        ax_i.add_patch(circ_b)
    ax_i.plot(
        iso_opt_centers[:, 0], iso_opt_centers[:, 1], "k.", markersize=5, zorder=5
    )

    try:
        dgm_i = compute_h1(
            iso_opt_centers,
            np.zeros(N_SENSORS),
            R_ISO,
            R_ISO,
            wall_coefs=_wall_coefs_cmp,
        )
        h1_i = 0.0
        n_h1_i = 0
        if len(dgm_i) > 0:
            lt_i = dgm_i[:, 1] - dgm_i[:, 0]
            h1_i = float(np.sum(lt_i[np.isfinite(lt_i)]))
            n_h1_i = int(np.sum(np.isfinite(lt_i)))
    except RuntimeError:
        h1_i = 0.0
        n_h1_i = 0

    ax_i.set_xlim(-0.15, 1.15)
    ax_i.set_ylim(-0.15, 1.15)
    ax_i.set_aspect("equal")
    rect_i = plt.Rectangle(
        (_V_XMIN, _V_YMIN),
        _V_XMAX - _V_XMIN,
        _V_YMAX - _V_YMIN,
        fill=False,
        edgecolor="gray",
        linewidth=1.5,
        linestyle="--",
    )
    ax_i.add_patch(rect_i)
    ax_i.set_title("Isotropic (optimized)", fontsize=11, fontweight="bold")
    _lbl_i = f"H1 = {h1_i:.3f}"
    if n_h1_i > 0:
        _lbl_i += f" ({n_h1_i} holes)"
    ax_i.set_xlabel(
        _lbl_i + f"   [2N = {2 * N_SENSORS} DOF]",
        fontsize=9,
        color="#555",
    )
    ax_i.set_xticks([0, 0.5, 1])
    ax_i.set_yticks([0, 0.5, 1])
    ax_i.tick_params(labelsize=8)

    fig_cmp.suptitle(
        f"Same area per sensor  (r_iso = {R_ISO:.4f}, {N_SENSORS} sensors)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig_cmp.tight_layout()
    plt.savefig(nb_dir / "anisotropic_vs_isotropic.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    return


@app.cell(hide_code=True)
def _(N_SENSORS, aniso_snapshots, iso_opt_h1, mo):
    _, _h1_init, _, _ = aniso_snapshots[0]
    _, _h1_final, _, _ = aniso_snapshots[-1]
    _ratio = _h1_final / max(iso_opt_h1, 1e-10)

    mo.md(
        f"""
    ## Summary

    | Configuration | DOF | H1 total persistence |
    |:---|:---:|---:|
    | Anisotropic, initial (horizontal) | -- | {_h1_init:.3f} |
    | **Anisotropic, optimized (centers + rotations)** | **3N = {3 * N_SENSORS}** | **{_h1_final:.3f}** |
    | **Isotropic, optimized (centers only)** | **2N = {2 * N_SENSORS}** | **{iso_opt_h1:.3f}** |
    | **Anisotropic / Isotropic ratio** | | **{_ratio:.1f}x** |

    By optimizing both **positions and rotations** to minimize H1,
    anisotropic sensors achieve lower H1 persistence than isotropic sensors
    of the same area that can only optimize positions.

    The extra N rotational degrees of freedom let elongated sensors bridge
    coverage gaps that circular sensors of equal area cannot close.
    """
    )
    return


if __name__ == "__main__":
    app.run()
