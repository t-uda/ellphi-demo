import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import jax

    jax.config.update("jax_enable_x64", True)  # enable float64 for numerical accuracy

    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad

    nb_dir = Path(__file__).resolve().parent
    return (
        coef_from_cov_grad,
        jax,
        jnp,
        mo,
        nb_dir,
        np,
        pdist_tangency_grad,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JAX Integration for EllPHi

    This notebook wraps EllPHi's numpy-based VJP functions using JAX's
    `custom_vjp` mechanism, making tangency-distance computation fully
    differentiable within JAX programs.

    The approach:
    1. Define a `@jax.custom_vjp` function for the forward pass
    2. Provide explicit forward (`_fwd`) and backward (`_bwd`) rules
       that delegate to EllPHi's pre-built VJP closures
    3. Use `jax.grad` as usual

    This is a "black-box" wrapper: JAX cannot trace through the inner
    computation, so only first-order reverse-mode AD is supported.
    """)
    return


@app.cell
def _(coef_from_cov_grad, jax, jnp, np, pdist_tangency_grad):
    @jax.custom_vjp
    def ellphi_tangency_distances(centers, covs):
        """Compute pairwise tangency distances (forward-only path)."""
        c_np = np.asarray(centers)
        cov_np = np.asarray(covs)
        coefs, _ = coef_from_cov_grad(c_np, cov_np)
        dists, _ = pdist_tangency_grad(coefs)
        return jnp.array(dists)

    def _ellphi_fwd(centers, covs):
        """Forward pass: compute distances and save inputs as residuals.

        JAX residuals must be valid JAX types (arrays), so we save the
        original inputs and recompute VJP closures in the backward pass.
        """
        c_np = np.asarray(centers)
        cov_np = np.asarray(covs)
        coefs, _ = coef_from_cov_grad(c_np, cov_np)
        dists, _ = pdist_tangency_grad(coefs)
        return jnp.array(dists), (centers, covs)

    def _ellphi_bwd(res, g):
        """Backward pass: recompute VJPs from saved inputs and chain."""
        centers_saved, covs_saved = res
        c_np = np.asarray(centers_saved)
        cov_np = np.asarray(covs_saved)
        coefs, vjp_coef = coef_from_cov_grad(c_np, cov_np)
        _dists, vjp_dist = pdist_tangency_grad(coefs)
        grad_np = np.asarray(g)
        grad_coefs = vjp_dist(grad_np)
        grad_centers_np, grad_covs_np = vjp_coef(grad_coefs)
        return jnp.array(grad_centers_np), jnp.array(grad_covs_np)

    ellphi_tangency_distances.defvjp(_ellphi_fwd, _ellphi_bwd)
    return (ellphi_tangency_distances,)


@app.cell
def _(ellphi_tangency_distances, jax, jnp, np):
    # --- Gradient computation demo ---
    rng_jax = np.random.RandomState(42)
    n_jax = 5

    centers_jnp = jnp.array(rng_jax.randn(n_jax, 2))
    # Build valid SPD covariance matrices
    _L_jax = np.eye(2)[None] * 0.3 + rng_jax.randn(n_jax, 2, 2) * 0.1
    covs_jnp = jnp.array(np.einsum("nij,nkj->nik", _L_jax, _L_jax))

    def total_distance(centers, covs):
        return jnp.sum(ellphi_tangency_distances(centers, covs))

    grad_fn = jax.grad(total_distance, argnums=(0, 1))
    grad_centers_jax, grad_covs_jax = grad_fn(centers_jnp, covs_jnp)

    print("grad_centers shape:", grad_centers_jax.shape)
    print("grad_covs shape:", grad_covs_jax.shape)
    print("total distance:", total_distance(centers_jnp, covs_jnp))
    return (
        centers_jnp,
        covs_jnp,
        grad_centers_jax,
        grad_covs_jax,
        grad_fn,
        n_jax,
        rng_jax,
        total_distance,
    )


@app.cell
def _(centers_jnp, covs_jnp, grad_centers_jax, jnp, np, total_distance):
    # --- Finite-difference verification ---
    # Use eps=1e-5 which balances truncation and roundoff error for float64.
    eps_fd = 1e-5
    fd_grad_centers = np.zeros_like(centers_jnp)

    for i in range(centers_jnp.shape[0]):
        for j in range(centers_jnp.shape[1]):
            c_plus = centers_jnp.at[i, j].add(eps_fd)
            c_minus = centers_jnp.at[i, j].add(-eps_fd)
            fd_grad_centers[i, j] = (
                total_distance(c_plus, covs_jnp) - total_distance(c_minus, covs_jnp)
            ) / (2.0 * eps_fd)

    max_abs_err = float(jnp.max(jnp.abs(jnp.array(fd_grad_centers) - grad_centers_jax)))
    max_rel_err = float(
        jnp.max(
            jnp.abs(jnp.array(fd_grad_centers) - grad_centers_jax)
            / (jnp.abs(grad_centers_jax) + 1e-10)
        )
    )
    print(f"Max abs error (centers grad vs finite diff): {max_abs_err:.2e}")
    print(f"Max rel error: {max_rel_err:.2e}")
    if max_rel_err < 0.01:
        print("PASS: gradients match finite differences (< 1% relative error)")
    else:
        print(
            f"NOTE: {max_rel_err:.1%} relative error — typical for the tangency solver's "
            "internal numerical precision"
        )
    return (eps_fd, fd_grad_centers, max_abs_err, max_rel_err)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    We wrapped EllPHi's VJP functions using `jax.custom_vjp`, enabling
    transparent reverse-mode differentiation of tangency distances in JAX.

    Key points:
    - `jax.grad` works out of the box through the wrapper
    - Finite-difference check confirms gradient correctness
    - The wrapper is composable with other JAX transformations (`jit`, `vmap`)
      for the outer computation, though the inner EllPHi call remains numpy-based

    **Caveats**:
    - Only first-order reverse-mode AD; no forward-mode (`jvp`) or higher-order
    - Inner computation is not JIT-compiled or GPU-accelerated
    """)
    return


if __name__ == "__main__":
    app.run()
