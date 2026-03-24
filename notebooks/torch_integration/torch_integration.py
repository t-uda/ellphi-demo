import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    from ellphi.grad import coef_from_cov_grad, pdist_tangency_grad

    nb_dir = Path(__file__).resolve().parent
    return (
        coef_from_cov_grad,
        mo,
        nb_dir,
        np,
        pdist_tangency_grad,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Integration for EllPHi

    This notebook demonstrates how to wrap EllPHi's numpy-based VJP functions
    as a **black-box differentiable operator** in PyTorch using
    `torch.autograd.Function`.

    The approach:
    1. Forward pass: convert torch tensors to numpy, call EllPHi, return torch tensor
    2. Backward pass: use stored VJP closures to propagate gradients back

    This lets us embed tangency-distance computation inside a standard PyTorch
    training loop and optimise with GPU-friendly optimizers like Adam.

    **Limitation**: because the inner computation is opaque to torch, we get
    first-order gradients only (no `torch.autograd.grad` of grad, no Hessians).
    """)
    return


@app.cell
def _(coef_from_cov_grad, np, pdist_tangency_grad, torch):
    class EllphiTangencyDistances(torch.autograd.Function):
        """Differentiable wrapper around EllPHi tangency distances."""

        @staticmethod
        def forward(ctx, centers, covs):
            c_np = centers.detach().cpu().numpy()
            cov_np = covs.detach().cpu().numpy()
            coefs, vjp_coef = coef_from_cov_grad(c_np, cov_np)
            dists, vjp_dist = pdist_tangency_grad(coefs)
            # Store VJP closures for backward
            ctx.vjp_coef = vjp_coef
            ctx.vjp_dist = vjp_dist
            return torch.from_numpy(dists.copy()).to(centers.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            grad_np = grad_output.detach().cpu().numpy()
            grad_coefs = ctx.vjp_dist(grad_np)
            grad_centers_np, grad_covs_np = ctx.vjp_coef(grad_coefs)
            grad_centers = torch.from_numpy(grad_centers_np.copy())
            grad_covs = torch.from_numpy(grad_covs_np.copy())
            return grad_centers.to(grad_output.dtype), grad_covs.to(grad_output.dtype)

    _ = np  # keep numpy in scope for downstream cells
    return (EllphiTangencyDistances,)


@app.cell
def _(EllphiTangencyDistances, np, torch):
    # --- Gradient check ---
    rng = np.random.RandomState(42)
    n_pts = 5

    centers_t = torch.tensor(
        rng.randn(n_pts, 2), dtype=torch.float64, requires_grad=True
    )
    # Build valid SPD covariance matrices via L @ L^T
    _L = np.eye(2)[None] * 0.3 + rng.randn(n_pts, 2, 2) * 0.1
    covs_np = np.einsum("nij,nkj->nik", _L, _L)
    covs_t = torch.tensor(covs_np, dtype=torch.float64, requires_grad=True)

    # Forward + backward
    dists_t = EllphiTangencyDistances.apply(centers_t, covs_t)
    loss_t = dists_t.sum()
    loss_t.backward()

    print("centers_t.grad shape:", centers_t.grad.shape)
    print("covs_t.grad shape:", covs_t.grad.shape)
    print("loss value:", loss_t.item())

    # Full numerical gradcheck (commented out — slow but useful for debugging)
    # torch.autograd.gradcheck(
    #     EllphiTangencyDistances.apply, (centers_t, covs_t), eps=1e-6
    # )
    return (centers_t, covs_np, covs_t, dists_t, loss_t, n_pts, rng)


@app.cell
def _(EllphiTangencyDistances, covs_np, n_pts, np, rng, torch):
    # --- Simple optimization: move centers to minimise total tangency distance ---
    centers_opt = torch.tensor(
        rng.randn(n_pts, 2), dtype=torch.float64, requires_grad=True
    )
    covs_fixed = torch.tensor(covs_np, dtype=torch.float64)  # keep covariances fixed

    optimizer = torch.optim.Adam([centers_opt], lr=0.05)
    loss_history = []

    for _step in range(60):
        optimizer.zero_grad()
        d = EllphiTangencyDistances.apply(centers_opt, covs_fixed)
        loss_val = d.sum()
        loss_val.backward()
        optimizer.step()
        loss_history.append(loss_val.item())

    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final   loss: {loss_history[-1]:.4f}")
    print(f"Optimised centers:\n{np.round(centers_opt.detach().numpy(), 4)}")
    return (centers_opt, covs_fixed, loss_history, optimizer)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    We wrapped EllPHi's `coef_from_cov_grad` and `pdist_tangency_grad` VJP
    functions inside a `torch.autograd.Function`. This gives us:

    - **Seamless backward pass** through tangency-distance computation
    - Compatibility with any PyTorch optimizer (Adam, SGD, L-BFGS, ...)
    - Easy integration into larger differentiable pipelines

    **Caveats**:
    - The inner computation runs on CPU/numpy; no GPU acceleration for the
      EllPHi part itself.
    - Only first-order derivatives are available (no higher-order autograd).
    """)
    return


if __name__ == "__main__":
    app.run()
