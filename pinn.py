"""
PDE residual, boundary, and initial condition losses for the 1D
time-dependent Schrödinger equation:

    i ∂ψ/∂t = -½ ∂²ψ/∂x² + V(x) ψ

with ħ = m = 1 and V(x) = ½ x² (quantum harmonic oscillator).

Splitting ψ = u + iv into real and imaginary parts gives two coupled PDEs:

    f_real = ∂u/∂t + ½ ∂²v/∂x² - V(x)·v = 0
    f_imag = ∂v/∂t - ½ ∂²u/∂x² + V(x)·u = 0

Initial condition (ground state of QHO):
    ψ(x, 0) = (1/π)^(1/4) · exp(-x²/2)
    → u(x, 0) = (1/π)^(1/4) · exp(-x²/2),  v(x, 0) = 0

Boundary conditions (ψ → 0 as |x| → ∞, enforced at finite x = ±x_max):
    u(±x_max, t) = 0,  v(±x_max, t) = 0
"""

import math
import torch
from model import SchrodingerMLP


# ---------------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------------

def potential(x: torch.Tensor) -> torch.Tensor:
    """Quantum harmonic oscillator: V(x) = ½ x²."""
    return 0.5 * x ** 2


# ---------------------------------------------------------------------------
# Derivative helper
# ---------------------------------------------------------------------------

def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    First-order derivative of `output` w.r.t. `inp`.
    create_graph=True keeps the computational graph so that higher-order
    derivatives and backprop through the residual both work correctly.
    """
    return torch.autograd.grad(
        output,
        inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def pde_residual(
    model: SchrodingerMLP,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Schrödinger PDE residuals at interior collocation points.

    x and t must have requires_grad=True so that autograd can differentiate
    the network outputs with respect to them.

    Args:
        model : SchrodingerMLP
        x     : spatial collocation points, shape (N, 1), requires_grad=True
        t     : time    collocation points, shape (N, 1), requires_grad=True

    Returns:
        f_real : ∂u/∂t + ½ ∂²v/∂x² - V·v,  shape (N, 1)
        f_imag : ∂v/∂t - ½ ∂²u/∂x² + V·u,  shape (N, 1)
    """
    u, v = model(x, t)

    # --- temporal derivatives ---
    u_t = _grad(u, t)   # ∂u/∂t
    v_t = _grad(v, t)   # ∂v/∂t

    # --- first spatial derivatives ---
    u_x  = _grad(u,   x)   # ∂u/∂x
    v_x  = _grad(v,   x)   # ∂v/∂x

    # --- second spatial derivatives ---
    u_xx = _grad(u_x, x)   # ∂²u/∂x²
    v_xx = _grad(v_x, x)   # ∂²v/∂x²

    V = potential(x)

    f_real = u_t + 0.5 * v_xx - V * v
    f_imag = v_t - 0.5 * u_xx + V * u

    return f_real, f_imag


def pde_loss(
    model: SchrodingerMLP,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared PDE residual: L_pde = mean(f_real²) + mean(f_imag²)."""
    f_real, f_imag = pde_residual(model, x, t)
    return torch.mean(f_real ** 2) + torch.mean(f_imag ** 2)


# ---------------------------------------------------------------------------
# Initial condition loss
# ---------------------------------------------------------------------------

def ground_state(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    QHO ground-state wavefunction: ψ₀(x) = (1/π)^(1/4) · exp(-x²/2).
    Returns (u0, v0) — real and imaginary parts at t = 0.
    """
    norm = math.pi ** (-0.25)          # (1/π)^(1/4)
    u0 = norm * torch.exp(-0.5 * x ** 2)
    v0 = torch.zeros_like(u0)
    return u0, v0


def ic_loss(
    model: SchrodingerMLP,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
) -> torch.Tensor:
    """
    Initial condition loss at t = 0.

    Args:
        x_ic : spatial points at t = 0, shape (N_ic, 1)
        t_ic : all-zero time tensor,     shape (N_ic, 1)

    Returns:
        L_ic = mean((u - u0)²) + mean((v - v0)²)
    """
    u_pred, v_pred = model(x_ic, t_ic)
    u0, v0 = ground_state(x_ic)
    return torch.mean((u_pred - u0) ** 2) + torch.mean((v_pred - v0) ** 2)


# ---------------------------------------------------------------------------
# Boundary condition loss
# ---------------------------------------------------------------------------

def bc_loss(
    model: SchrodingerMLP,
    t_bc: torch.Tensor,
    x_max: float = 5.0,
) -> torch.Tensor:
    """
    Dirichlet boundary condition loss: ψ(±x_max, t) = 0.

    The wavefunction decays exponentially for large |x|, so enforcing
    u = v = 0 at ±x_max is physically justified when x_max is large enough
    (x_max = 5 covers ~5σ of the ground-state Gaussian).

    Args:
        t_bc  : time points at the boundary, shape (N_bc, 1)
        x_max : domain half-width

    Returns:
        L_bc = mean(u_left²) + mean(v_left²) + mean(u_right²) + mean(v_right²)
    """
    n = t_bc.shape[0]
    device = t_bc.device

    x_left  = torch.full((n, 1), -x_max, dtype=t_bc.dtype, device=device)
    x_right = torch.full((n, 1),  x_max, dtype=t_bc.dtype, device=device)

    u_l, v_l = model(x_left,  t_bc)
    u_r, v_r = model(x_right, t_bc)

    return (
        torch.mean(u_l ** 2) + torch.mean(v_l ** 2) +
        torch.mean(u_r ** 2) + torch.mean(v_r ** 2)
    )


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def total_loss(
    model: SchrodingerMLP,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    x_ic:  torch.Tensor,
    t_ic:  torch.Tensor,
    t_bc:  torch.Tensor,
    lambda_pde: float = 1.0,
    lambda_ic:  float = 1.0,
    lambda_bc:  float = 1.0,
    x_max: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted sum of all loss terms.

    Returns:
        loss      : scalar total loss
        L_pde     : PDE residual loss (unweighted)
        L_ic      : initial condition loss (unweighted)
        L_bc      : boundary condition loss (unweighted)
    """
    L_pde = pde_loss(model, x_col, t_col)
    L_ic  = ic_loss(model, x_ic, t_ic)
    L_bc  = bc_loss(model, t_bc, x_max)

    loss = lambda_pde * L_pde + lambda_ic * L_ic + lambda_bc * L_bc
    return loss, L_pde, L_ic, L_bc
