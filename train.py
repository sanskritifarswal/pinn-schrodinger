"""
Training script for the Schrödinger PINN.

Sampling strategy
-----------------
Interior collocation : Latin Hypercube Sampling (LHS) over x ∈ [-5, 5], t ∈ [0, 1].
Initial condition    : uniform grid in x at t = 0.
Boundary condition   : uniform grid in t at x = ±5.

All point sets are re-sampled every epoch so the network sees a fresh
distribution of collocation points rather than memorising a fixed grid.
"""

import torch
from model import SchrodingerMLP
from pinn import total_loss

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 2000
LR          = 1e-3
N_COL       = 2000   # interior collocation points
N_IC        = 200    # initial-condition points
N_BC        = 200    # boundary-condition points (per boundary per epoch)

X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX =  0.0, 1.0

LAMBDA_PDE = 1.0
LAMBDA_IC  = 10.0   # upweight constraints so the network first satisfies IC/BC
LAMBDA_BC  = 10.0

LOG_EVERY  = 200

# ---------------------------------------------------------------------------
# Model & optimiser
# ---------------------------------------------------------------------------
model = SchrodingerMLP(hidden_layers=5, hidden_dim=64).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Device : {DEVICE}")
print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
print("-" * 60)


# ---------------------------------------------------------------------------
# LHS sampler
# ---------------------------------------------------------------------------

def lhs_sample(n: int, x_lo: float, x_hi: float,
               t_lo: float, t_hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2-D Latin Hypercube Sample returning (x, t) tensors of shape (n, 1).

    Each of the n intervals in [0,1] is sampled once per dimension, then
    the two columns are independently shuffled — this guarantees even
    marginal coverage along both x and t.
    """
    # stratified sample in [0, 1]
    perm_x = torch.randperm(n)
    perm_t = torch.randperm(n)

    x_unit = (perm_x.float() + torch.rand(n)) / n   # shape (n,)
    t_unit = (perm_t.float() + torch.rand(n)) / n

    x = x_unit * (x_hi - x_lo) + x_lo
    t = t_unit * (t_hi - t_lo) + t_lo

    x = x.unsqueeze(1).to(DEVICE)   # (n, 1)
    t = t.unsqueeze(1).to(DEVICE)
    return x, t


def uniform_sample_1d(n: int, lo: float, hi: float) -> torch.Tensor:
    """Uniformly spaced 1-D grid, shape (n, 1)."""
    return torch.linspace(lo, hi, n, device=DEVICE).unsqueeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimiser.zero_grad()

    # --- interior collocation points (LHS, fresh every epoch) ---
    x_col, t_col = lhs_sample(N_COL, X_MIN, X_MAX, T_MIN, T_MAX)
    x_col = x_col.requires_grad_(True)
    t_col = t_col.requires_grad_(True)

    # --- initial condition: t = 0, x uniform ---
    x_ic = uniform_sample_1d(N_IC, X_MIN, X_MAX)
    t_ic = torch.zeros(N_IC, 1, device=DEVICE)

    # --- boundary condition: x = ±5, t uniform ---
    t_bc = uniform_sample_1d(N_BC, T_MIN, T_MAX)

    loss, L_pde, L_ic, L_bc = total_loss(
        model,
        x_col, t_col,
        x_ic,  t_ic,
        t_bc,
        lambda_pde=LAMBDA_PDE,
        lambda_ic=LAMBDA_IC,
        lambda_bc=LAMBDA_BC,
        x_max=X_MAX,
    )

    loss.backward()
    optimiser.step()

    if epoch % LOG_EVERY == 0 or epoch == 1:
        print(
            f"Epoch {epoch:>5d} | "
            f"Total {loss.item():.4e} | "
            f"PDE {L_pde.item():.4e} | "
            f"IC {L_ic.item():.4e} | "
            f"BC {L_bc.item():.4e}"
        )

print("-" * 60)
print("Training complete.")

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "schrodinger_pinn.pt")
print("Model saved to schrodinger_pinn.pt")
