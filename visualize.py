"""
Visualize the PINN solution to the 1D Schrödinger equation.

Loads the trained checkpoint, evaluates ψ(x,t) on a fine grid, and
produces a 2D heatmap of the probability density |ψ|² = u² + v².
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import SchrodingerMLP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT  = "schrodinger_pinn.pt"
OUTPUT_FILE = "wavefunction_evolution.png"
NX, NT      = 500, 300          # grid resolution
X_MIN, X_MAX = -5.0,  5.0
T_MIN, T_MAX =  0.0,  1.0
DEVICE = torch.device("cpu")    # inference is fast on CPU

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = SchrodingerMLP(hidden_layers=5, hidden_dim=64).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")

# ---------------------------------------------------------------------------
# Build evaluation grid
# ---------------------------------------------------------------------------
x_vals = torch.linspace(X_MIN, X_MAX, NX, device=DEVICE)  # (NX,)
t_vals = torch.linspace(T_MIN, T_MAX, NT, device=DEVICE)  # (NT,)

# meshgrid → (NT, NX) each, then flatten to (NT*NX, 1)
T_grid, X_grid = torch.meshgrid(t_vals, x_vals, indexing="ij")
x_flat = X_grid.reshape(-1, 1)
t_flat = T_grid.reshape(-1, 1)

# ---------------------------------------------------------------------------
# Forward pass (no gradients needed)
# ---------------------------------------------------------------------------
with torch.no_grad():
    u_flat, v_flat = model(x_flat, t_flat)

prob_flat = u_flat ** 2 + v_flat ** 2          # |ψ|²

# Reshape back to (NT, NX) numpy arrays
prob  = prob_flat.reshape(NT, NX).cpu().numpy()
u_np  = u_flat.reshape(NT, NX).cpu().numpy()
v_np  = v_flat.reshape(NT, NX).cpu().numpy()
x_np  = x_vals.cpu().numpy()
t_np  = t_vals.cpu().numpy()

# ---------------------------------------------------------------------------
# Exact ground-state norm for reference annotation
# ---------------------------------------------------------------------------
exact_norm_per_row = np.pi ** (-0.5) * np.ones(NT)   # ∫|ψ₀|²dx = 1 analytically
dx = (X_MAX - X_MIN) / (NX - 1)
pinn_norm = prob.sum(axis=1) * dx                     # numerical ∫|ψ|²dx per t-slice

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 9),
    gridspec_kw={"height_ratios": [2.2, 1]},
)
fig.suptitle(
    r"PINN Solution — 1D Schrödinger Equation (QHO),  $\hbar = m = 1$",
    fontsize=14, fontweight="bold", y=0.98,
)

# ── (0,0) Probability density heatmap ──────────────────────────────────────
ax0 = axes[0, 0]
im = ax0.pcolormesh(
    x_np, t_np, prob,
    cmap="inferno", shading="auto",
)
cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_label(r"$|\psi(x,t)|^2$", fontsize=11)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax0.set_xlabel(r"$x$",  fontsize=12)
ax0.set_ylabel(r"$t$",  fontsize=12)
ax0.set_title(r"Probability Density $|\psi(x,t)|^2$", fontsize=12)

# ── (0,1) Real and imaginary parts at selected times ───────────────────────
ax1 = axes[0, 1]
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
colors    = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_slices)))
for ti, col in zip(t_slices, colors):
    idx = int(round(ti * (NT - 1)))
    ax1.plot(x_np, u_np[idx], color=col, lw=1.8,  label=f"u  t={ti:.2f}")
    ax1.plot(x_np, v_np[idx], color=col, lw=1.8, ls="--")
ax1.axhline(0, color="gray", lw=0.6, ls=":")
ax1.set_xlabel(r"$x$", fontsize=12)
ax1.set_ylabel(r"$u(x,t)$ [—]   $v(x,t)$ [- -]", fontsize=10)
ax1.set_title(r"Real $u$ and Imaginary $v$ Parts", fontsize=12)
ax1.legend(fontsize=8, ncol=2, loc="upper right")

# ── (1,0) Norm conservation over time ──────────────────────────────────────
ax2 = axes[1, 0]
ax2.plot(t_np, pinn_norm, color="royalblue", lw=2, label=r"PINN $\int|\psi|^2\,dx$")
ax2.axhline(1.0, color="tomato", lw=1.5, ls="--", label="Exact = 1")
ax2.set_xlabel(r"$t$", fontsize=12)
ax2.set_ylabel(r"$\int|\psi|^2\,dx$", fontsize=12)
ax2.set_title("Norm Conservation", fontsize=12)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 2)

# ── (1,1) Probability density slices ───────────────────────────────────────
ax3 = axes[1, 1]
for ti, col in zip(t_slices, colors):
    idx = int(round(ti * (NT - 1)))
    ax3.plot(x_np, prob[idx], color=col, lw=1.8, label=f"t={ti:.2f}")
ax3.set_xlabel(r"$x$", fontsize=12)
ax3.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=12)
ax3.set_title(r"$|\psi|^2$ Slices at Selected Times", fontsize=12)
ax3.legend(fontsize=8, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {OUTPUT_FILE}")
print(f"Grid: {NX} x-points × {NT} t-points")
print(f"Norm at t=0:   {pinn_norm[0]:.4f}  (exact: 1.0000)")
print(f"Norm at t=0.5: {pinn_norm[NT//2]:.4f}  (exact: 1.0000)")
print(f"Norm at t=1.0: {pinn_norm[-1]:.4f}  (exact: 1.0000)")
