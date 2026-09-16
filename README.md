# PINN Schrödinger — Quantum Harmonic Oscillator

A Physics-Informed Neural Network (PINN) built from scratch in PyTorch to solve the **1D time-dependent Schrödinger equation** for a particle in a quantum harmonic oscillator potential — with no training data.

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \psi}{\partial x^2} + V(x)\psi, \qquad V(x) = \tfrac{1}{2}x^2$$

with $\hbar = m = 1$.

---

## How it works

Traditional neural networks learn from labeled data. This network learns from **physics**. The loss function directly penalizes violations of the Schrödinger equation at randomly sampled spacetime points — no dataset required.

Since $\psi(x,t)$ is complex-valued, the network outputs two real scalars:

$$\psi(x,t) = u(x,t) + iv(x,t)$$

Substituting into the PDE yields two coupled real equations that must both be satisfied:

$$f_{\text{real}} = \frac{\partial u}{\partial t} + \frac{1}{2}\frac{\partial^2 v}{\partial x^2} - V(x)\,v = 0$$

$$f_{\text{imag}} = \frac{\partial v}{\partial t} - \frac{1}{2}\frac{\partial^2 u}{\partial x^2} + V(x)\,u = 0$$

All derivatives are computed via `torch.autograd.grad` with `create_graph=True`, enabling second-order differentiation through the network.

---

## Results

After 2000 epochs the total residual loss reaches **3.7 × 10⁻³**, with IC and BC losses three orders of magnitude smaller — confirming the network correctly learned the QHO ground state $\psi_0(x) = \pi^{-1/4} e^{-x^2/2}$ and its time evolution $\psi(x,t) = \psi_0(x)\,e^{-iE_0 t}$ (where $E_0 = \tfrac{1}{2}$).

| Metric | Value |
|---|---|
| Total loss at epoch 2000 | 3.7 × 10⁻³ |
| Norm $\int\|\psi\|^2\,dx$ at $t=0$ | 0.9980 (exact: 1.0) |
| Norm $\int\|\psi\|^2\,dx$ at $t=1$ | 0.9848 (exact: 1.0) |

![Wavefunction Evolution](wavefunction_evolution.png)

*Top-left: probability density heatmap. Top-right: real and imaginary parts at selected times. Bottom-left: norm conservation. Bottom-right: density slices.*

---

## Project structure

```
.
├── model.py        # MLP architecture: (x, t) → (u, v)
├── pinn.py         # PDE residual, IC loss, BC loss, total loss
├── train.py        # Training loop (Adam, LHS sampling, 2000 epochs)
├── visualize.py    # Loads checkpoint, plots |ψ|² heatmap
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (saves schrodinger_pinn.pt)
python train.py

# 3. Visualize (saves wavefunction_evolution.png)
python visualize.py
```

---

## Loss breakdown

The total loss has three weighted terms:

| Term | Weight | Purpose |
|---|---|---|
| PDE residual $f_{\text{real}}^2 + f_{\text{imag}}^2$ | 1.0 | Enforce Schrödinger equation at interior points |
| Initial condition | 10.0 | Match ground-state Gaussian at $t = 0$ |
| Boundary condition | 10.0 | Enforce $\psi(\pm 5, t) = 0$ |

IC and BC are upweighted so the network satisfies the constraints before fitting the interior PDE.

---

## Sampling strategy

Interior collocation points are drawn via **Latin Hypercube Sampling (LHS)** at every epoch, guaranteeing even marginal coverage across both $x$ and $t$ without repetition. A new sample is drawn each epoch so the network never memorises a fixed grid.

| Point set | Count | Method |
|---|---|---|
| Interior collocation | 2000 | LHS over $x \in [-5,5]$, $t \in [0,1]$ |
| Initial condition | 200 | Uniform grid in $x$ at $t = 0$ |
| Boundary condition | 200 | Uniform grid in $t$ at $x = \pm 5$ |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Matplotlib
