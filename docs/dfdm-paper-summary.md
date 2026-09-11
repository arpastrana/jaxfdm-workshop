# DFDM paper — theoretical foundation for the workshop

Summary of Pastrana, Oktay, Bletzinger, Adams, Adriaenssens,
*"Differentiable force density method for the design of lightweight structures,"*
Computer Methods in Applied Mechanics and Engineering 458 (2026) 118783.

Source: `pastrana-dfdm-cmame-2026.pdf`. Sections 2 and 3 are the theoretical
backbone of the workshop; equation numbers below are the paper's.

## The one-sentence thesis

Form-finding is a **forward** map from parameters to geometry, but design is an
**inverse** problem. Making the FDM differentiable lets gradient-based
optimization invert it, so a designer can state what they want and have the
method find the parameters that deliver it while equilibrium is guaranteed
throughout.

---

## Section 2.1 — The form-finding problem

**This is the section the workshop's opening demo should mirror.** The paper
introduces everything on a **four-bar system with a single free node** — one
free vertex `x₁ ∈ R³` connected to four supports `s_j`, carrying a load `p₁`.

With prescribed internal forces `f_j`, the residual is **nonlinear**, because the
bar length `l_j = ||x₁ − s_j||` depends on the unknown:

```
(1)   r₁(x₁) = Σⱼ (fⱼ / lⱼ)(x₁ − sⱼ) − p₁ = 0
```

Prescribing the **force density** instead, the ratio of force to length,

```
(3)   qⱼ = fⱼ / lⱼ
```

makes the residual **linear** in `x₁`:

```
(2)   r₁(x₁) = Σⱼ qⱼ (x₁ − sⱼ) − p₁ = 0
```

Both are solved by the same Newton step:

```
(4)   (∂r₁/∂x₁) Δx = −r₁
(5)   x₁* = x₁ − K⁻¹ r₁
```

but the force-density form converges in **one** step because its Jacobian is a
constant diagonal matrix:

```
(6)   K = diag(Σⱼ qⱼ,  Σⱼ qⱼ,  Σⱼ qⱼ)
```

Expanding the single step gives a closed form:

```
(7)   x₁* = (Σⱼ qⱼ sⱼ + p₁) / (Σⱼ qⱼ)
```

**Key insight from Eq. 7:** the equilibrium is *independent of the starting
position* `x₁`. No initial guess, no divergence, no tuning — as long as
`Σⱼ qⱼ ≠ 0`.

Figure 2 shows the worked example: `x₁ = [0, 0, 0]ᵀ → x₁* = [0.15, 0, 0.58]ᵀ`,
with a force polygon that is open at the start and closed at equilibrium
(graphic statics). **These are exactly the numbers in `01_form_finding/fourbar.py`.**

## Section 2.2 — Mechanical model (energy view)

Motivates the FDM from minimum potential energy rather than force balance.
Bar internal energy from the second Piola-Kirchhoff force and Green-Lagrange
strain, integrated over the undeformed length `l₀`:

```
(8)   π_int = ∫ f_PK2 ε_GL dl = (f_PK2 / 2l₀)(l² − l₀²)
(9)   ε_GL  = (l² − l₀²) / 2l₀²          # geometrically nonlinear
(10)  Π = Π_int + Π_ext → min
```

Setting the first variation to zero recovers the equilibrium condition, and the
deformed lengths **cancel algebraically**:

```
(13)  Σⱼ (f_PK2 / l₀)ⱼ (x₁ − sⱼ) − p₁ = 0
(14)  q = f_PK2 / l₀
```

Three things worth teaching from this:

- The force density is a constant ratio to the **undeformed** length, which is
  what linearizes the energy variation.
- The residual `r` is the **gradient** of the potential energy; the stiffness
  `K` is its **Hessian**. Newton's method on the residual is Newton's method on
  the energy.
- Eq. 13 is singular for co-planar force vectors; the reparametrization by `q`
  removes that singularity.

## Section 2.3 — The general force density method

Generalizes to any pin-jointed bar system on a graph `G` with `n_v` vertices and
`n_e` edges.

```
(15)  C̃ ∈ {−1, 0, 1}^(n_e × n_v)     connectivity matrix
(16)  C  = I₃ ⊗ C̃                    block form for 3D
(17)  C  = [C_u  C_s]                 split into free and fixed columns
```

Parameters are `θ = [q, p, s]` — force densities, applied loads, support
positions — totalling `n_p = n_e + 3n_u + 3n_s`.

```
(19)  r(x, θ) = K(θ) x − b(θ) = 0
(20)  K(θ)    = C_uᵀ Q C_u,   Q = I₃ ⊗ diag(q)
(21)  b(θ)    = p − d(θ)
(22)  d(θ)    = C_uᵀ Q C_s s
(23)  x(θ)    = K(θ)⁻¹ b(θ)          # one linear solve
```

Completing the state:

```
(24)  l = √((U ⊙ U) 1)                edge lengths
(25)  f = q ⊙ l                       edge forces
(26)  t = p_s − C_sᵀ Q u              support reactions
```

Mixed-sign force densities can put zeros on `K`'s diagonal, making trusses and
tensegrities singular — a real caveat, handled by regularization outside the
paper's scope.

> Workshop note: plan.md forbids connectivity matrices in the introductory
> derivation. Sections 2.1 and 2.2 are the derivation to teach; Section 2.3 is
> what `jax_fdm.equilibrium.fdm()` does under the hood.

## Section 2.4 — Shape-dependent loads

Constant loads misrepresent reality when the load *is* the shape (self-weight,
pressure, wind). The loads become a function of geometry:

```
(27)  r(x, θ) = K(θ)x − b(x, θ) = 0,   b(x, θ) = p(x, θ) − d(θ)
```

Solved with the **updated reference strategy (URS)**: hold `q` fixed, recompute
loads from the current geometry, re-solve, repeat to a fixed point.

```
(28)  p_i^(t+1) = Σ_{k ∈ F(i)} a_k^(t) ψ_k^(t)     tributary-area loads
(29)  face normals via the shoelace formula
(32)  x^(t+1) = K(θ)⁻¹ b(x^(t), θ)
(34)  Δx = Σ ||x_i^(t+1) − x_i^(t)|| ≤ η           convergence
```

Figure 5 is a striking result for a lecture: **an arch under constant loads is
roughly half the height of the true equilibrium under updated self-weight.**
Convergence takes ~6 URS iterations for scale-dependent loads, ~12 when the
load direction also follows the surface.

## Section 2.5 — Inverse problems by gradient-based optimization

```
(35)  L(θ, x) = Σᵢ αᵢ gᵢ(x(θ))                     weighted sum of goals
(36)  min L(θ)  s.t.  equality h, inequality m, bounds θ_lb ≤ θ ≤ θ_ub,  r(θ) = 0
(37)  θ^(s+1) = θ^(s) − μ ∇_θ L(θ^(s))
```

The critical architectural idea, and the one worth stating explicitly in the
workshop: this is a **bilevel** scheme. The inner loop solves equilibrium with
the FDM; the outer loop tunes `θ` so the equilibrium geometry meets the goals.
Because they are decoupled, **every candidate the optimizer sees is already in
equilibrium** — the search happens strictly inside the space of funicular
geometries. This is the NAND approach (nested analysis and design), as opposed
to SAND, where combining equilibrium and goals into one objective can converge
to something that is neither in equilibrium nor on target.

## Sections 2.6–2.8 — Derivatives

```
(38)  ∇_θ L = (∂L/∂x)(∂x/∂θ) + ∂L/∂θ
(39)  ∂x/∂θ = [∂x/∂q, ∂x/∂p, ∂x/∂s]
```

Reverse-mode AD computes the whole gradient in one backward pass via
vector-Jacobian products, without materializing Jacobians. The paper then
replaces two bottlenecks with **analytical adjoints**:

- **2.7, linear solver adjoint.** By implicit differentiation, avoid
  backpropagating through the solver internals: solve `Kᵀλ = v` (Eq. 47) and
  reuse the same factorization, since `K` is symmetric. Unlike prior work, this
  adjoint accounts for dependence on both `K` and `b`, so `q`, `p` **and** `s`
  can all be optimized.
- **2.8, URS adjoint.** Implicit differentiation at the fixed point (Eqs. 48–54)
  avoids unrolling the iteration history, sidestepping memory cost and vanishing
  gradients.

Speedups over generic AD: 1.4×–12.2× (URS adjoint), up to 13× combined.

---

## Section 3 — Four design tasks

| § | Task | Optimized | Goals |
| --- | --- | --- | --- |
| 3.1 | Shape matching | `q` | point distance to target |
| 3.2 | Pneumatic under pressure | `q` | edge length, plane, L2 reg |
| 3.3 | Load-finding (masonry dome) | `q`, `λ` loads | target heights, load variance |
| 3.4 | Support-finding (gridshell) | `q`, `δ` supports | reactions, planarity, fairness, area |

**3.1 Shape matching** — the template for workshop Exercise 3:

```
(55)  g₁(θ) = (1/n_u) Σᵢ ||xᵢ(θ) − x̂ᵢ||²
(56)  min g₁  s.t.  q_lb ≤ qⱼ ≤ q_ub
```

Bounds `q ∈ [−50, −10⁻³]` force compression-only. Self-weight `ψ = 1 kN/m²`
lumped by tributary area. Three targets (Bumps, Tristar, Cone), Hausdorff
distance ≤ 2.2%, gradient-based **three orders of magnitude** faster than
Powell / simulated annealing. Axial share of strain energy rises to 87–96%.

**3.2 Pneumatic** — half-toroid cable net, 600 edges, `ψ = 0.25 kN/m²`, three
goals combined (Eq. 58 length, Eq. 59 plane, Eq. 60 L2 regularizer on `q`).

**3.3 Load-finding** — retrofit a cracked masonry dome by optimizing loads *and*
force densities: `p_i = w_i + λᵢ nᵢ` (Eq. 61). Distance to the medial surface
drops from 39.1% to 3.8% of shell thickness. A variance goal (Eq. 63) makes the
post-tensioning forces uniform enough to actually build.

**3.4 Support-finding** — British Museum gridshell, 630 variables, four goals
including **panel planarity** (Eq. 66):

```
(66)  g₂(θ) = (1/n_f) Σ_k Σ_j ||n_k(θ)ᵀ e_j(θ)||
```

Support positions parametrized as `s_i(δᵢ) = s_0ᵢ + δᵢ nᵢ` (Eq. 71). Thrust on
the north/south supports falls from 46.1% to 6.9% of self-weight, and 96.8% of
panels come out flat.
