# 01 — Form finding from scratch

The smallest structure that still has something to teach: one free node held by
four bars, pulled down by a 1 kN load.

You write the equilibrium solver yourself, in about thirty lines of JAX, and use
it to answer one question — **why does the force density method exist?** Then you
throw your solver away and get the same answer from JAX FDM in a single call.

```bash
uv run python four_bars.py
uv run python four_bars_jaxfdm.py
```

Each script opens a 3D viewer. `four_bars.py` also charts how the residual came
down.

## What `four_bars.py` does

A node is in equilibrium when the forces pulling on it cancel. What is left over
is the **residual**, and form-finding means moving the node until the residual
vanishes.

The script writes that residual twice, from the same geometry and the same
load, changing only what is prescribed:

```python
# prescribed forces: the direction depends on where the node is
directions = edge_vectors / lengths
force_vectors = forces * directions

# prescribed force densities: no direction, no length, no division
force_vectors = q * edge_vectors
```

The first is **nonlinear**, because the bar length `l = ||x - s||` contains the
unknown. The second is **linear**, because the force density `q = f / l` has
already absorbed the length.

Both are then handed to the *same* Newton solver, which never learns which one
it is solving:

```python
r = residual(x)
K = jax.jacobian(residual)(x)
dx = jnp.linalg.solve(K, -r)
x = x + dx
```

`jax.jacobian` is the whole reason this fits on a page. That Jacobian is the
tangent stiffness matrix of the structure, and JAX differentiates the residual
to get it, so nobody has to derive it by hand.

## The punchline

```
Prescribed forces                      Force density method
Step 0: |r| = 1.959e+00                Step 0: |r| = 1.959e+00
Step 1: |r| = 2.914e-01                Step 1: |r| = 1.110e-16
Step 2: |r| = 4.316e-03
Step 3: |r| = 2.229e-06
Step 4: |r| = 5.792e-13
```

Four Newton updates against one. **The solver did not change. The
parametrization did.** Force density does not make the structure easier — it
makes the equations linear, and a linear problem is solved exactly by one Newton
step from any starting point.

## Why the two land on different shapes

```
prescribed forces      x = [0.1844, 0, 0.5951]
prescribed densities   x = [0.1520, 0, 0.5760]
```

They disagree because `q = f / l` is computed on the **starting** geometry,
before the node has moved. Once it moves, the lengths change, so `q · l` is no
longer the force you asked for. Prescribing forces fixes the forces and lets the
shape follow; prescribing densities fixes the ratios and lets both follow.

That is not a bug to fix, it is the trade the method makes.

## Tasks

1. Run `four_bars.py` and read the two traces. How many updates does each need?
2. Make the load ten times larger. The force density method still lands in one
   step. Prescribed forces runs away to `1e52`. Four bars carrying about 1 kN
   each cannot hold up 10 kN, so no equilibrium exists and Newton has nothing to
   converge to. Force densities always have a solution, because the force in a
   bar grows as the bar stretches.
3. Move the starting point `x0` to `[3.0, 3.0, 3.0]`. The force density method
   still takes one step, but it lands somewhere else entirely. Look at where `q`
   comes from and explain why. One step is guaranteed; the answer is not.
4. Set one entry of `forces` to a positive number, so one bar pulls instead of
   pushes. Newton still converges. Where does the node end up, and does the
   shape make sense?
5. In `solve_formfinding`, replace the Jacobian with the identity,
   `K = jnp.eye(3)`. That turns Newton into plain gradient descent with a step
   of one, and it diverges immediately. What was the Jacobian doing for you?

## The same problem in one call

`four_bars_jaxfdm.py` builds the same four bars as a `FDNetwork`, gives them the
same force densities, and asks for equilibrium:

```python
eq_network = fdm(network)
```

One line replaces the residual, the Jacobian and the Newton loop. It returns
`x = [0.152, 0, 0.576]` — the same point your own force density solver found,
to every digit shown.

Everything after this folder is that one call, plus goals.

## Files

| File | What it is |
| --- | --- |
| `four_bars.py` | The residual, the Newton solver, and the comparison |
| `four_bars_jaxfdm.py` | The same problem solved by `fdm()` |
| `four_bars_model.py` | The same problem through the numerical core, with no network |
| `helpers.py` | Vector norms and network building, not part of the lesson |
| `visualization.py` | Convergence chart and 3D viewer, not part of the lesson |
