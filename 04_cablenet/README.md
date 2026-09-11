# 04 — The cable-net

A square net of cables, 10 m a side with 10 cables each way, pinned at all four
corners with **two opposite corners lifted 5 m**. That one diagonal is what makes
it a saddle. Every cable pulls, so every force density is positive.

In Exercise 1 you form-find it and find out that `q` does something different here
than it did in `02_arches`. In Exercise 2 you state targets on **forces and
lengths** instead of on geometry.

Both scripts open a 3D viewer with the net colored by force.

## How the net is built

```python
mesh = FDMesh.from_meshgrid(length, nx=num_cells)
```

One line gives 121 vertices, 220 edges and 100 quad faces. The corners are the only
vertices with just two neighbors, which is how both scripts find them:

```python
if mesh.vertex_degree(vertex) == 2:
```

All four become supports. Then the pair with the smallest and the largest `x + y`,
one diagonal, is lifted to `support_height`. Lift all four instead and the saddle
flattens into a roof; lift the other diagonal and you get the same saddle turned
a quarter turn.

## Exercise 1 — what does `q` control?

```bash
uv run python cablenet.py
```

One positive force density on all 220 cables, then `fdm(mesh)`. Only one line
matters:

```python
# TUNE THIS NUMBER
q = 1.0
```

**Tasks**

1. Run it. Read the force range that `print_stats` reports. Why do `Forces` and
   `Lengths` print the *same* numbers? Look at what `q` is set to.
2. Change `q` to 10 and run again. What happened to the forces? What happened to
   the shape?
3. Fill in the table:

   | `q` | max force | max height |
   | --- | --- | --- |
   | 0.5 | 1.222 | 5.000 |
   | 1.0 | ? | ? |
   | 10 | ? | ? |
   | 100 | ? | ? |

4. In `02_arches`, `q` set the rise: `rise = 11.25 / |q|`. Here it does not move the
   shape at all. What is different about this structure?
5. Make `q` negative and run it. Nothing breaks, and the shape is identical. Explain
   why the solver is content and an engineer is not.

**What is going on**

There is **no applied load** on this net. With no load and a single `q` shared by
every cable, equilibrium is a homogeneous system: scale every `q` by the same factor
and both sides scale with it, so the shape cannot move. The forces, `F = q · L`,
scale exactly with `q`.

Across a 200-fold range of `q`, the vertices move by less than `1e-14` m while the
forces grow 200 times over.

So `q` is a **prestress dial** here, not a shape dial. The shape comes from the
supports. That is the mirror image of the arches, where the load was fixed and `q`
chose the geometry.

Which raises the question Exercise 2 answers: if you want a *particular* force in a
particular cable, setting `q` by hand is the wrong move.

## Exercise 2 — targets on forces and lengths

```bash
uv run python cablenet_constrained.py
```

The 40 rim cables should each carry 20 units of force, and the 180 interior cables
should each be 1 m long. Two goals, on two different quantities, on two different
groups of edges, and nothing geometric anywhere.

```python
goal = EdgeForceGoal(edge, force_boundary)
goal = EdgeLengthGoal(edge, length_interior)

error_force = MeanSquaredError(goals_force, name="BoundaryForce")
error_length = MeanSquaredError(goals_length, name="InteriorLength")
loss = Loss(error_force, error_length)
```

Naming the error terms pays off: `print_stats` and the loss plot break the history
down per term, so you can see which goal is still costing you.

### Why not just set `q = F / L`?

It is tempting to compute `q = 20 / L` on each rim cable and be done. Try it, and
the rim forces land between **15.2 and 18.3**, averaging 16.6 against a target of
20.

`q = F / L` uses the length *before* form-finding. The net then moves, `L` changes,
and `F = q · L` ends up somewhere else. Prescribing `q` fixes `q`; it does not fix
`F`. To get `F` you have to search for the `q` that delivers it, which is what
`constrained_fdm` does.

### The shipped script is the end of a conversation

Each piece of it is repairing something the previous version got wrong.

| setup | iterations | rim force | interior length | compression |
| --- | --- | --- | --- | --- |
| rim force goal only | 17 | 20.000 | 0.441 – 1.351 | none |
| both goals, `q` unbounded | 123 | 19.96 – 20.08 | 0.619 – 1.218 | yes, down to −1.22 |
| both goals, `q` bounded *(shipped)* | 953 | 19.996 – 20.005 | 0.964 – 1.047 | none |

**Row 1.** The rim goal alone is satisfied exactly, in 17 iterations. Forty goals
steering 220 force densities is wildly underdetermined, so the optimizer has room to
spare and spends it however it likes. The interior comes out ragged, cables running
from 0.44 m to 1.35 m. Same lesson as `02_arches`: freedom is not worth anything
until you say how to spend it.

**Row 2.** The interior length goal fixes the raggedness, and some force densities
go **negative**. A negative `q` in a cable-net is a cable pushing.

**Row 3.** Bound the parameters so it cannot happen:

```python
parameter = EdgeForceDensityParameter(edge, q_bound_low, q_bound_up)
```

Check `FDs` in the output: the minimum sits at exactly `0.1`, the value of
`q_bound_low`. The bound is **active** — the optimizer wanted to go lower and was
not allowed. Note the cost, too: 953 iterations against 17.

**Tasks**

1. Run it. Read the `Error breakdown` at the bottom. Which term is larger at the
   end, and by how much?
2. Set `length_interior = 1.5`. It converges, but the interior lands at 0.35 – 2.07
   m. The net is 10 m across with 10 cells, so what length does the grid *want*?
3. Raise `q_bound_low` to 5.0. The interior collapses to about 0.47 m. Why does a
   higher floor on `q` shrink the cables?
4. Drop the interior goal (`loss = Loss(error_force)`) and check whether any cable
   goes slack. Then drop the bounds as well.
5. Swap `MeanSquaredError` for `SquaredError` on both terms. The rim has 40 goals
   and the interior has 180 — which goal does the unaveraged loss now favor?

## Build it yourself

This exercise was written by conversation with an AI assistant, one request at a
time. Reproduce it in an empty file, prompting Cursor for each step and reading the
diff before you accept it:

1. *"Create a tensile cable-net from a square mesh grid, 10 m per side, 10 cables
   per side. Follow this repo's conventions."*
2. *"Add a viewer so I can see the mesh in 3D, with default settings."*
3. *"Make the four corners supports, and lift two opposite corners by 5 m."*
4. *"Set each force density as a target force over that edge's current length: 2 for
   the interior cables, 10 for the boundary ones."*
5. *"Form-find with `fdm` and show that result instead of the input mesh."*
6. *"Report the forces on the boundary cables."*
7. *"Drive the boundary forces to 10 with a loss, using mean squared error and
   LBFGSB, 1000 iterations, tolerance 1e-8."*
8. *"Give me statistics on the length and force of the interior edges."*
9. *"The interior edges are ragged. Add a goal giving them a target length of 1."*
10. *"Some edges are in compression. Keep them tension-only with box constraints on
    the force density parameters."*

Steps 6, 8 and 9 are the ones that matter, and none of them is a coding request.
Each is you reading the numbers, noticing something wrong, and asking for a fix. The
assistant will not volunteer any of the three.

## Stretch

- Lift all four corners instead of two. Is there still a saddle, and is the net
  still in tension everywhere?
- Add a downward load at every free vertex. Does `q` start to control the shape?
- Give the rim one shared `q` with `EdgeGroupForceDensityParameter` instead of 220
  independent ones. Can the rim force goal still be met?

## Files

| File | What it is |
| --- | --- |
| `cablenet.py` | Exercise 1 — forward form-finding in tension |
| `cablenet_constrained.py` | Exercise 2 — goals on forces and lengths |
