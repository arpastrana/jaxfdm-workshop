# 02 — The arches

Two exercises on the same structure: a pair of compression arches spanning 10 m
that cross at a shared crown, loaded with 1 kN down at every free node, and that
should rise 3 m where they meet.

In Exercise 1 you find the shape by hand. In Exercise 2 you state what you want
and let gradients find it for you.

Both scripts open a 3D viewer with the form-found cross colored by force and a
magenta plane at the target rise.

## How the structure is built

Both scripts build the cross the same way, in three lines:

```python
arch_x = create_arch_network(span, num_segments)

rotation = Rotation.from_axis_and_angle([0.0, 0.0, 1.0], radians(90.0))
arch_y = arch_x.transformed(rotation)

network, shared_nodes = fuse_networks(arch_x, arch_y)
```

One arch, the same arch turned a quarter turn, and a fuse that welds the node
they have in common. `fuse_networks` hands back the keys of the shared nodes,
which is how the scripts know the crown without hunting for it. Keep
`num_segments` even, or the two arches have no node at the center, nothing
welds, and `shared_nodes` comes back empty.

## Exercise 1 — tune the force density by hand

```bash
uv run python arches.py
```

The script assigns one force density `q` to every edge of both arches and calls
`fdm(network)` to find the shape that hangs in equilibrium. It prints the rise
it reached and how far off you are.

Only one line matters:

```python
# TUNE THIS NUMBER
q = -2.5
```

**Tasks**

1. Run it. Which direction do you need to move `q`?
2. Change `q`, run again, and repeat until the crown is within 1 cm of 3 m.
   Count how many tries it takes you.
3. Keep a small table of what you tried:

   | `q` | rise |
   | --- | --- |
   | -1.0 | 11.250 |
   | -2.0 | ? |
   | -3.0 | ? |

4. Look at your table. Multiply each `q` by its rise. What do you notice?
5. Using that pattern, **predict** the `q` for a 2 m rise before you run it.
   Then check.
6. Make `q` positive. What happens, and why does the viewer now call it tension?

**What is going on**

Multiplying rise by `|q|` always gives 11.25, so `rise = 11.25 / |q|` and the
exact answer for a 3 m rise is `q = -11.25 / 3 = -3.75`.

A single arch on its own, same span and load, gives `rise = 12.5 / |q|`. Crossing
the arches lowers the crown by about a tenth for the same force density, because
four bars now hold it up instead of two. Try it: comment out the rotation and
the fuse, form-find `arch_x` alone, and check the constant.

That kind of tidy relationship exists only because this structure is so simple.
Change the load pattern or move one support up and nothing is waiting for you —
which is what Exercise 2 is for.

## Exercise 2 — let the optimizer find it

```bash
uv run python arches_constrained.py
```

Same structure, same target. Nothing is tuned by hand. Instead the script states
three things:

```python
# what we want
goals.append(NodeZCoordinateGoal(crown_node, target_rise))

# how we measure the miss
loss = Loss(SquaredError(goals))

# what may change
parameter = EdgeForceDensityParameter(edge)
```

and hands them to `constrained_fdm()`, which differentiates through the
form-finding solve and walks downhill on the gradient. An `OptimizationRecorder`
logs every iteration, and a `LossPlotter` charts the loss afterwards, so you can
watch how the search actually behaved rather than just reading the answer.

It prints the rise twice: the starting structure first, then what the optimizer
found.

**Tasks**

1. Run it. Look at the loss plot. Where does most of the progress happen?
2. How does the answer compare to the `q` you tuned by hand in Exercise 1?
   Check the `FDs` line that `print_stats()` prints.
3. Change `target_rise` to 2.0 and run again. Did you have to change anything
   else? In Exercise 1 you would have re-tuned from scratch.
4. Change the starting `q` to -15.0. Does it still get there, and does the loss
   plot look different?
5. Drop `learning_rate` back to `0.01`. The optimizer now runs out of iterations
   before it arrives. Gradient descent takes a fixed-size step, so the step size
   is yours to choose and it matters.

**One goal, twenty unknowns**

The script gives every edge its own force density: twenty unknowns steering a
single goal. Swap the parameter block for one shared force density and compare:

```python
from jax_fdm.parameters import EdgeGroupForceDensityParameter

parameters = [EdgeGroupForceDensityParameter(list(network.edges()))]
```

| parameters | optimizer | steps | force densities found |
| --- | --- | --- | --- |
| one shared `q` | `GradientDescent` | 129 | -3.750 everywhere |
| per-edge `q` | `GradientDescent` | 1826 | -4.224 … -2.286 |
| one shared `q` | `LBFGSB` | 8 | -3.750 everywhere |
| per-edge `q` | `LBFGSB` | 10 | -4.079 … -2.365 |

Both parametrizations hit the target. But the shared one lands on -3.750, the
exact value you derived in Exercise 1, and gets there the same way every time.
The per-edge version has an infinite family of answers to choose from — one goal
cannot pin down twenty numbers — so it returns whichever member the optimizer
happened to wander into, and a different optimizer returns a different one.

More freedom is not automatically better. It is only better once you have
something to say about how it should be spent.

## Stretch

- Ask Cursor: *"what happens if I use `NodePointGoal` instead of
  `NodeZCoordinateGoal` for the crown?"* Try it. Why can the cross now move
  sideways?
- Give a quarter-span node its own target height as a second goal. Can a single
  shared `q` satisfy both? Can twenty?
- Bound the force densities with `EdgeForceDensityParameter(edge, -50.0, -1e-3)`
  so no edge can flip into tension. Does the answer change?

## Files

| File | What it is |
| --- | --- |
| `arches.py` | Exercise 1 — forward form-finding, manual tuning |
| `arches_constrained.py` | Exercise 2 — inverse problem via `constrained_fdm()` |
| `helpers.py` | Building and fusing arch networks, not part of the lesson |
