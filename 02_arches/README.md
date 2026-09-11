# 02 — The arch

Two exercises on the same structure: a compression arch spanning 10 m, loaded
with 1 kN down at every free node, that should rise 3 m at the crown.

In Exercise 1 you find the shape by hand. In Exercise 2 you state what you want
and let gradients find it for you.

Both scripts open a 3D viewer showing the flat starting network, the form-found
arch colored by force, and a magenta plane at the target rise.

## Exercise 1 — tune the force density by hand

```bash
uv run python arch.py
```

The script builds the arch flat, assigns one force density `q` to every edge,
and calls `fdm(network)` to find the shape that hangs in equilibrium. It prints
the rise it reached and how far off you are.

Only one line matters:

```python
# TUNE THIS ONE NUMBER
q = -1.0
```

**Tasks**

1. Run it. The arch is far too tall. Which direction do you need to move `q`?
2. Change `q`, run again, and repeat until the apex is within 1 cm of 3 m.
   Count how many tries it takes you.
3. Keep a small table of what you tried:

   | `q` | rise |
   | --- | --- |
   | -1.0 | 12.500 |
   | -2.0 | ? |
   | -4.0 | ? |

4. Look at your table. Multiply each `q` by its rise. What do you notice?
5. Using that pattern, **predict** the `q` for a 2 m rise before you run it.
   Then check.
6. Make `q` positive. What happens, and why does the viewer now call it tension?

**What is going on**

Multiplying rise by `|q|` always gives 12.5, so `rise = 12.5 / |q|`. That is not
a coincidence: for a uniformly loaded arch the closed-form parabola has
`rise = w·L² / (8·H)`, and here the load per length `w` is 1 kN/m, the span `L`
is 10 m, and the horizontal thrust `H` is exactly `|q|`. The discrete
force-density arch reproduces the textbook parabola, so the exact answer is
`q = -12.5 / 3 = -4.1667`.

That formula exists only because this arch is so simple. Change the load
pattern or move one support up and nothing is waiting for you — which is what
Exercise 2 is for.

## Exercise 2 — let the optimizer find it

```bash
uv run python arch_constrained.py
```

Same arch, same target. Nothing is tuned by hand. Instead the script states
three things:

```python
# what we want
my_goal = NodeZCoordinateGoal(apex_node, target_rise)

# how we measure the miss
loss = Loss(SquaredError(goals))

# what may change
parameter = EdgeGroupForceDensityParameter(edges)
```

and hands them to `constrained_fdm()`, which differentiates through the
form-finding solve and walks downhill on the gradient.

It prints twice: first the starting arch at `q = -1.0`, then what the optimizer
found. With the `GradientDescent` optimizer it reaches `q = -4.157` after 376
steps — close to the exact `-4.1667` you derived in Exercise 1, though it stops
because the loss stopped changing rather than because it arrived.

**Tasks**

1. Run it. How does the value it found compare to the one you tuned by hand?
2. Change `target_rise` to 2.0 and run again. Did you have to change anything
   else? In Exercise 1 you would have re-tuned from scratch.
3. Change the starting `q` to -15.0. Does it still get there?
4. Swap the optimizer. The script uses `GradientDescent()`; just above it,
   `LBFGSB()` sits commented out. Use `LBFGSB` instead and compare:

   | Optimizer | Steps | Force density found | Error |
   | --- | --- | --- | --- |
   | `GradientDescent` | 376 | -4.157 | +7 mm |
   | `LBFGSB` | 11 | -4.167 | ~0 |

   Both follow the same gradient. Why does one need 34 times fewer steps, and
   land on the exact answer from Exercise 1?

**Stretch**

- Ask Cursor: *"what happens if I use `NodePointGoal` instead of
  `NodeZCoordinateGoal` for the apex?"* Try it. Why can the arch now move
  sideways?
- Give the quarter-span node its own target height as a second goal. Can a
  single shared `q` satisfy both? What does the optimizer do when it cannot?
- Swap `EdgeGroupForceDensityParameter` for one `EdgeForceDensityParameter` per
  edge, so every edge may differ. Now can it satisfy both goals? This is the
  jump from one design variable to ten, and the whole point of the afternoon.

## Files

| File | What it is |
| --- | --- |
| `arch.py` | Exercise 1 — forward form-finding, manual tuning |
| `arch_constrained.py` | Exercise 2 — inverse problem via `constrained_fdm()` |
