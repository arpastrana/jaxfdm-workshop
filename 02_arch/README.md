# 02 — The arch

Two exercises on the same structure: a compression arch spanning 10 m, loaded
with 1 kN down at every free node, that should rise 3 m at the crown.

In Exercise 1 you find the shape by hand. In Exercise 2 you state what you want
and let gradients find it for you.

## Exercise 1 — tune the force density by hand

```bash
uv run python arch_forward.py
```

The script builds the arch flat, assigns one force density `q` to every edge,
and calls `fdm(network)` to find the shape that hangs in equilibrium. It prints
the crown rise and how far off the target you are.

Only one line matters:

```python
# TUNE THIS ONE NUMBER
q_uniform = -1.0
```

**Tasks**

1. Run it. The arch is far too tall. Which direction do you need to move `q`?
2. Change `q_uniform`, run again, and repeat until the crown is within 1 cm of
   3 m. Count how many tries it takes you.
3. Keep a small table of what you tried:

   | `q` | crown rise |
   | --- | --- |
   | -1.0 | 12.500 |
   | -2.0 | ? |
   | -4.0 | ? |

4. Look at your table. Multiply each `q` by its rise. What do you notice?
5. Using that pattern, **predict** the `q` for a 2 m rise before you run it.
   Then check.

**What is going on**

Multiplying rise by `|q|` always gives 12.5, so `rise = 12.5 / |q|`. That is not
a coincidence: for a uniformly loaded arch the closed-form parabola has
`rise = w·L² / (8·H)`, and here the load per length `w` is 1 kN/m, the span `L`
is 10 m, and the horizontal thrust `H` is exactly `|q|`. The discrete
force-density arch reproduces the textbook parabola.

That relationship exists because the arch is this simple. Change the load
pattern or the support heights and no such formula is waiting for you — which is
what Exercise 2 is for.

## Exercise 2 — let the optimizer find it

```bash
uv run python arch_optimized.py
```

Same arch, same target. Nothing is tuned by hand. Instead the script states
three things:

```python
# what we want
goals.append(NodeZCoordinateGoal(crown, target_rise))

# how we measure the miss
loss = Loss(SquaredError(goals))

# what may change
shared_q = EdgeGroupForceDensityParameter(edge_keys, bound_low=-20.0, bound_up=-0.1)
```

and hands them to `constrained_fdm()`, which differentiates through the
form-finding solve and follows the gradient downhill.

It finds `q = -4.1667` in 6 iterations. Compare that to how many tries you
needed in Exercise 1.

**Tasks**

1. Run it. Does the value it finds match the one you tuned by hand?
2. Change `target_rise` to 2.0 and run again. Did you need to change anything
   else? (In Exercise 1 you would have re-tuned by hand.)
3. Change `q_start` to -15.0. Does the optimizer still get there?
4. Set `q_bound_up` to `-5.0`, which forbids the answer. What does it do
   instead, and does that make structural sense?
5. Ask Cursor: *"what happens if I use `NodePointGoal` instead of
   `NodeZCoordinateGoal` for the crown?"* Try it. Why does the arch move
   sideways now?

**Stretch**

- Give the quarter-span node its own target height as a second goal. Can one
  shared `q` satisfy both? What does the optimizer do when it cannot?
- Swap `EdgeGroupForceDensityParameter` for one `EdgeForceDensityParameter` per
  edge, so every edge may differ. Now can it satisfy both goals? This is the
  jump from one design variable to ten, and the whole point of the afternoon.

## Files

| File | What it is |
| --- | --- |
| `arch_forward.py` | Exercise 1 — forward form-finding, manual tuning |
| `arch_optimized.py` | Exercise 2 — inverse problem via `constrained_fdm()` |
| `arch_plots.py` | Shared plotting helpers, not part of the lesson |
