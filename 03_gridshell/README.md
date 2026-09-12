# 03 — The gridshell

A flat quad grid, pinned along its whole perimeter, form-found until it matches
a doubly curved target surface — in compression only, under its own weight.

This is the first exercise with **competing goals**. One pulls the vertices onto
the target. One flattens the quad panels so they could be cut from flat glass.
One keeps the mesh even and free of creases. They disagree, and you decide by
how much.

```bash
uv run python gridshell.py
```

The viewer shows the target as a plain translucent shape, with the form-found
gridshell inside it: nodes, edges, load arrows, and every panel painted by how
far it is from flat.

## Where the target comes from

`data/mesh_freeform.json` is a doubly curved quad mesh modeled in Rhino 8 and
exported with `rhino_mesh_exporter.ghx`: 100 vertices, 81 quad faces, spanning
about 11 × 11 m and rising 5.5 m.

It arrives as a plain COMPAS `Mesh`, not an `FDMesh`. Rhino 8 runs on Python 3.9,
which cannot host JAX, so the file carries geometry only and the structural side is
added on this end:

```python
mesh_target = Mesh.from_json(FILE_TARGET)
mesh = mesh_target.copy(FDMesh)
```

`copy(FDMesh)` keeps the vertex numbering and the face winding, and picks up the
force density attributes that `FDMesh` defines and a plain mesh does not.

## What the script says

The starting mesh is that copy, so it *begins* on the target: vertex *i* pairs with
vertex *i* by construction, and the distance to the target starts at zero.

Equilibrium is what pulls it off. The shell has to hang under its own weight in
compression, and its panels have to be flat, and neither is true of the surface you
modeled. What the optimizer looks for is the force densities whose equilibrium shape
stays as close to the target as those other demands allow.

All 36 boundary vertices are supports, 16 of them well above the ground, so the rim
stays on the target curve and only the interior is free to move.

Three goals go into the loss:

```python
error_shape      = MeanSquaredError(goals_shape)
error_planarity  = MeanPredictionError(goals_planarity,  alpha=weight_planarity)
error_smoothness = MeanPredictionError(goals_smoothness, alpha=weight_smoothness)

loss = Loss(error_shape, error_planarity, error_smoothness)
```

Two kinds of error, not one, because the goals are measured differently. The
point goal has a **target** to reach, so its miss is squared. Planarity and
smoothness are **energies**, already at zero when a panel is flat or a mesh is
fair, so they are minimized directly — squaring them would flatten the gradient
exactly where you still want the optimizer to push.

The optimizer may change 180 force densities, all bounded negative so the shell
stays in compression, plus the height of each of the 36 supports.

## The three knobs

Everything you need to experiment with sits at the top of the file.

### `weight_planarity` — flat panels versus the target shape

This is the main trade-off. Self-weight and the rim are fixed, so buying flat
panels costs shape fidelity.

| weight | mean distance | mean flatness | panels buildable |
| --- | --- | --- | --- |
| 0 | 0.038 m | 3.57 | 13 of 81 (16 %) |
| 10 | ? | ? | ? |
| 100 *(shipped)* | ? | ? | ? |
| 1000 | ? | ? | ? |

### `support_z_tolerance` — how far the rim may slide up or down

Supports never move during form-finding, but they *can* be optimized. Letting
the rim breathe vertically is the single most effective change in this file.

| tolerance | mean distance | mean flatness | panels buildable |
| --- | --- | --- | --- |
| 0 (pinned) | 0.624 m | 0.70 | 66 of 81 (81 %) |
| 0.25 m | 0.524 m | 1.13 | **52 of 81 (64 %)** |
| 0.5 m *(shipped)* | 0.484 m | 0.42 | 72 of 81 (89 %) |
| 1.0 m | 0.381 m | 0.16 | 79 of 81 (98 %) |

Note the dip. A little rim freedom is **worse** than none: at 0.25 m the
optimizer spends the new freedom improving the shape term instead, and only past
about 0.5 m is there enough room to find genuinely flat panels.

### `weight_smoothness` — an even mesh, free of creases

This one fixes something the panel count does not measure. Creasing is the angle
between neighboring panels; the target's own worst crease is 21.4°.

| rim | smoothness | worst crease | edge length spread |
| --- | --- | --- | --- |
| pinned | 0 | 38.5° | 34.9 % |
| pinned | 0.5 | 33.6° | 29.4 % |
| 0.5 m | 0 | 21.5° | 24.8 % |
| 0.5 m *(shipped)* | 0.5 | 21.5° | 24.4 % |

Read the last two rows carefully. **With the rim free, the mesh already comes out
as smooth as the target**, and the smoothness goal adds almost nothing. It earns
its keep only when the rim is pinned, where it takes the worst crease from 38.5°
down to 33.6°. Keeping a goal in the loss that is doing no work is worth
noticing — it costs gradient evaluations and it hides which knob is responsible
for what.

## Tasks

1. Run it as shipped. 72 of 81 panels are buildable. Spin the model and find the
   nine that are not — the colormap runs from flat to warped, so they are the
   bright faces. Where on the shell do they sit, and why there?
2. Fill in the `weight_planarity` table. Plot distance against buildable panels.
   There is no single best answer: you are choosing a point on a trade-off, and
   that choice is the design decision. Which one would you build?
3. Set `weight_planarity = 1000`. Nearly every panel is buildable, but look at
   the shape. Is it still the building you were asked for?
4. Reproduce the dip: set `support_z_tolerance` to 0.25 and confirm you get
   *fewer* buildable panels than at 0. Explain it to the person next to you.
5. Set `weight_smoothness = 0` and then `10`. The panel count barely moves at
   first, then gets worse. What is that goal actually for, and how would you see
   its effect?
6. Tighten `max_deviation` to `0.005`. No optimization changes — only the
   yardstick. How many panels are buildable now?

## Stretch

- Let the supports move sideways too, with `VertexSupportXParameter` and
  `VertexSupportYParameter`. With ±1 m in all three directions, every panel comes
  out buildable. Is that a fair thing to do to a building?
- Swap the smoothness goal for `EdgesLengthEqualGoal`, which equalizes panel
  sizes instead. It gets the sizes more uniform than the target itself, but it
  buys that by **creasing** the surface — worst crease climbs past 48°. Two
  regularizers, opposite side effects.
- Set `q_bound_up = 1.0` so edges may go into tension. The shell is no longer
  compression-only. What happens to the shape, and would you still build it?

## What to notice

The optimizer never leaves the space of funicular geometries. Every shape it
tries is already in equilibrium — the goals only steer *which* equilibrium you
end up with. That is why the answer is always buildable as a structure, even
when it is a poor match to the target or a nuisance to clad.

## Files

| File | What it is |
| --- | --- |
| `gridshell.py` | The exercise |
| `planarity.py` | Panel flatness, coloring and distance measures, not part of the lesson |
| `../data/mesh_freeform.json` | Target surface, 100 vertices and 81 quads |
