# JAX FDM Workshop — Handover Plan 

I am preparing a 5-hour hands-on workshop, **“Introduction to Differentiable Form-Finding with JAX FDM,”** for ~5 architects/structural engineers. 
The workshop teaches form-finding, not agentic coding.
Cursor is an accessibility layer that helps participants read and modify Python/JAX code.

## Pinned pedagogical arc

Do not restructure this unless a concrete technical problem requires it:

1. Visual opening
2. Optional Bletzinger lecture: Munich Olympic roof → Linkwitz/Schek → FDM history
3. **Lecture I:** lightweight structures → funicularity → FDM as forward solver → nonlinear vs linear equilibrium
4. Very short **JAX toolbox** introduction
5. **Live demo:** simple 2D two-bar equilibrium
   - prescribed forces → nonlinear residual
   - solve with Newton using `jax.jacobian`
   - introduce `q = f/l`
   - prescribed force densities → linear residual
   - same Newton machinery → equilibrium in one step
6. **Exercise 1:** simple arch with `jax_fdm.fdm()`; participants manually tune force density to match target rise
7. **Lecture II:** rich funicular design space → practical constraints → inverse problems → optimization → gradients → AD → DFDM application gallery
8. **Exercise 2:** solve the same arch target automatically using `constrained_fdm()`
9. Break
10. **Exercise 3:** 3D shape matching
11. **Exercise 4:** add panel planarity/fabrication requirements
12. **Open design challenge**
13. Five-person results + closing

## Pinned implementation decisions

- Dedicated workshop repo; JAX FDM is an installed dependency, not developed inside the workshop repo.
- Python 3.12, CPU only.
- `uv` + committed `uv.lock` for reproducibility.
- Participants expected to be Windows-heavy.
- Cursor Teams ($40/member/month) is the chosen AI-assisted IDE.
- `.cursor/rules/*.mdc` + root `AGENTS.md` provide coding/pedagogical rules.
- Agent edits should be small and inspectable through Cursor diffs.
- Agentic coding is **not** workshop content.
- Prefer plain Python and simple explicit numerical code.
- No connectivity matrices in the introductory FDM derivation.
- Avoid `None` indexing and unnecessary reshaping.
- Establish correct array shapes at creation; use `keepdims=True` where appropriate.
- Use `jax.numpy as jnp`.
- Prefer explicit loops over clever abstractions in teaching examples.
- Do not introduce advanced JAX concepts unless necessary.

## Form-finding demo

To be defined between a two-bar (new), a four-bar (example present in CMAME paper), or a full parabolic arch (cleaner for story and slides).

This is the canonical scratch mechanics example.

The prescribed-force residual uses

`edge vectors → lengths → directions → force vectors → summed residual`.

Newton uses explicitly:

    r = residual(x)
    K = jax.jacobian(residual)(x)
    dx = jnp.linalg.solve(K, -r)
    x = x + dx

With the current tested setup, prescribed-force equilibrium takes ~5 Newton updates. Replacing `f/l` with prescribed force density `q` makes the residual linear and the **same Newton algorithm converges in one update**. The two formulations are constructed to share the same equilibrium and forces there.

The purpose is to make one idea unforgettable:

**Force density linearizes the equilibrium problem by changing the parametrization; the solver itself did not change.**

Then immediately generalize verbally to networks and reveal:

    network = fdm(network)

Do NOT introduce connectivity matrices here.

## Current critical path

Infrastructure is sufficiently complete. Stop redesigning tooling or the macro schedule.

Next priorities:

1. Finalize and test form-finding scratch example.
1. Finalize and test Exercise 1: forward arch + manual target-height matching.
2. Finalize Exercise 2: same arch + `constrained_fdm`.
3. Finalize Exercise 3: 3D shape matching.
4. Extend the same 3D problem with planarity if technically practical.
5. Build the design challenge from those existing components.
6. Only afterward perform the final surgical slide assembly.
7. Full rehearsal from a clean clone, including Windows.

When helping me, preserve these decisions and focus on getting the exercises simple, robust, visually compelling, and achievable by participants with limited Python experience.