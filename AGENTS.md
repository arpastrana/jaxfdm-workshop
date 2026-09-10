# JAX FDM Workshop

This repository contains guided exercises for the JAX FDM workshop.

## Purpose

Help participants understand and modify the workshop examples.
Prefer small, pedagogically transparent changes over large rewrites.
Avoid list comprehensions and prefer explicit for loops.

## Environment

- Use `uv` for all Python commands.
- Run code with `uv run ...`.
- CPU only.
- Do not change `pyproject.toml` or `uv.lock`.
- Do not add dependencies unless explicitly requested.

## Tool-specific instructions

Editor- and language-specific rules may provide additional guidance.
When working in Cursor, follow applicable rules under `.cursor/rules/`.

## JAX

- Use `jax.numpy` in differentiable numerical code.
- Preserve differentiability.
- Prefer simple array operations that participants can understand.
- Do not introduce advanced JAX abstractions unless requested.

## JAX FDM

- Use the public JAX FDM API demonstrated in the workshop examples.
- Inspect nearby examples before inventing API calls.
- Do not modify the installed JAX FDM package source unless the participant explicitly asks.

## Workshop behavior

- Explain unfamiliar code succinctly when asked.
- Make the smallest change that satisfies the participant's request.
- Preserve existing plots and visualization.
- Do not solve later exercises in advance.
- Do not create replacement files when a small edit is sufficient.
- After changes, run the relevant example when practical.
