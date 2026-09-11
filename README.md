# Introduction to Differentiable Form-Finding with JAX FDM

Workshop at IASS 2026 in Turin, Italy.

**September 13th, 2026 — Room 8i, Politecnico di Torino**

Form-finding computes the shape a structure takes in static equilibrium. Making that
computation **differentiable** turns it into a design tool: instead of only asking
"what shape do these forces give me?", we can ask "what forces give me the shape I
want?" and let gradients answer.

This workshop builds that idea from scratch, then puts it to work with
[JAX FDM](https://github.com/arpastrana/jax_fdm).

## Objectives

By the end of the workshop you will be able to:

1. **Explain how the force density method, automatic differentiation, and
   gradient-based optimization work** — and why combining them makes form-finding
   differentiable.
2. **Use JAX FDM to solve constrained form-finding tasks, assisted by agentic
   coding** — expressing design intent as goals and constraints, and working with an
   AI coding assistant to get there.

## Prerequisites

- Comfort reading and editing Python. No JAX experience required.
- Basic structural intuition (equilibrium, tension and compression).
- **A laptop.** Everyone works locally, on their own machine, on the **CPU**. There
  is no GPU requirement. Windows ARM64 machines cannot install JAX at all; a cloud
  fallback is available for them through the instructor.
- **[Cursor](https://cursor.com)** installed. We use it as the agentic coding
  assistant throughout, and objective 2 depends on it.

Please complete the installation below **before the workshop**, not on the day.

## Installation

Four things to install and one command to verify: **Cursor**, **git**, **uv**, then
the environment itself. Pick your operating system below.

### macOS

**1. Install Cursor**

Download it from [cursor.com](https://cursor.com) and drag it into `/Applications`.

**2. Install git**

macOS ships git with the Xcode Command Line Tools. Check and install if needed:

```bash
git --version
xcode-select --install
```

**3. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal so `uv` lands on your `PATH`, then confirm:

```bash
uv --version
```

**4. Get the repository**

```bash
git clone https://github.com/arpastrana/jaxfdm-workshop.git
cd jaxfdm-workshop
```

**5. Create the environment**

```bash
uv sync
```

**6. Check your setup**

```bash
uv run python check.py
```

### Windows

Use **PowerShell** for all of the commands below. Open it from the Start menu by
typing "PowerShell".

**1. Install Cursor**

Download the Windows installer from [cursor.com](https://cursor.com) and run it.

**2. Install git**

```powershell
git --version
winget install --id Git.Git -e
```

Run the `winget` line only if `git --version` reports that git is not recognized.
Close and reopen PowerShell afterwards.

**3. Install uv**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell so `uv` lands on your `PATH`, then confirm:

```powershell
uv --version
```

**4. Get the repository**

```powershell
git clone https://github.com/arpastrana/jaxfdm-workshop.git
cd jaxfdm-workshop
```

**5. Create the environment**

```powershell
uv sync
```

**6. Check your setup**

```powershell
uv run python check.py
```

### What those last two commands do

`uv sync` installs the exact versions recorded in `uv.lock` — Python 3.12, JAX, and
JAX FDM 0.14.0 among them. You do **not** need to install Python yourself: this
repository pins Python 3.12 in `.python-version` and uv downloads it for you. Use
`uv sync --locked` if you want uv to fail rather than silently re-resolve when the
lockfile is out of date.

`check.py` reports your OS and architecture, verifies every required package imports,
confirms your Python and JAX FDM versions match the pins, and runs small JIT,
automatic differentiation, and plotting tests. A green report means you are ready:

```
Result
──────

  ✓  Everything looks good!
```

If anything fails, send a screenshot of the whole report to the instructor.

### Run an example

```bash
uv run python examples/twobar.py
```

Prefix every Python command with `uv run` — that is what puts the workshop
environment on the path.

### Supported machines

| Machine | Status |
| --- | --- |
| Apple Silicon Mac (M1 and later) | Supported |
| Windows, x64 | Supported |
| Linux, x86_64 or arm64 | Supported |
| **Windows on ARM** | **Not supported — contact the instructor** |
| **Intel Mac** | **Not supported — contact the instructor** |

JAX publishes no wheels for Windows on ARM, and stopped publishing macOS x86_64
wheels after jaxlib 0.4.38, so `uv sync` cannot succeed on either machine.

If you are on one of them, please contact the instructor well before September 13th.
Windows ARM64 machines can be set up with a cloud fallback; Intel Macs need a
different machine. Either way, get in touch ahead of time rather than on the day —
`check.py` will tell you the same thing if you run it first.

## Repository structure

```bash
jaxfdm-workshop/
│
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── check.py
│
├── data/
│   └── mesh_27.json
│
├── 00_jax/
│   └── jax_crash_course.ipynb
├── 01_equilibrium/
│   └── fdm_from_scratch.py
├── 02_arch/
│   └── arch.py
├── 03_gridshell/
│   ├── gridshell.py
│   └── planarity.py
├── 04_challenge/
│   └── design_challenge.py
│
└── examples/
    ├── twobar.py
    ├── fourbar.py
    ├── arch_formfinding_newton_vs_fdm.py
    ├── truss_fourbar.py
    └── visualization.py
```

Exercise folders are added as the material is finalized, so a fresh clone may not
have all of them yet. The `examples/` folder holds worked examples you can read and
run at any point.

## Workshop outline

1. **`00_jax` — a JAX crash course.** Arrays, `jit`, `grad`, and the functional style
   JAX expects.
2. **`01_equilibrium` — equilibrium from scratch.** Derive the force density method
   by hand, and see why prescribing force densities instead of forces turns a
   nonlinear problem into a linear one.
3. **`02_arch` — the arch.** Form-finding a real structure, and differentiating
   through the solve.
4. **`03_gridshell` — constrained form-finding.** Goals, losses, and constraints on a
   gridshell with JAX FDM's optimization API.
5. **`04_challenge` — a design challenge.** An open-ended task, tackled with an
   agentic assistant.

## Working with an AI assistant

`AGENTS.md` holds the house rules for coding assistants in this repository — use the
public JAX FDM API, keep changes small and pedagogically transparent, don't solve
later exercises ahead of time. `CLAUDE.md` points Claude Code at the same file. Both
are worth skimming before you start prompting, and Cursor reads `AGENTS.md`
automatically.

## Resources

- [JAX FDM](https://github.com/arpastrana/jax_fdm) — repository and documentation
- [JAX](https://docs.jax.dev/) — official documentation
- [uv](https://docs.astral.sh/uv/) — official documentation
- [Cursor](https://docs.cursor.com/) — official documentation
