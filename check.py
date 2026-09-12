"""
JAX FDM Workshop — Environment Check

Run with:

    uv run python check.py
"""

from __future__ import annotations

import importlib
import os
import platform
import struct
import sys
import tomllib
from pathlib import Path

# Windows shenanigans
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

CHECK = "✓"
CROSS = "✗"
WARN = "⚠"
SKIP = "○"
ARROW = "→"

results: list[tuple[str, str, str]] = []
versions: dict[str, str] = {}


def heading(text: str) -> None:
    print(f"\n{text}")
    print("─" * len(text))


def record(status: str, name: str, message: str = "") -> None:
    results.append((status, name, message))

    symbol = {
        "pass": CHECK,
        "fail": CROSS,
        "warn": WARN,
        "skip": SKIP,
    }[status]

    text = f"  {symbol}  {name}"
    if message:
        text += f" — {message}"

    print(text)


def passed(name: str, message: str = "") -> None:
    record("pass", name, message)


def failed(name: str, message: str = "") -> None:
    record("fail", name, message)


def warned(name: str, message: str = "") -> None:
    record("warn", name, message)


def skipped(name: str, message: str = "") -> None:
    record("skip", name, message)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print()
print("╭──────────────────────────────────────────╮")
print("│   JAX FDM Workshop — Environment Check   │")
print("╰──────────────────────────────────────────╯")


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

heading("System")

system = platform.system()
machine = platform.machine()
processor = platform.processor()
python_version = platform.python_version()
python_bits = struct.calcsize("P") * 8

print(f"  OS            {system}")
print(f"  Architecture  {machine}")
print(f"  Processor     {processor or 'Not reported'}")
print(f"  Python        {python_version} ({python_bits}-bit)")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

heading("Architecture")

machine_lower = machine.lower()

is_x64 = machine_lower in {"amd64", "x86_64", "x64"}
is_arm64 = machine_lower in {"arm64", "aarch64"}

if python_bits == 64:
    passed("64-bit Python")
else:
    failed("64-bit Python", "32-bit Python detected")

if system == "Windows":
    if is_x64:
        passed("Windows x64", "compatible with native JAX CPU")

    elif is_arm64:
        warned(
            "Windows ARM64",
            "JAX publishes no Windows ARM64 wheels",
        )
        print(f"     {ARROW} Please contact the workshop instructor.")

    else:
        warned(
            "Windows architecture",
            f"unrecognized architecture: {machine}",
        )

elif system == "Darwin":
    if is_arm64:
        passed("Apple Silicon", "compatible with JAX")

    elif is_x64:
        warned(
            "Intel Mac",
            "JAX no longer publishes macOS x86_64 wheels",
        )
        print(f"     {ARROW} Please contact the workshop instructor.")

    else:
        warned(
            "macOS architecture",
            f"unrecognized architecture: {machine}",
        )

elif system == "Linux":
    passed("Linux", machine)

else:
    warned("Operating system", f"unrecognized system: {system}")


# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------

heading("Packages")


def package_available(module_name: str, display_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)

        if version:
            versions[display_name] = version
            passed(display_name, f"version {version}")
        else:
            passed(display_name)

        return True

    except Exception as exc:
        failed(
            display_name,
            f"{type(exc).__name__}: {exc}",
        )
        return False


jax_available = package_available("jax", "JAX")
numpy_available = package_available("numpy", "NumPy")
scipy_available = package_available("scipy", "SciPy")
matplotlib_available = package_available("matplotlib", "Matplotlib")

# Add/remove these depending on the final workshop environment.
compas_available = package_available("compas", "COMPAS")
jax_fdm_available = package_available("jax_fdm", "JAX FDM")


# ---------------------------------------------------------------------------
# Pinned versions
# ---------------------------------------------------------------------------

heading("Versions")


def version_series(version: str) -> str:
    """
    Return the major.minor part of a version string.
    """
    parts = version.split(".")

    return ".".join(parts[:2])


def python_pinned() -> str | None:
    """
    Return the Python version pinned in .python-version, if any.
    """
    version_file = ROOT_DIR / ".python-version"

    if not version_file.exists():
        return None

    return version_file.read_text().strip()


def jaxfdm_pinned() -> str | None:
    """
    Return the JAX FDM version pinned in pyproject.toml, if any.
    """
    project_file = ROOT_DIR / "pyproject.toml"

    if not project_file.exists():
        return None

    config = tomllib.loads(project_file.read_text())
    project = config.get("project", {})

    for dependency in project.get("dependencies", []):
        if dependency.startswith("jax-fdm") and "==" in dependency:
            return dependency.split("==")[1].strip()

    return None


# Local uv pin is 3.13; Colab may still be 3.12
SUPPORTED_PYTHON = ("3.12", "3.13")
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

python_expected = python_pinned()
python_found = version_series(python_version)

if python_expected is None:
    skipped("Pinned Python", ".python-version not found")

elif python_found == version_series(python_expected):
    passed("Pinned Python", f"{python_version} matches {python_expected}")

elif python_found in SUPPORTED_PYTHON:
    passed(
        "Supported Python",
        f"{python_version} is supported (local pin is {python_expected})",
    )

else:
    failed(
        "Pinned Python",
        f"expected {python_expected} or 3.12, found {python_version}",
    )


jaxfdm_expected = jaxfdm_pinned()
jaxfdm_found = versions.get("JAX FDM")

if jaxfdm_found is None:
    skipped("Pinned JAX FDM", "JAX FDM is unavailable")

elif jaxfdm_expected is None:
    skipped("Pinned JAX FDM", "no pin found in pyproject.toml")

elif jaxfdm_found == jaxfdm_expected:
    passed("Pinned JAX FDM", f"{jaxfdm_found} matches the pin")

else:
    failed(
        "Pinned JAX FDM",
        f"expected {jaxfdm_expected}, found {jaxfdm_found}",
    )


# ---------------------------------------------------------------------------
# JAX devices
# ---------------------------------------------------------------------------

heading("JAX Devices")

jax = None
jnp = None

if jax_available:
    try:
        import jax
        import jax.numpy as jnp

        devices = jax.devices()

        if devices:
            for device in devices:
                passed("JAX device", str(device))
        else:
            failed("JAX devices", "no devices reported")

    except Exception as exc:
        failed(
            "JAX devices",
            f"{type(exc).__name__}: {exc}",
        )

else:
    skipped("JAX devices", "JAX is unavailable")


# ---------------------------------------------------------------------------
# Basic JAX computation
# ---------------------------------------------------------------------------

heading("Compute")

if jax_available and jnp is not None:
    try:
        x = jnp.array([1.0, 2.0, 3.0])
        result = float(jnp.sum(x**2))

        if abs(result - 14.0) < 1e-6:
            passed("Array computation")
        else:
            failed(
                "Array computation",
                f"expected 14.0, got {result}",
            )

    except Exception as exc:
        failed(
            "Array computation",
            f"{type(exc).__name__}: {exc}",
        )
else:
    skipped("Array computation", "JAX is unavailable")


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------

if jax_available and jax is not None and jnp is not None:
    try:

        @jax.jit
        def f(x):
            return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

        result = float(f(1.234))

        if abs(result - 1.0) < 1e-5:
            passed("JIT compilation")
        else:
            failed(
                "JIT compilation",
                f"unexpected result: {result}",
            )

    except Exception as exc:
        failed(
            "JIT compilation",
            f"{type(exc).__name__}: {exc}",
        )
else:
    skipped("JIT compilation", "JAX is unavailable")


# ---------------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------------

if jax_available and jax is not None and jnp is not None:
    try:

        def energy(x):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 2.0, 3.0])
        gradient = jax.grad(energy)(x)
        expected = jnp.array([2.0, 4.0, 6.0])

        if bool(jnp.allclose(gradient, expected)):
            passed("Automatic differentiation")
        else:
            failed(
                "Automatic differentiation",
                f"unexpected gradient: {gradient}",
            )

    except Exception as exc:
        failed(
            "Automatic differentiation",
            f"{type(exc).__name__}: {exc}",
        )
else:
    skipped("Automatic differentiation", "JAX is unavailable")


# ---------------------------------------------------------------------------
# Matplotlib smoke test
# ---------------------------------------------------------------------------

heading("Visualization")

if matplotlib_available:
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plt.close(fig)

        passed(
            "Matplotlib figure creation",
            f"backend: {matplotlib.get_backend()}",
        )

    except Exception as exc:
        failed(
            "Matplotlib figure creation",
            f"{type(exc).__name__}: {exc}",
        )
else:
    skipped(
        "Matplotlib figure creation",
        "Matplotlib is unavailable",
    )


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

heading("Result")

failures = [r for r in results if r[0] == "fail"]
warnings = [r for r in results if r[0] == "warn"]
skips = [r for r in results if r[0] == "skip"]
passes = [r for r in results if r[0] == "pass"]

if not failures:
    print()
    print(f"  {CHECK}  Everything looks good!")
    print()
    if IN_COLAB:
        print("     This Colab runtime is ready for the JAX FDM workshop.")
    else:
        print("     Your computer is ready for the JAX FDM workshop.")
        print("     uv, JAX and JAX FDM all run locally.")

    if warnings:
        print()
        print(f"     {WARN}  {len(warnings)} warning(s) reported.")

else:
    print()
    print(f"  {WARN}  Your setup needs attention.")
    print()
    print(
        f"     {len(failures)} failed   "
        f"{len(warnings)} warning(s)   "
        f"{len(skips)} skipped   "
        f"{len(passes)} passed"
    )

    print()
    print("  Failed checks")
    print("  ─────────────")

    for _, name, message in failures:
        if message:
            print(f"  {CROSS}  {name}")
            print(f"     {message}")
        else:
            print(f"  {CROSS}  {name}")

    if system == "Windows" and is_arm64:
        print()
        print(f"     {ARROW} JAX publishes no Windows ARM64 wheels.")
        print(f"     {ARROW} Please contact the workshop instructor.")

    elif system == "Darwin" and is_x64:
        print()
        print(f"     {ARROW} JAX publishes no macOS x86_64 wheels.")
        print(f"     {ARROW} Please contact the workshop instructor.")

    else:
        print()
        print(f"     {ARROW} First try: uv sync")
        print(f"     {ARROW} Then rerun: uv run python check.py")

    print()
    print(
        f"     {ARROW} If problems remain, send a screenshot "
        "of this entire report to the workshop instructor."
    )

print()

if failures:
    sys.exit(1)
