"""
Two-bar form-finding problem in 2D comparing Newton's and the force density method.
"""

from math import sqrt

import jax
import jax.numpy as jnp

from plotting import plot_convergence
from plotting import plot_formfinding
from plotting import show_plots


# ------------------------------------------------------------------------------
# Input data
# ------------------------------------------------------------------------------

# Initial node position (x, y)
x0 = [0.7, 0.7]

# Support nodes (x, y)
supports = [
    [-1.0, 1.0],
    [ 1.0, 1.0],
]

# Load vector (x, y)
load = [0.0, -2.0]
# load = [0.0, 0.0]

# Edge internal forces (magnitude)
forces = [sqrt(2), sqrt(2)]

# Edge force densities (magnitude)
q = [1.0, 1.0]

# ------------------------------------------------------------------------------
# Convert to JAX arrays
# ------------------------------------------------------------------------------

x0 = jnp.array(x0)
supports = jnp.array(supports)
load = jnp.array(load)
q = jnp.array(q)
forces = jnp.array(forces)

# ------------------------------------------------------------------------------
# Nonlinear equilibrium: prescribed forces
# ------------------------------------------------------------------------------

def residual_forces(x):

    # Extract edge vectors
    edge_vectors = x - supports

    # Find edge lengths
    lengths = jnp.linalg.norm(edge_vectors, axis=1, keepdims=True)

    # Compute force directions
    directions = edge_vectors / lengths

    # Generate force vectors
    force_vectors = forces * directions

    # Obtain internal forces
    internal_forces = jnp.sum(force_vectors, axis=0)

    # Compute residual
    return internal_forces - load

# ------------------------------------------------------------------------------
# FDM equilibrium: prescribed force densities
# ------------------------------------------------------------------------------

def residual_forcedensities(x):

    # Extract edge vectors
    edge_vectors = x - supports

    # Generate force vectors
    force_vectors = q * edge_vectors

    # Obtain internal forces
    internal_forces = jnp.sum(force_vectors, axis=0)

    # Compute residual
    return internal_forces - load


# ------------------------------------------------------------------------------
# Solve the form-finding problem with Newton's method
# ------------------------------------------------------------------------------

def solve_formfinding(residual, x, num_steps):

    residual_norms = []

    # Iterate for a fixed number of steps
    for step in range(num_steps):

        # Compute residual
        r = residual(x)

        # Log progress
        residual_norm = jnp.linalg.norm(r)
        residual_norms.append(residual_norm)
        print(f"Step {step}: x = {x}, |r| = {residual_norm:.3e}")

        # Compute Jacobian (the geometric stiffness matrix!)
        K = jax.jacobian(residual)(x)

        # Shape update
        dx = jnp.linalg.solve(K, -r)

        # Update node positions
        x = x + dx

    return x, residual_norms


# ------------------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------------------

print("\nPrescribed forces")
x_force, residuals_force = solve_formfinding(residual_forces, x0, num_steps=10)

print("\nForce density method")
x_fdm, residuals_fdm = solve_formfinding(residual_forcedensities, x0, num_steps=10)


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

plot_convergence(
    [residuals_force, residuals_fdm],
    ["Prescribed forces", "Force density method"],
)

plot_formfinding(
    supports,
    [
        {"x": x0, "color": "0.7", "q": q},
        {"x": x_force, "color": "C0", "q": q},
        {"x": x_fdm, "color": "C1", "q": q},
    ],
    load=load,
)

show_plots()

