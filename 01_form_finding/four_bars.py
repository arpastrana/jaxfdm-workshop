"""
Four-bar form-finding problem in 3D comparing Newton's and the force density method.

One free node is connected to four supports. Edge forces are signed:
negative values mean compression.
"""

import jax
import jax.numpy as jnp

from helpers import compute_vector_norm
from helpers import compute_vectors_norm
from visualization import plot_convergence
from visualization import view_structure
from visualization import show_plots
from visualization import show_viewer


# ------------------------------------------------------------------------------
# Input data
# ------------------------------------------------------------------------------

# Initial free-node position (x, y, z)
x0 = [0.0, 0.0, 0.0]

# Support nodes (x, y, z)
supports = [
    [-1.0, 0.0, 0.0],
    [ 1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.5],
    [ 0.0, -1.0, 0.5],
]

# Load vector (x, y, z)
load = [0.0, 0.0, -1.0]

# Edge internal forces (one value per edge, signed: negative is compression)
forces = [
    [-0.5],
    [-1.0],
    [-1.0],
    [-1.0],
]


# ------------------------------------------------------------------------------
# Convert to JAX arrays
# ------------------------------------------------------------------------------

x0 = jnp.array(x0)
supports = jnp.array(supports)
load = jnp.array(load)
forces = jnp.array(forces)

# ------------------------------------------------------------------------------
# Force densities from the initial geometry: q = force / length
# ------------------------------------------------------------------------------

edge_vectors = x0 - supports
lengths = compute_vectors_norm(edge_vectors)
q = forces / lengths

# ------------------------------------------------------------------------------
# Nonlinear equilibrium: prescribed forces
# ------------------------------------------------------------------------------

def residual_forces(x):
    """
    Compute the residual at the free node using prescribed forces.
    """
    # Extract edge vectors
    edge_vectors = x - supports

    # Find edge lengths
    lengths = compute_vectors_norm(edge_vectors)

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
    """
    Compute the residual at the free node using force densities.
    """
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
    """
    Solve the equilibrium problem with Newton's method.
    """
    residual_norms = []

    # Iterate for a fixed number of steps
    for step in range(num_steps):

        # Compute residual
        r = residual(x)

        # Compute Jacobian (the geometric stiffness matrix!)
        K = jax.jacobian(residual)(x)

        # Shape update
        dx = jnp.linalg.solve(K, -r)

        # Update node positions
        x = x + dx

        # Log progress
        residual_norm = compute_vector_norm(r)
        residual_norms.append(residual_norm)
        print(f"Step {step}: x = {x}, |r| = {residual_norm:.3e}")

    return x, residual_norms


# ------------------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------------------

print("\nPrescribed forces")
x_force, residuals_force = solve_formfinding(residual_forces, x0, num_steps=10)

print("\nForce density method")
x_fdm, residuals_fdm = solve_formfinding(residual_forcedensities, x0, num_steps=10)

edge_vectors_fdm = x_fdm - supports
lengths_fdm = compute_vectors_norm(edge_vectors_fdm)
forces_fdm = q * lengths_fdm

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

plot_convergence(residuals_force, label="Prescribed forces")
plot_convergence(residuals_fdm, label="Force density method")
show_plots()

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

view_structure(
    supports,
    x0,
    load=load,
    forces=forces,
    color="0.7",
    edgewidth=0.02,
    residual=residual_forces(x0),
    name="Initial",
    )

view_structure(
    supports,
    x_force,
    load=load,
    forces=forces,
    color="C1",
    edgewidth=0.05,
    residual=residual_forces(x_force),
    name="Prescribed forces",
    )

view_structure(
    supports,
    x_fdm,
    load=load,
    forces=forces_fdm,
    color="C0",
    edgewidth=0.05,
    residual=residual_forcedensities(x_fdm),
    name="Force density method",
    )

show_viewer()
