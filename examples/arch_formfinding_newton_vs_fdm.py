"""
Planar arch form-finding problem comparing Newton's method with the force density method.

The horizontal coordinates of the arch nodes are fixed. The unknowns are the vertical
coordinates of the three free nodes.

The two formulations are constructed to have the same equilibrium geometry and the
same internal forces at equilibrium:

1. Prescribed internal forces -> nonlinear equilibrium -> several Newton iterations.
2. Prescribed force densities -> linear equilibrium -> one Newton iteration.

Note:
With uniform vertical nodal loads and fixed horizontal spacing, this discrete funicular
arch is parabolic rather than an exact self-weight catenary.
"""

from math import sqrt

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Input data
# ------------------------------------------------------------------------------

# Horizontal node coordinates
x_coordinates = [0.0, 1.0, 2.0, 3.0, 4.0]

# Initial vertical coordinates of free nodes
y0 = [0.1, 0.1, 0.1]

# Fixed support heights
y_supports = [0.0, 0.0]

# Vertical loads at free nodes
loads = [
    [0.0, -0.5],
    [0.0, -0.5],
    [0.0, -0.5],
]

# Edge force densities.
# Negative values produce a compression-only arch with the sign convention used here.
q = [
    [-1.0],
    [-1.0],
    [-1.0],
    [-1.0],
]

# Prescribed edge forces chosen so that the nonlinear problem has the same
# equilibrium solution as the force density problem.
forces = [
    [-1.25],
    [-sqrt(17.0) / 4.0],
    [-sqrt(17.0) / 4.0],
    [-1.25],
]


# ------------------------------------------------------------------------------
# Convert to JAX arrays
# ------------------------------------------------------------------------------

x_coordinates = jnp.array(x_coordinates)

y0 = jnp.array(y0)

y_supports = jnp.array(y_supports)

loads = jnp.array(loads)

q = jnp.array(q)

forces = jnp.array(forces)


# ------------------------------------------------------------------------------
# Assemble full node coordinates
# ------------------------------------------------------------------------------

def coordinates(y):
    """Return the coordinates of all nodes from the free-node heights."""

    y_coordinates = jnp.concatenate(
        (
            y_supports[0:1],
            y,
            y_supports[1:2],
        )
    )

    return jnp.stack(
        (
            x_coordinates,
            y_coordinates,
        ),
        axis=1,
    )


# ------------------------------------------------------------------------------
# Nonlinear equilibrium: prescribed forces
# ------------------------------------------------------------------------------

def residual_forces(y):
    """Return the vertical residual forces at the free nodes."""

    xyz = coordinates(y)

    # Extract edge vectors
    edge_vectors = xyz[1:] - xyz[:-1]

    # Find edge lengths
    lengths = jnp.linalg.norm(
        edge_vectors,
        axis=1,
        keepdims=True,
    )

    # Compute force directions
    directions = edge_vectors / lengths

    # Generate force vectors
    force_vectors = forces * directions

    # Sum the forces from adjacent edges at each free node
    internal_forces = force_vectors[1:] - force_vectors[:-1]

    # Compute vertical residuals
    return internal_forces[:, 1] + loads[:, 1]


# ------------------------------------------------------------------------------
# FDM equilibrium: prescribed force densities
# ------------------------------------------------------------------------------

def residual_forcedensities(y):
    """Return the vertical residual forces at the free nodes."""

    xyz = coordinates(y)

    # Extract edge vectors
    edge_vectors = xyz[1:] - xyz[:-1]

    # Generate force vectors
    force_vectors = q * edge_vectors

    # Sum the forces from adjacent edges at each free node
    internal_forces = force_vectors[1:] - force_vectors[:-1]

    # Compute vertical residuals
    return internal_forces[:, 1] + loads[:, 1]


# ------------------------------------------------------------------------------
# Solve the form-finding problem with Newton's method
# ------------------------------------------------------------------------------

def solve_formfinding(residual, y, num_steps=10, tolerance=1e-6):
    """Solve equilibrium with Newton's method."""

    residual_norms = []
    coordinates_history = [coordinates(y)]

    for step in range(num_steps):

        # Compute residual
        r = residual(y)

        # Log progress
        residual_norm = jnp.linalg.norm(r)
        residual_norms.append(residual_norm)

        print(
            f"Step {step}: "
            f"y = {y}, "
            f"|r| = {residual_norm:.3e}"
        )

        # Stop at equilibrium
        if residual_norm < tolerance:
            break

        # Compute Jacobian (tangent stiffness matrix)
        K = jax.jacobian(residual)(y)

        # Shape update
        dy = jnp.linalg.solve(K, -r)

        # Update node positions
        y = y + dy

        coordinates_history.append(coordinates(y))

    return y, residual_norms, coordinates_history


# ------------------------------------------------------------------------------
# Compare nonlinear equilibrium and FDM
# ------------------------------------------------------------------------------

print("\nPrescribed forces: nonlinear equilibrium")
y_newton, residuals_newton, history_newton = solve_formfinding(
    residual_forces,
    y0,
)

print("\nPrescribed force densities: FDM")
y_fdm, residuals_fdm, history_fdm = solve_formfinding(
    residual_forcedensities,
    y0,
)


# ------------------------------------------------------------------------------
# Verify that both formulations converge to the same arch
# ------------------------------------------------------------------------------

xyz_newton = coordinates(y_newton)
xyz_fdm = coordinates(y_fdm)

print("\nNewton equilibrium:")
print(xyz_newton)

print("\nFDM equilibrium:")
print(xyz_fdm)

print(
    "\nMaximum coordinate difference:",
    jnp.max(jnp.abs(xyz_newton - xyz_fdm)),
)


# ------------------------------------------------------------------------------
# Plot form-finding
# ------------------------------------------------------------------------------

xyz_initial = coordinates(y0)

plt.figure()

plt.plot(
    xyz_initial[:, 0],
    xyz_initial[:, 1],
    "o--",
    label="Initial",
)

plt.plot(
    xyz_newton[:, 0],
    xyz_newton[:, 1],
    "o-",
    label="Newton",
)

plt.plot(
    xyz_fdm[:, 0],
    xyz_fdm[:, 1],
    "o-",
    label="FDM",
)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------------------------
# Plot convergence
# ------------------------------------------------------------------------------

plt.figure()

plt.semilogy(
    range(len(residuals_newton)),
    residuals_newton,
    "o-",
    label="Newton",
)

plt.semilogy(
    range(len(residuals_fdm)),
    residuals_fdm,
    "o-",
    label="FDM",
)

plt.xlabel("Newton step")
plt.ylabel("Residual norm")
plt.legend()
plt.tight_layout()

plt.show()
