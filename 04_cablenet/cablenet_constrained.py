"""
Prestress a square cable-net to meet target forces and target cable lengths.

Same net as cablenet.py. Now the boundary cables should carry a given force
and the interior cables should share a given length. An optimizer searches
for the force densities that do both.
"""

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeForceGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Side length of the square net
length = 10.0

# Number of grid cells per side
num_cells = 10

# Lift of two opposite corner supports
support_height = 5.0

# Force density to start with
q = 1.0

# Target force in the boundary cables
force_boundary = 20.0

# Target length of the interior cables
length_interior = 1.0

# Force densities stay positive, so no cable goes slack
q_bound_low = 0.1
q_bound_up = 100.0

# Optimizer budget
max_iterations = 1000
tol = 1e-8

# ------------------------------------------------------------------------------
# Build a square cable-net
# ------------------------------------------------------------------------------

mesh = FDMesh.from_meshgrid(length, nx=num_cells)

# ------------------------------------------------------------------------------
# Supports: the four corners, two of them lifted
# ------------------------------------------------------------------------------

corners = []
for vertex in mesh.vertices():
    if mesh.vertex_degree(vertex) == 2:
        corners.append(vertex)

mesh.vertices_supports(corners)

# Lift the pair of opposite corners with the smallest and largest x + y
low = corners[0]
high = corners[0]
for vertex in corners:
    x, y, z = mesh.vertex_coordinates(vertex)
    x_low, y_low, z_low = mesh.vertex_coordinates(low)
    x_high, y_high, z_high = mesh.vertex_coordinates(high)
    if x + y < x_low + y_low:
        low = vertex
    if x + y > x_high + y_high:
        high = vertex

mesh.vertex_attribute(low, "z", support_height)
mesh.vertex_attribute(high, "z", support_height)

# ------------------------------------------------------------------------------
# Assign force densities
# ------------------------------------------------------------------------------

for edge in mesh.edges():
    mesh.edge_forcedensity(edge, q)

# ------------------------------------------------------------------------------
# Split the edges: rim versus interior
# ------------------------------------------------------------------------------

edges_boundary = []
edges_interior = []
for edge in mesh.edges():
    if mesh.is_edge_on_boundary(edge):
        edges_boundary.append(edge)
    else:
        edges_interior.append(edge)

# ------------------------------------------------------------------------------
# Starting point in equilibrium
# ------------------------------------------------------------------------------

eq_mesh = fdm(mesh)

print(f"Target boundary force: {force_boundary}, interior length: {length_interior}")
eq_mesh.print_stats()

# ------------------------------------------------------------------------------
# State the design intent
# ------------------------------------------------------------------------------

# What we want: a taut rim, and even interior cables
goals_force = []
for edge in edges_boundary:
    goal = EdgeForceGoal(edge, force_boundary)
    goals_force.append(goal)

goals_length = []
for edge in edges_interior:
    goal = EdgeLengthGoal(edge, length_interior)
    goals_length.append(goal)

# How we measure the miss
error_force = MeanSquaredError(goals_force, name="BoundaryForce")
error_length = MeanSquaredError(goals_length, name="InteriorLength")
loss = Loss(error_force, error_length)

# ------------------------------------------------------------------------------
# Optimization parameters
# ------------------------------------------------------------------------------

# Every cable gets its own force density, floored so it stays in tension
parameters = []
for edge in mesh.edges():
    parameter = EdgeForceDensityParameter(edge, q_bound_low, q_bound_up)
    parameters.append(parameter)

# ------------------------------------------------------------------------------
# Solve the inverse problem
# ------------------------------------------------------------------------------

optimizer = LBFGSB()

# The recorder stores the parameters per iteration, to plot the loss afterwards
recorder = OptimizationRecorder(optimizer)

opt_mesh = constrained_fdm(
    mesh,
    optimizer,
    loss,
    parameters=parameters,
    maxiter=max_iterations,
    tol=tol,
    callback=recorder,
)

# ------------------------------------------------------------------------------
# How close did we get?
# ------------------------------------------------------------------------------

boundary_forces = []
for edge in edges_boundary:
    force = opt_mesh.edge_force(edge)
    boundary_forces.append(force)

interior_lengths = []
for edge in edges_interior:
    cable_length = opt_mesh.edge_length(edge)
    interior_lengths.append(cable_length)

extra_stats = {
    "Boundary force": boundary_forces,
    "Interior length": interior_lengths,
}
opt_mesh.print_stats(extra_stats)

# ------------------------------------------------------------------------------
# Plot the loss history
# ------------------------------------------------------------------------------

plotter = LossPlotter(loss, mesh, dpi=150, figsize=(8, 4))
plotter.plot(recorder.history)
plotter.show()

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Form-found net in equilibrium
viewer.add(
    eq_mesh,
    show_vertices=True,
    show_reactions=False,
    faceopacity=0.3,
    name="Equilibrium",
)

# Prestressed net
viewer.add(
    opt_mesh,
    edgecolor="force",
    show_vertices=True,
    show_reactions=False,
    faceopacity=0.5,
    name="Opt",
)

viewer.show()
