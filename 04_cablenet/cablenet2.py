"""
A square tensile cable-net on a regular mesh grid.

Two opposite corners are lifted, so the net is a saddle. Starting from
q = F / L, an optimizer searches for the force densities that put the rim
at a target force and give the interior cables a target length, without
letting any cable fall into compression.
"""

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import EdgeForceGoal
from jax_fdm.goals import EdgeLengthGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanSquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Side length of the square net
length = 10.0

# Number of grid cells per side, one cable per cell
num_cells = 10

# Target force in the interior cables
force_interior = 2.0

# Target force in the boundary cables
force_boundary = 10.0

# Lift of two opposite corner supports
support_height = 5.0

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

print(mesh)

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
# Assign force densities
# ------------------------------------------------------------------------------

# q is the target force over the length of the starting geometry
for edge in edges_boundary:
    cable_length = mesh.edge_length(edge)
    force_density = force_boundary / cable_length
    mesh.edge_forcedensity(edge, force_density)

for edge in edges_interior:
    cable_length = mesh.edge_length(edge)
    force_density = force_interior / cable_length
    mesh.edge_forcedensity(edge, force_density)

# ------------------------------------------------------------------------------
# Form-find the tensile net
# ------------------------------------------------------------------------------

eq_mesh = fdm(mesh)

# ------------------------------------------------------------------------------
# What did q = F / L actually deliver on the rim?
# ------------------------------------------------------------------------------

forces_boundary = []
for edge in edges_boundary:
    force = eq_mesh.edge_force(edge)
    forces_boundary.append(force)

lengths_interior = []
forces_interior = []
for edge in edges_interior:
    lengths_interior.append(eq_mesh.edge_length(edge))
    forces_interior.append(eq_mesh.edge_force(edge))

stats_start = {
    "Boundary force": forces_boundary,
    "Interior length": lengths_interior,
    "Interior force": forces_interior,
}
print(f"\nTarget boundary force: {force_boundary}, interior length: {length_interior}")
eq_mesh.print_stats(stats_start)

# ------------------------------------------------------------------------------
# State the design intent
# ------------------------------------------------------------------------------

# What we want: a rim at the target force, and even interior cables
goals_force = []
for edge in edges_boundary:
    goal = EdgeForceGoal(edge, force_boundary)
    goals_force.append(goal)

goals_length = []
for edge in edges_interior:
    goal = EdgeLengthGoal(edge, length_interior)
    goals_length.append(goal)

# How we measure the miss, one named term per group
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

opt_mesh = constrained_fdm(
    mesh,
    optimizer,
    loss,
    parameters=parameters,
    maxiter=max_iterations,
    tol=tol,
)

# ------------------------------------------------------------------------------
# How close did we get?
# ------------------------------------------------------------------------------

forces_boundary_opt = []
for edge in edges_boundary:
    force = opt_mesh.edge_force(edge)
    forces_boundary_opt.append(force)

lengths_interior_opt = []
forces_interior_opt = []
for edge in edges_interior:
    lengths_interior_opt.append(opt_mesh.edge_length(edge))
    forces_interior_opt.append(opt_mesh.edge_force(edge))

stats_opt = {
    "Boundary force": forces_boundary_opt,
    "Interior length": lengths_interior_opt,
    "Interior force": forces_interior_opt,
}
opt_mesh.print_stats(stats_opt)

# ------------------------------------------------------------------------------
# Visualize the mesh
# ------------------------------------------------------------------------------

viewer = Viewer()
viewer.add(opt_mesh, edgecolor="force")
viewer.show()
