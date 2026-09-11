"""
Exercise 3: match a target shape with a compression-only gridshell.

The target mesh is the starting structure. Form-finding then looks for a
compression shape that stays close to it.
Three goals compete: one pulls the vertices onto the target, one flattens the
quad panels so they could be built from flat glass, and one keeps the mesh even
and free of creases. The supports may also slide up and down to help the panels
lie flat.
"""

from pathlib import Path

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.goals import MeshPlanarityGoal
from jax_fdm.goals import MeshSmoothGoal
from jax_fdm.goals import VertexPointGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import MeanPredictionError
from jax_fdm.losses import MeanSquaredError
from jax_fdm.optimization import LBFGSB
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import VertexSupportZParameter
from jax_fdm.visualization import Viewer

from planarity import distance_to_target
from planarity import face_colors
from planarity import face_flatness
from planarity import panels_buildable


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Self-weight per unit area of the target surface
load_area = -1.0

# Force density every edge starts with
q_start = -1.5

# Force densities stay negative, so every edge is in compression
q_bound_low = -100.0
q_bound_up = -1e-3

# TUNE THESE NUMBERS
# How much flat panels matter compared to hitting the target shape
weight_planarity = 100.0

# How much an even, crease-free mesh matters
weight_smoothness = 0.5

# How far the supports may slide up or down to help the panels lie flat
support_z_tolerance = 0.5

# A panel is buildable from flat glass if it warps less than this fraction of its size
max_deviation = 0.01  # 1%

# Optimizer budget
max_iterations = 1000
tol = 1e-6

# ------------------------------------------------------------------------------
# Build filepath
# ------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
FILE_TARGET = ROOT_DIR / "data" / "mesh_freeform.json"

# ------------------------------------------------------------------------------
# Load the target shape
# ------------------------------------------------------------------------------

mesh_target = FDMesh.from_json(FILE_TARGET)
print(mesh_target)

# ------------------------------------------------------------------------------
# Starting mesh: same connectivity and vertex numbering as the target
# ------------------------------------------------------------------------------

mesh = mesh_target.copy()

# ------------------------------------------------------------------------------
# Define the structural system
# ------------------------------------------------------------------------------

# Set the initial force densities
mesh.edges_forcedensities(q=q_start)

# Support vertices on the boundary
mesh.vertices_supports(mesh.vertices_on_boundary())

# Snap the support vertices onto the target
for vertex in mesh.vertices_on_boundary():
    xyz = mesh_target.vertex_coordinates(vertex)
    mesh.vertex_attributes(vertex, "xyz", xyz)

# Self-weight lumped at each vertex via tributary area
for vertex in mesh.vertices():
    tributary_area = mesh_target.vertex_area(vertex)
    load = [0.0, 0.0, tributary_area * load_area]
    mesh.vertex_load(vertex, load)

# ------------------------------------------------------------------------------
# Goal 1: Approximate the target shape to follow design intent
# ------------------------------------------------------------------------------

goals_shape = []
for vertex in mesh.vertices_free():
    xyz = mesh_target.vertex_coordinates(vertex)
    goals_shape.append(VertexPointGoal(vertex, xyz))

error_shape = MeanSquaredError(goals_shape)

# ------------------------------------------------------------------------------
# Goal 2: Flatten the gridshell panels
# ------------------------------------------------------------------------------

goals_planarity = []
goals_planarity.append(MeshPlanarityGoal())

# Planarity is an energy to drive to zero, so it pairs with a prediction error
error_planarity = MeanPredictionError(goals_planarity, alpha=weight_planarity)

# ------------------------------------------------------------------------------
# Goal 3: Keep the mesh even and free of creases
# ------------------------------------------------------------------------------

goals_smoothness = []
goals_smoothness.append(MeshSmoothGoal())

# Smoothness is an energy too, so it also pairs with a prediction error
error_smoothness = MeanPredictionError(goals_smoothness, alpha=weight_smoothness)

# ------------------------------------------------------------------------------
# Assemble goals in the loss function
# ------------------------------------------------------------------------------

loss = Loss(error_shape, error_planarity, error_smoothness)

# ------------------------------------------------------------------------------
# You give and you take: Parameters
# ------------------------------------------------------------------------------

parameters = []
for edge in mesh.edges():
    parameter = EdgeForceDensityParameter(edge, q_bound_low, q_bound_up)
    parameters.append(parameter)

# Support finding: let the rim slide up and down, but not sideways
# Active if the z tolerance is greater than zero
if support_z_tolerance > 0.0:
    for vertex in mesh.vertices_supports():
        z = mesh.vertex_attribute(vertex, "z")
        parameter = VertexSupportZParameter(vertex, z - support_z_tolerance, z + support_z_tolerance)
        parameters.append(parameter)

# ------------------------------------------------------------------------------
# Solve the constrained form-finding problem
# ------------------------------------------------------------------------------

optimizer = LBFGSB()

opt_mesh = constrained_fdm(
    mesh,
    optimizer,
    loss,
    parameters=parameters,
    maxiter=max_iterations,
    tol=tol
)

opt_mesh.print_stats()

# ------------------------------------------------------------------------------
# Solution statistics
# ------------------------------------------------------------------------------

distance_mean, distance_max = distance_to_target(opt_mesh, mesh_target)

flatness = face_flatness(opt_mesh, max_deviation)
flatness_values = list(flatness.values())
flatness_mean = sum(flatness_values) / len(flatness_values)

num_buildable, percent_buildable = panels_buildable(flatness)

print(f"\nWeights\t\tplanarity {weight_planarity}\tsmoothness {weight_smoothness}")
print(f"Distance to target\tMean = {distance_mean:.4f} m\tMax = {distance_max:.4f} m")
print(f"Panel flatness\t\tMean = {flatness_mean:.2f}\tMax = {max(flatness_values):.2f}")
print(f"Panels within {max_deviation * 100:.0f}% tolerance\t"
      f"{num_buildable} of {len(flatness_values)} ({percent_buildable:.0f} %)")

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(width=1200, height=800, show_grid=True)

# The target is only a shape to aim at, so it carries no structural annotation
viewer.add(
    mesh_target,
    opacity=0.3,
    show_vertices=False,
    show_supports=False,
    show_loads=False,
    show_reactions=False,
)

# Form-found gridshell, each panel painted by how far it is from flat
viewer.add(
    opt_mesh,
    facecolor=face_colors(flatness),
    edgewidth=(0.02, 0.06),
    faceopacity=0.9,
    show_vertices=True,
    show_loads=True,
    show_reactions=False,
)

viewer.show()
