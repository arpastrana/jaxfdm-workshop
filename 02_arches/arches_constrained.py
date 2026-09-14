"""
Let the computer find the force densities of two arches that cross at the crown.

Same structure as arches.py, same target rise. Now we state the target as a
goal and an optimizer searches for the force densities that reach it.
"""
from math import radians

from compas.geometry import Plane
from compas.geometry import Rotation

from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NodeZCoordinateGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import GradientDescent
from jax_fdm.optimization import OptimizationRecorder
from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.visualization import LossPlotter
from jax_fdm.visualization import Viewer

from helpers import create_arch_network
from helpers import fuse_networks


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Horizontal distance between opposite supports
span = 10.0

# Number of straight segments along each arch, keep it even so they share a crown
num_segments = 10

# Downward load applied at every free node
load_node = -1.0

# Height the shared crown should reach
target_rise = 3.0

# Force density to start with
q = -2.0

# Optimizer budget, gradient descent needs a tight tolerance to stop on the answer
learning_rate = 0.05
max_iterations = 5000
tol = 1e-9

# ------------------------------------------------------------------------------
# Two arches crossing at the crown
# ------------------------------------------------------------------------------

arch_x = create_arch_network(span, num_segments)

# The second arch is the first one turned a quarter turn about the vertical axis
rotation = Rotation.from_axis_and_angle([0.0, 0.0, 1.0], radians(90.0))
arch_y = arch_x.transformed(rotation)

# Fusing welds the node the two arches have in common
network, shared_nodes = fuse_networks(arch_x, arch_y)
crown_node = shared_nodes.pop()

# ------------------------------------------------------------------------------
# Assign force densities
# ------------------------------------------------------------------------------

for edge in network.edges():
    network.edge_forcedensity(edge, q)

# ------------------------------------------------------------------------------
# Supports
# ------------------------------------------------------------------------------

for node in network.nodes():
    if network.is_leaf(node):
        network.node_support(node)

# ------------------------------------------------------------------------------
# Loads
# ------------------------------------------------------------------------------

for node in network.nodes_free():
    network.node_load(node, [0.0, 0.0, load_node])

# ------------------------------------------------------------------------------
# Starting point in equilibrium
# ------------------------------------------------------------------------------

eq_network = fdm(network)

# ------------------------------------------------------------------------------
# How close did we get to the target?
# ------------------------------------------------------------------------------

x, y, z = eq_network.node_coordinates(crown_node)
error = z - target_rise
print(f"Rise = {z:.3f}\tTarget = {target_rise:.3f}\tError = {error:+.3f}")

# ------------------------------------------------------------------------------
# State the design intent
# ------------------------------------------------------------------------------

# What we want: the crown at the target height
goals = []
goals.append(NodeZCoordinateGoal(crown_node, target_rise))

# How we measure the miss
loss = Loss(SquaredError(goals))

# ------------------------------------------------------------------------------
# Optimization parameters
# ------------------------------------------------------------------------------

# Every edge gets its own force density, so the problem has twenty unknowns
# For one force density shared by every edge, swap the loop below for these lines
from jax_fdm.parameters import EdgeGroupForceDensityParameter
parameters = [EdgeGroupForceDensityParameter(list(network.edges()))]

# parameters = []
# for edge in network.edges():
#     parameter = EdgeForceDensityParameter(edge)
#     parameters.append(parameter)

# ------------------------------------------------------------------------------
# Solve the inverse problem
# ------------------------------------------------------------------------------

# Follow the gradient downhill with a fixed step size until convergence
optimizer = GradientDescent(learning_rate=learning_rate)

# The recorder stores the parameters per iteration, to plot the loss afterwards
recorder = OptimizationRecorder(optimizer)

opt_network = constrained_fdm(
    network,
    optimizer,
    loss,
    parameters=parameters,
    maxiter=max_iterations,
    tol=tol,
    callback=recorder,
)

opt_network.print_stats()

# ------------------------------------------------------------------------------
# How close did we get to the target?
# ------------------------------------------------------------------------------

x, y, z = opt_network.node_coordinates(crown_node)
error = z - target_rise
print(f"Rise = {z:.3f}\tTarget = {target_rise:.3f}\tError = {error:+.3f}")

# ------------------------------------------------------------------------------
# Plot the loss history
# ------------------------------------------------------------------------------

plotter = LossPlotter(loss, network, dpi=150, figsize=(8, 4))
plotter.plot(recorder.history)
plotter.show()

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Form-found cross in equilibrium
viewer.add(eq_network, show_nodes=True, show_reactions=False, name="Equilibrium")

# Form-found cross after optimization
viewer.add(opt_network, show_nodes=True, show_reactions=False, edgecolor="force", name="Opt")

# Target rise plane
plane = Plane([0.0, 0.0, target_rise], [0.0, 0.0, 1.0])
viewer.add(plane, planesize=span / 2.0, opacity=0.2, color=(1.0, 0.0, 1.0), show_lines=False, name="Target")

viewer.show()
