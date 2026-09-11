"""
Form-find two arches that cross at the crown.

One force density controls every edge of both arches. Change it until the
shared crown reaches the target rise.
"""
from math import radians

from compas.geometry import Plane
from compas.geometry import Rotation

from jax_fdm.equilibrium import fdm
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

# TUNE THIS NUMBER
# Negative force densities put the arches in compression, positive in tension
q = -2.5

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
# Loads on the free nodes
# ------------------------------------------------------------------------------

for node in network.nodes_free():
    network.node_load(node, [0.0, 0.0, load_node])

# ------------------------------------------------------------------------------
# Form-find the arch
# ------------------------------------------------------------------------------

eq_network = fdm(network)

# ------------------------------------------------------------------------------
# How close did we get to the target?
# ------------------------------------------------------------------------------

x, y, z = eq_network.node_coordinates(crown_node)
error = z - target_rise
print(f"q {q:+.3f}\tRise = {z:.3f}\tTarget = {target_rise:.3f}\tError = {error:+.3f}")

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Initial network
viewer.add(network, show_nodes=True, name="Initial")

# Form-found cross
viewer.add(eq_network, show_nodes=True, show_reactions=False, edgecolor="force", name="Equilibrium")

# Target rise plane
plane = Plane([0.0, 0.0, target_rise], [0.0, 0.0, 1.0])
viewer.add(plane, planesize=span / 2.0, opacity=0.2, color=(1.0, 0.0, 1.0), show_lines=False, name="Target")

viewer.show()
