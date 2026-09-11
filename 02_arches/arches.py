"""
Form-find two arches that cross at the crown.

Each arch has its own force density. Change them until the shared crown
reaches the target rise.
"""
from compas.geometry import Plane

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Horizontal distance between opposite supports
span = 10.0

# Number of straight segments along each arch
# Keep this even so the two arches share a node at the center
num_segments = 10

# Downward load applied at every free node
load_node = -1.0

# Height the shared crown should reach
target_rise = 3.0

# TUNE THIS NUMBER
# Negative force densities put the arches in compression, positive in tension
q = -2.0

# ------------------------------------------------------------------------------
# First arch, along X
# ------------------------------------------------------------------------------

network = FDNetwork()

# Start flat, on a straight line between the supports
for i in range(num_segments + 1):
    x = -0.5 * span + span * i / num_segments
    network.add_node(i, x=x, y=0.0, z=0.0)

for i in range(num_segments):
    edge = network.add_edge(i, i + 1)

# ------------------------------------------------------------------------------
# Second arch, along Y, sharing the crown
# ------------------------------------------------------------------------------

# The first arch's middle node is the crossing
crown = num_segments // 2

# Walk from y = -span/2 to y = +span/2. Reuse the crown at y = 0
keys_y = []
next_key = num_segments + 1

for i in range(num_segments + 1):
    y = -0.5 * span + span * i / num_segments
    if i == crown:
        keys_y.append(crown)
    else:
        network.add_node(next_key, x=0.0, y=y, z=0.0)
        keys_y.append(next_key)
        next_key = next_key + 1

for i in range(num_segments):
    edge = network.add_edge(keys_y[i], keys_y[i + 1])

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

# Free nodes of the first arch, including the crown
for node in network.nodes_free():
    network.node_load(node, [0.0, 0.0, load_node])

# ------------------------------------------------------------------------------
# Solve for static equilibrium
# ------------------------------------------------------------------------------

eq_network = fdm(network)

# ------------------------------------------------------------------------------
# How close did we get?
# ------------------------------------------------------------------------------

x, y, z = eq_network.node_coordinates(crown)
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
