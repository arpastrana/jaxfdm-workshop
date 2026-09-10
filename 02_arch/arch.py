"""
Exercise 1: form-find an arch by hand.

One force density value controls the whole arch. Change it until the crown
reaches the target rise.
"""
from compas.geometry import Plane

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Horizontal distance between the two supports
span = 10.0

# Number of straight segments along the arch
num_segments = 10

# Downward load applied at every free node
load_node = -1.0

# Height the crown should reach
target_rise = 3.0

# TUNE THIS ONE NUMBER
# Negative force densities put the arch in compression, positive in tension
q = -1.0

# ------------------------------------------------------------------------------
# Build the arch
# ------------------------------------------------------------------------------

network = FDNetwork()

# Start flat, on a straight line between the supports
for i in range(num_segments + 1):
    x = span * i / num_segments
    network.add_node(i, x=x, y=0.0, z=0.0)

for i in range(num_segments):
    edge = network.add_edge(i, i + 1)
    network.edge_forcedensity(edge, q)

# Fix both ends
network.node_support(0)
network.node_support(num_segments)

# Load every node in between
for i in range(1, num_segments):
    network.node_load(i, [0.0, 0.0, load_node])

# ------------------------------------------------------------------------------
# Solve for static equilibrium
# ------------------------------------------------------------------------------

eq_network = fdm(network)

# ------------------------------------------------------------------------------
# How close did we get?
# ------------------------------------------------------------------------------

apex_node = num_segments // 2
x, y, z = eq_network.node_coordinates(apex_node)
error = z - target_rise

print(f"Force density {q:+.3f}\tRise = {z:.3f}\tTarget = {target_rise:.3f}\tError = {error:+.3f}")

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Initial network
viewer.add(network, show_nodes=True, name="Initial")

# Form-found arch
viewer.add(eq_network, show_nodes=True, show_reactions=False, edgecolor="force", name="Equilibrium")

# Target rise plane
plane = Plane([span / 2.0, 0.0, target_rise], [0.0, 0.0, 1.0])
viewer.add(plane, planesize=span / 2.0, opacity=0.2, color=(1.0, 0.0, 1.0), show_lines=False)

viewer.show()
