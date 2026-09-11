"""
Four-bar form-finding with JAX FDM.

Same geometry, load and force densities as four_bars.py, so it lands on the
same shape. The difference is the solver: one linear FDM solve here, a local
Newton loop there.
"""

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Initialize an empty network
# ------------------------------------------------------------------------------

network = FDNetwork()

# ------------------------------------------------------------------------------
# Add the free node
# ------------------------------------------------------------------------------

free_node = network.add_node(x=0.0, y=0.0, z=0.0)

# ------------------------------------------------------------------------------
# Add fixed nodes
# ------------------------------------------------------------------------------

supports_xyz = [
    [-1.0, 0.0, 0.0],
    [ 1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.5],
    [ 0.0, -1.0, 0.5],
]

fixed_nodes = []

for xyz in supports_xyz:
    # add a fixed node at the support location
    x, y, z = xyz
    fixed_node = network.add_node(x=x, y=y, z=z)

    # fix the node
    network.node_support(fixed_node)

    # add the node to the list of fixed nodes for later use
    fixed_nodes.append(fixed_node)

# ------------------------------------------------------------------------------
# Add edges
# ------------------------------------------------------------------------------

# Edge internal forces, signed: negative means compression
forces = [-0.5, -1.0, -1.0, -1.0]

for i, fixed_node in enumerate(fixed_nodes):
    # create an edge
    edge = network.add_edge(free_node, fixed_node)

    # the force density is the force over the length of the starting geometry
    length = network.edge_length(edge)
    network.edge_forcedensity(edge, forces[i] / length)

# ------------------------------------------------------------------------------
# Apply the load
# ------------------------------------------------------------------------------

load = [0.0, 0.0, -1.0]
network.node_load(free_node, load)

# ------------------------------------------------------------------------------
# Solve the form-finding problem with the linear FDM solve
# ------------------------------------------------------------------------------

eq_network = fdm(network)

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Initial network
viewer.add(
    network,
    edgewidth=0.02,
    show_reactions=False,
    show_nodes=True,
    name="Initial",
    show_loads=False,
    )

# Form-found four-bar
viewer.add(
    eq_network,
    edgewidth=0.05,
    show_nodes=True,
    show_reactions=True,
    edgecolor="force",
    name="Equilibrium",
    )

viewer.show()
