"""
Four-bar form-finding with JAX FDM's numerical core.

Same geometry, load and force densities as four_bars_jaxfdm.py, so it lands on
the same shape. The difference is what the solver is handed: no network, only
the arrays the force density method actually needs.
"""

import jax.numpy as jnp
import numpy as np

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from helpers import create_network


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Initial free-node position (x, y, z)
x0 = [0.0, 0.0, 0.0]

# Support nodes (x, y, z)
supports_xyz = [
    [-1.0, 0.0, 0.0],
    [ 1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.5],
    [ 0.0, -1.0, 0.5],
]

# Load at the free node (x, y, z)
load = [0.0, 0.0, -1.0]

# Edge internal forces, signed: negative means compression
forces = [-0.5, -1.0, -1.0, -1.0]

# ------------------------------------------------------------------------------
# Structure: which nodes exist, what connects them, and which ones are held
# ------------------------------------------------------------------------------

# Node 0 is free, nodes 1 to 4 are the supports
nodes = np.asarray([0, 1, 2, 3, 4], dtype=int)
edges = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int)
supports = np.asarray([0, 1, 1, 1, 1], dtype=int)

structure = EquilibriumStructure(nodes, edges, supports)

# ------------------------------------------------------------------------------
# Parameters: force densities, support coordinates, and loads
# ------------------------------------------------------------------------------

xyz_free = jnp.asarray(x0)
xyz_fixed = jnp.asarray(supports_xyz)

# The force density is the force over the length of the starting geometry
edge_vectors = xyz_fixed - xyz_free
lengths = jnp.linalg.norm(edge_vectors, axis=1)
q = jnp.asarray(forces) / lengths

# One load vector per node, zero everywhere but the free node
loads_zero = jnp.zeros((len(nodes), 3))
loads_nodes = loads_zero.at[0].set(jnp.asarray(load))
loads = LoadState(nodes=loads_nodes, edges=0.0, faces=0.0)

parameters = EquilibriumParametersState(q=q, xyz_fixed=xyz_fixed, loads=loads)

# ------------------------------------------------------------------------------
# Solve: one linear FDM step
# ------------------------------------------------------------------------------

model = EquilibriumModel()
eq_state = model(parameters, structure)

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

# A network is only for drawing, the solve above never used one
network = create_network(supports_xyz, x0, load=load, forces=forces)
eq_network = datastructure_updated(network, eq_state, parameters)

viewer = Viewer(show_grid=True)

# Initial four-bar
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
