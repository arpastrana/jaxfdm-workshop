"""
Arch form-finding.
"""
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer


# Parameters
q = -5.0
pz = -1.0

# create empty network
network = FDNetwork()

# Adding nodes
node_keys = []
for x in range(10):
    key = network.add_node(x=x, y=0.0, z=0.0)
    node_keys.append(key)

# Adding edges
for i in range(len(node_keys) - 1):
    network.add_edge(node_keys[i], node_keys[i + 1])

# Define structural system
for edge in network.edges():
    network.edge_forcedensity(edge, q)

# Supports
network.node_support(node_keys[0])
network.node_support(node_keys[-1])

# Loads
for node in network.nodes():
    network.node_load(node, [0.0, 0.0, pz])

# Form-find
eq_network = fdm(network)

# What is the maximum height?
rise = max(eq_network.node_coordinates(node)[2] for node in eq_network.nodes())
print(f"Maximum height: {rise}")


# Target rise = 5

# Parameters
from jax_fdm.parameters import EdgeForceDensityParameter
parameters = []
for edge in eq_network.edges():
    parameter = EdgeForceDensityParameter(edge, bound_low=-100.0, bound_up=-1.0e-6)
    parameters.append(parameter)

# Define the loss function
target_node = 4

from jax_fdm.goals import NodeZCoordinateGoal

goal_rise_1 = NodeZCoordinateGoal(target_node, 5.0, weight=1.0)
goal_rise_2 = NodeZCoordinateGoal(int(target_node + 1), 5.0, weight=1.0)

# from jax_fdm.goals import EdgeForceGoal
# goal_force = EdgeForceGoal(edge, target=-10.0, weight=3.0)
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError

error = SquaredError([goal_rise_1, goal_rise_2], alpha=1.0)
loss = Loss(error)

# Optimization
from jax_fdm.optimization import LBFGSB
optimizer = LBFGSB()

# Solve the constrained optimization problem
from jax_fdm.equilibrium import constrained_fdm

opt_network = constrained_fdm(
    network,
    optimizer,
    loss,
    parameters,
)

# Visualize solution
viewer = Viewer(show_grid=True)
viewer.add(eq_network, show_nodes=True, show_load=True)
viewer.add(opt_network, show_nodes=True, show_load=True, edgecolor="force")
viewer.show()
