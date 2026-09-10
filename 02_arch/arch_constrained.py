"""
Exercise 2: let the computer find the force density.

Same arch as Exercise 1, same target rise. 
Now we state the target as a goal and an optimizer searches for the force density that reaches it.
"""

from compas.geometry import Plane

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import constrained_fdm
from jax_fdm.equilibrium import fdm
from jax_fdm.goals import NodeZCoordinateGoal
from jax_fdm.losses import Loss
from jax_fdm.losses import SquaredError
from jax_fdm.optimization import GradientDescent
from jax_fdm.parameters import EdgeGroupForceDensityParameter
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

span = 10.0
num_segments = 10
load_node = -1.0

# Height the apex should reach
target_rise = 3.0

# Force density to start with
q = -1.0

# How far the optimizer may move the force density
q_bound_low = -20.0
q_bound_up = -0.1

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
# State the design intent
# ------------------------------------------------------------------------------

# What we want: the apex at the target height
goals = []
my_goal = NodeZCoordinateGoal(apex_node, target_rise)
goals.append(my_goal)

# How we measure the miss
loss = Loss(SquaredError(goals))

# What the optimizer may change: one force density shared by every edge
parameters = []
edges = list(network.edges())
parameter = EdgeGroupForceDensityParameter(edges, q_bound_low, q_bound_up)
parameters.append(parameter)

# ------------------------------------------------------------------------------
# Solve the inverse problem
# ------------------------------------------------------------------------------

# Initialize the optimizer: Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
# optimizer = LBFGSB()
optimizer = GradientDescent()

opt_network = constrained_fdm(
    network,
    optimizer,
    loss,
    parameters=parameters,
    maxiter=500,
)

# ------------------------------------------------------------------------------
# What did it find?
# ------------------------------------------------------------------------------

q = parameter.evaluate(opt_network)
x, y, z = opt_network.node_coordinates(apex_node)
error = z - target_rise

print(f"\nForce density {q:+.3f}\tRise = {z:.3f}\tTarget = {target_rise:.3f}\tError = {error:+.3f}")

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Initial network
viewer.add(network, show_nodes=True, name="Initial")

# Form-found arch
viewer.add(eq_network, show_nodes=True, show_reactions=False, name="Equilibrium")

# Constrained arch
viewer.add(opt_network, show_nodes=True, show_reactions=False, edgecolor="force", name="Opt")

# Target rise plane
plane = Plane([span / 2.0, 0.0, target_rise], [0.0, 0.0, 1.0])
viewer.add(plane, planesize=span / 2.0, opacity=0.2, color=(1.0, 0.0, 1.0), show_lines=False)

# Show le creme
viewer.show()
