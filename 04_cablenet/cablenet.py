"""
Form-find a square cable-net in tension.

Two opposite corners are lifted. One force density, kept positive, puts
every cable in tension. There is no applied load, so changing q scales
the forces and leaves the shape alone.
"""

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import fdm
from jax_fdm.visualization import Viewer


# ------------------------------------------------------------------------------
# Design parameters
# ------------------------------------------------------------------------------

# Side length of the square net
length = 10.0

# Number of grid cells per side
num_cells = 10

# Lift of two opposite corner supports
support_height = 5.0

# TUNE THIS NUMBER
# Positive force densities put the cables in tension
q = 1.0

# ------------------------------------------------------------------------------
# Build a square cable-net
# ------------------------------------------------------------------------------

mesh = FDMesh.from_meshgrid(length, nx=num_cells)

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
# Assign force densities
# ------------------------------------------------------------------------------

for edge in mesh.edges():
    mesh.edge_forcedensity(edge, q)

# ------------------------------------------------------------------------------
# Form-find the tensile net
# ------------------------------------------------------------------------------

eq_mesh = fdm(mesh)

# ------------------------------------------------------------------------------
# Forces, not shape, are what q controls here
# ------------------------------------------------------------------------------

eq_mesh.print_stats()

# ------------------------------------------------------------------------------
# Visualize the results
# ------------------------------------------------------------------------------

viewer = Viewer(show_grid=True)

# Starting net, two corners already lifted
viewer.add(mesh, show_vertices=True, show_reactions=False, name="Initial")

# Form-found cable-net
viewer.add(
    eq_mesh,
    edgecolor="force",
    show_vertices=True,
    show_reactions=False,
    faceopacity=0.5,
    name="Equilibrium",
)

viewer.show()
