#!/usr/bin/env python3
from math import fabs

# compas
from compas.colors import Color
from compas.datastructures import Mesh
# from compas.datastructures import network_find_cycles
from compas.geometry import length_vector
from compas.geometry import Line
from compas.geometry import Polyline
# from compas.utilities import pairwise

# static equilibrium
from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import fdm

from jax_fdm.visualization import Plotter
from jax_fdm.visualization import Viewer


# ==========================================================================
# Initial parameters
# ==========================================================================

# geometry
nodes = {0: (0.0, 0.0, 0.0),
         1: (-1.0, 0.0, 0.0),
         2: (1.0, 0.0, 0.0),
         3: (0.0, 1.0, 0.5),
         4: (0.0, -1.0, 0.5),
         }

edges = {(0, 1): -0.5,
         (0, 2): -1.0,  # -0.5
         (0, 3): -1.0,
         (0, 4): -1.0,
         }

pz = -1.0
pz_iter = -0.6

supports = (1, 2, 3, 4)


view = True

# plotting
plot = True
plot_save = False
plot_extension = "pdf"

show_nodes = True
draw_connecting_lines = True

nodesizepolicy = "absolute"
nodesize = 4.0
edgewidth = 2.0

# ==========================================================================
# Create network
# ==========================================================================

network = FDNetwork()

# add nodes
for key, xyz in nodes.items():
    network.add_node(key, attr_dict={k: v for k, v in zip("xyz", xyz)})

# assign supports
for key in supports:
    network.node_support(key)

# add edges
for (u, v), force in edges.items():
    edge = network.add_edge(u, v)
    length = network.edge_length(edge)
    q = force / length
    print((u, v), f"q:{round(q, 2)}\tlength: {round(length, 2)}")
    network.edge_forcedensity((u, v), q)

# apply node load
network.node_load(0, [0.0, 0.0, pz])

# ==========================================================================
# Run the force density method
# ==========================================================================

print("\nFixed network")
fixed_network = network.copy()
fixed_network.node_support(0)
fixed_network = fdm(fixed_network, tmax=1, sparse=False, is_load_local=False)
fixed_network.print_stats()

key = 0
px, py, pz = fixed_network.node_attributes(key, ["rx", "ry", "rz"])
print(f"\tNode {key} r:{round(length_vector([px, py, pz]), 2)}")
print(f"\tNode {key} rx:{round(px, 2)}\try:{round(py, 2)}\trz:{round(pz, 2)}")

px, py, pz = fixed_network.node_attributes(key, ["px", "py", "pz"])
print(f"\tNode {key} px:{round(px, 2)}\tpy:{round(py, 2)}\tpz:{round(pz, 2)}")

for key in network.nodes():
    x, y, z = fixed_network.node_coordinates(key)
    print(f"\tNode {key} x:{round(x, 6)}\ty:{round(y, 6)}\tz:{round(z, 6)}")    

for edge in network.edges():
    force = fixed_network.edge_force(edge)
    print(f"\tEdge {edge} force:{round(force, 2)}")

print("\nForm found network")
eq_network = fdm(network, tmax=1, is_load_local=False)
eq_network.print_stats()

key = 0
px, py, pz = eq_network.node_attributes(key, ["px", "py", "pz"])
print(f"\tNode {key} px:{round(px, 2)}\tpy:{round(py, 2)}\tpz:{round(pz, 2)}")
for key in network.nodes():
    x, y, z = eq_network.node_coordinates(key)
    print(f"\tNode {key} x:{round(x, 9)}\ty:{round(y, 9)}\tz:{round(z, 9)}")

for edge in network.edges():
    force = eq_network.edge_force(edge)
    print(f"\tEdge {edge} force:{round(force, 2)}")

# print("\nForm found network with shape dependent loads")

# network.node_load(0, [0.0, 0.0, 0.0])
# apply edge loads
# pz_nodes_sum = pz_iter * network.number_of_nodes()
# pz_edge = pz_nodes_sum / network.number_of_edges()
# network.edges_loads([0.0, 0.0, pz_edge])
# py_edges_sum = pz_edge * network.number_of_edges()

# assert py_edges_sum == pz_nodes_sum

# network.nodes_loads([0.0, 0.0, 0.0])
# eq_network_iter = fdm(network, tmax=100, is_load_local=False, verbose=False)
# eq_network_iter.print_stats()

# key = 0
# px, py, pz = eq_network_iter.node_attributes(key, ["px", "py", "pz"])
# print(f"\tNode {key} px:{round(px, 2)}\tpy:{round(py, 2)}\tpz:{round(pz, 2)}")

# for key in network.nodes():
#     x, y, z = eq_network_iter.node_coordinates(key)
#     print(f"\tNode {key} x:{round(x, 6)}\ty:{round(y, 6)}\tz:{round(z, 6)}")

# for edge in network.edges():
#     force = eq_network_iter.edge_force(edge)
#     print(f"\tEdge {edge} force:{round(force, 2)}")

# ymax = max(eq_network.nodes_attribute("y"), key=lambda x: fabs(x))
# print(f"{q_init=}, ymax={round(ymax, 4)}")

# ==========================================================================
# Viewer
# ==========================================================================

if view:

    viewer = Viewer(width=900, height=900, show_grid=False)
    # viewer.view.camera.distance = 4.0  # 2.5

    # plot network with edge load in equilibrium
    viewer.add(eq_network,  # fixed_network, eq_network_iter
               show_nodes=False,
               show_loads=True,
               nodes=[0],
               show_reactions=True,
               nodesize=nodesize,
               edgecolor=Color.black(),
               edgewidth=(0.02, 0.05),
               reactionscale=1.0,
               loadscale=0.5)  # 0.25

    # plot network with edge load in equilibrium
    # viewer.add(eq_network,
    #            show_nodes=False,
    #            show_loads=True,
    #            show_reactions=False,
    #            nodesize=nodesize,
    #            edgecolor=Color.black(),
    #            edgewidth=(0.03, 0.05),
    #            reactionscale=1.0,
    #            loadscale=0.5)

    # plot network with edge load in equilibrium
    viewer.add(network,
               as_wireframe=True)

    # view polygon
    points = []
    for i in [1, 4, 2, 3, 1]:
        point = [c for c in nodes[i]]
        point[2] = 0.0
        points.append(point)
    viewer.add(Polyline(points))

    # add mesh
    # _network = eq_network.copy()
    # for a, b in pairwise([1, 4, 2, 3, 1]):
    #     _network.add_edge(a, b)
    # faces = network_find_cycles(_network)[1:]
    # vertices = {vkey: _network.node_coordinates(vkey) for vkey in network.nodes()}
    # mesh = Mesh.from_vertices_and_faces(vertices, faces)

    # viewer.add(mesh, opacity=0.4, show_points=False)

    viewer.show()

# ==========================================================================
# Plotter
# ==========================================================================

if plot:

    plotter = Plotter(dpi=150)

    # plot network with edge load in equilibrium
    # plotter.add(eq_network_iter,
    #             show_nodes=True,
    #             show_loads=True,
    #             show_reactions=False,
    #             nodesize=nodesize,
    #             edgecolor="force",
    #             edgewidth=edgewidth,
    #             reactionscale=1.0,
    #             sizepolicy=nodesizepolicy,
    #             loadscale=0.5)

    # plot network with static load in equilibrium
    for edge in network.edges():
        edge_line = Line(*eq_network.edge_coordinates(edge))
        plotter.add(edge_line,
                    draw_as_segment=True,
                    linestyle="dashed",  # linestyles: "solid", "dotted", "dashed", "dashdot"
                    linewidth=edgewidth,
                    color=Color.from_rgb255(12, 119, 184),
                    )

    # plot starting network
    plotter.add(network,
                show_nodes=True,
                show_loads=True,
                show_reactions=False,
                nodesize=nodesize,
                edgecolor=Color(0.9, 0.9, 0.9),
                edgewidth=edgewidth,
                reactionscale=1.0,
                sizepolicy=nodesizepolicy,
                loadscale=0.5)

    # for u, v in network.edges():
    #     edge_line = Line(*network.edge_coordinates(u, v))
    #     plotter.add(edge_line,
    #                 draw_as_segment=True,
    #                 linestyle="solid",  # linestyles: "solid", "dotted", "dashed", "dashdot"
    #                 linewidth=edgewidth,
    #                 color=Color(0.9, 0.9, 0.9),
    #                 )

    # if draw_connecting_lines:
    #     for node in network.nodes():
    #         line = Line(eq_network.node_coordinates(node),
    #                     eq_network_iter.node_coordinates(node))
    #         plotter.add(line,
    #                     draw_as_segment=True,
    #                     linestyle="dashed",  # linestyles: "solid", "dotted", "dashed", "dashdot"
    #                     linewidth=0.25,
    #                     color=Color.black(),
    #                     zorder=5000)

    # save le crème
    plotter.zoom_extents()

    if plot_save:
        filepath = f"twobar.{plot_extension}"

        plotter.save(filepath,
                     bbox_inches='tight',
                     transparent=True,
                     pad_inches=0.0)

        print(f"Saved plot to {filepath}")

    plotter.show()
