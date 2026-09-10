from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

from compas.colors import Color
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import Plotter
from jax_fdm.visualization import Viewer


_convergence_ax = None
_convergence_count = 0
_plotter = None
_viewer = None


# ------------------------------------------------------------------------------
# Convergence
# ------------------------------------------------------------------------------

def plot_convergence(residuals, label=None, title="Convergence"):

    global _convergence_ax
    global _convergence_count

    if _convergence_ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("Newton step")
        ax.set_ylabel("Residual norm")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":")
        _convergence_ax = ax
        _convergence_count = 0

    steps = []
    norms = []
    for i in range(len(residuals)):
        steps.append(i)
        norms.append(float(residuals[i]))

    markers = ["o", "s", "^", "D", "v"]
    marker = markers[_convergence_count % len(markers)]

    _convergence_ax.semilogy(steps, norms, marker=marker, label=label)
    _convergence_ax.legend()
    _convergence_count = _convergence_count + 1

    return _convergence_ax


# ------------------------------------------------------------------------------
# Funicular shape (JAX FDM Plotter)
# ------------------------------------------------------------------------------

def _as_xyz(point):

    x = float(point[0])
    y = float(point[1])
    z = 0.0
    if len(point) > 2:
        z = float(point[2])

    return [x, y, z]


def _as_color(color):

    if color is None:
        return None

    rgb = to_rgb(color)
    return Color(rgb[0], rgb[1], rgb[2])


def _as_scalar(value):

    if getattr(value, "ndim", 0) > 0:
        value = value[0]

    return float(value)


def network_from_free_node(supports, x, load=None, forces=None, residual=None):

    network = FDNetwork()

    xyz = _as_xyz(x)
    network.add_node(0, attr_dict={"x": xyz[0], "y": xyz[1], "z": xyz[2]})

    if load is not None:
        network.node_load(0, _as_xyz(load))

    if residual is not None:
        # JAX FDM stores rx, ry, rz as load - internal. Our residual is
        # internal - load, so negate it before drawing.
        residual_xyz = _as_xyz(residual)
        network.node_attributes(
            0,
            ["rx", "ry", "rz"],
            [-residual_xyz[0], -residual_xyz[1], -residual_xyz[2]],
        )

    for i in range(len(supports)):
        key = i + 1
        xyz = _as_xyz(supports[i])
        network.add_node(key, attr_dict={"x": xyz[0], "y": xyz[1], "z": xyz[2]})
        network.node_support(key)
        network.add_edge(0, key)

        if forces is not None:
            force = _as_scalar(forces[i])
            length = network.edge_length((0, key))
            network.edge_attribute((0, key), "force", force)
            network.edge_forcedensity((0, key), force / length)

    return network


def plot_funicular_shape(
    supports, 
    x, 
    load=None, 
    forces=None, 
    residual=None,
    color=None, 
    linewidth=2.0, 
    nodesize=4.0, 
    loadscale=0.25,
    residualscale=0.5,
    ):

    global _plotter

    if _plotter is None:
        _plotter = Plotter(dpi=150)

    show_loads = False
    show_reactions = False
    if load is not None:
        show_loads = True
    if residual is not None:
        show_reactions = True

    network = network_from_free_node(
        supports,
        x,
        load=load,
        forces=forces,
        residual=residual,
    )

    _plotter.add(
        network,
        show_nodes=True,
        show_loads=show_loads,
        show_reactions=show_reactions,
        nodesize=nodesize,
        edgecolor=_as_color(color),
        edgewidth=linewidth,
        sizepolicy="absolute",
        loadscale=loadscale,
        reactionscale=residualscale,
    )

    return _plotter


# ------------------------------------------------------------------------------
# Funicular shape (JAX FDM Viewer)
# ------------------------------------------------------------------------------

def view_funicular_shape(
    supports,
    x,
    load=None,
    forces=None,
    residual=None,
    color=None,
    nodesize=0.1,
    loadscale=0.5,
    residualscale=1.0,
    edgewidth=None,
    as_wireframe=False,
    show_reactions=False,
    show_loads=True,
    name=None,
    ):

    global _viewer

    if _viewer is None:
        _viewer = Viewer(width=900, height=900, show_grid=False)

    if edgewidth is None:
        edgewidth = (0.02, 0.05)

    if residual is not None:
        show_reactions = True

    network = network_from_free_node(
        supports,
        x,
        load=load,
        forces=forces,
        residual=residual,
    )

    if as_wireframe:
        _viewer.add(network, as_wireframe=True, name=name)
    else:
        _viewer.add(
            network,
            show_nodes=True,
            show_loads=show_loads,
            show_reactions=show_reactions,
            nodesize=nodesize,
            edgecolor=_as_color(color),
            edgewidth=edgewidth,
            reactionscale=residualscale,
            loadscale=loadscale,
            name=name,
        )

    return _viewer


def show_plots():

    global _convergence_ax
    global _convergence_count
    global _plotter
    global _viewer

    if _plotter is not None:
        _plotter.zoom_extents()
        _plotter.show()

    plt.show()

    if _viewer is not None:
        _viewer.show()

    _convergence_ax = None
    _convergence_count = 0
    _plotter = None
    _viewer = None
