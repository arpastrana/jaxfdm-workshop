"""
Visualization functions for the form-finding examples.
"""

from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

from compas.colors import Color
from compas.geometry import Translation

from jax_fdm.visualization import Viewer

from helpers import create_network


_convergence_ax = None
_convergence_count = 0
_viewer = None
_view_shift_x = 0.0


# ------------------------------------------------------------------------------
# Convergence
# ------------------------------------------------------------------------------

def plot_convergence(residuals, label=None, title="Convergence"):
    """
    Plot the convergence of the residual norm.
    """

    global _convergence_ax
    global _convergence_count

    if _convergence_ax is None:
        _, ax = plt.subplots()
        ax.set_xlabel("Newton step")
        ax.set_ylabel("Residual norm")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":")
        _convergence_ax = ax
        _convergence_count = 0

    steps = []
    norms = []
    for i, residual in enumerate(residuals):
        steps.append(i)
        norms.append(float(residual))

    markers = ["o", "s", "^", "D", "v"]
    marker = markers[_convergence_count % len(markers)]

    _convergence_ax.semilogy(steps, norms, marker=marker, label=label)
    _convergence_ax.legend()
    _convergence_count = _convergence_count + 1

    return _convergence_ax


# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------

def _as_color(color):

    if color is None:
        return None

    rgb = to_rgb(color)
    return Color(rgb[0], rgb[1], rgb[2])


def horizontal_extent(network):
    """
    Width of a network along X, measured from its node coordinates.
    """
    # A planar network has a flat bounding box, which network.aabb() cannot frame
    xs = []
    for node in network.nodes():
        xs.append(network.node_attribute(node, "x"))

    return max(xs) - min(xs)


def show_plots():
    """
    Show the convergence chart and reset it for the next run.
    """
    global _convergence_ax
    global _convergence_count

    plt.show()

    _convergence_ax = None
    _convergence_count = 0


# ------------------------------------------------------------------------------
# Funicular shape (JAX FDM Viewer)
# ------------------------------------------------------------------------------

def view_structure(
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
    show_reactions=False,
    show_loads=True,
    name=None,
    ):
    """
    View the funicular shape of the network.
    """
    global _viewer
    global _view_shift_x

    if _viewer is None:
        _viewer = Viewer(width=1200, height=800, show_grid=True)
        _view_shift_x = 0.0

    if edgewidth is None:
        edgewidth = (0.02, 0.05)

    if residual is not None:
        show_reactions = True

    network = create_network(
        supports,
        x,
        load=load,
        forces=forces,
        residual=residual,
    )

    # Place each network beside the last, one bounding-box width along X
    box_width = horizontal_extent(network) * 1.2
    shift_vector = [_view_shift_x, 0.0, 0.0]
    translation = Translation.from_vector(shift_vector)
    network = network.transformed(translation)
    _view_shift_x = _view_shift_x + box_width

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


def show_viewer():
    """
    Show the viewer.
    """
    global _viewer
    global _view_shift_x

    if _viewer is not None:
        _viewer.show()

    _viewer = None
    _view_shift_x = 0.0
