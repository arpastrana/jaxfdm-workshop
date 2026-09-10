from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

from compas.colors import Color
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import Plotter


# ------------------------------------------------------------------------------
# Convergence
# ------------------------------------------------------------------------------

def plot_convergence(residual_histories, labels, title="Convergence"):

    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v"]

    for i in range(len(residual_histories)):

        residuals = residual_histories[i]

        steps = []
        norms = []
        for j in range(len(residuals)):
            steps.append(j)
            norms.append(float(residuals[j]))

        marker = markers[i % len(markers)]
        ax.semilogy(steps, norms, marker=marker, label=labels[i])

    ax.set_xlabel("Newton step")
    ax.set_ylabel("Residual norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle=":")

    return fig, ax


# ------------------------------------------------------------------------------
# Form-finding geometry (JAX FDM Plotter)
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
        return Color.black()

    rgb = to_rgb(color)
    return Color(rgb[0], rgb[1], rgb[2])


def network_from_free_node(supports, x, load=None, force_densities=None):

    network = FDNetwork()

    xyz = _as_xyz(x)
    network.add_node(0, attr_dict={"x": xyz[0], "y": xyz[1], "z": xyz[2]})

    if load is not None:
        network.node_load(0, _as_xyz(load))

    for i in range(len(supports)):
        key = i + 1
        xyz = _as_xyz(supports[i])
        network.add_node(key, attr_dict={"x": xyz[0], "y": xyz[1], "z": xyz[2]})
        network.node_support(key)
        network.add_edge(0, key)

        if force_densities is not None:
            network.edge_forcedensity((0, key), float(force_densities[i]))

    return network


def plot_formfinding(supports, configurations, load=None, load_at=None, title="Form finding"):

    plotter = Plotter(dpi=150)

    for i in range(len(configurations)):

        config = configurations[i]

        color = Color.black()
        if "color" in config:
            color = _as_color(config["color"])

        linewidth = 2.0
        if "linewidth" in config:
            linewidth = config["linewidth"]

        force_densities = None
        if "q" in config:
            force_densities = config["q"]

        network_load = None
        show_loads = False
        if load is not None and i == len(configurations) - 1:
            network_load = load
            show_loads = True

        network = network_from_free_node(
            supports,
            config["x"],
            load=network_load,
            force_densities=force_densities,
        )

        plotter.add(
            network,
            show_nodes=True,
            show_loads=show_loads,
            show_reactions=False,
            nodesize=4.0,
            edgecolor=color,
            edgewidth=linewidth,
            sizepolicy="absolute",
            loadscale=0.5,
        )

    plotter.zoom_extents()
    plotter.show()

    return plotter


def show_plots():

    plt.show()
