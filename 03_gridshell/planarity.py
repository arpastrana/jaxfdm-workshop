"""
Measuring and coloring how buildable the panels of a gridshell are.
"""

from math import sqrt

from compas.colors import ColorMap


def face_flatness(mesh, max_deviation):
    """
    Flatness of every quad face, as a fraction of the buildability tolerance.

    A flat quad has intersecting diagonals. A warped one leaves a gap between
    them, and that gap over the average edge length is the flatness. Dividing by
    the tolerance puts a panel at 1.0 exactly when it is as warped as allowed.
    """
    flatness = {}
    for face in mesh.faces():
        flatness[face] = mesh.face_flatness(face, maxdev=max_deviation)

    return flatness


def face_colors(flatness):
    """
    Color every face from flat to warped.
    """
    colormap = ColorMap.from_mpl("plasma")

    values = list(flatness.values())
    low = min(values)
    high = max(values)

    colors = {}
    for face, value in flatness.items():
        colors[face] = colormap((value - low) / (high - low))

    return colors


def panels_buildable(flatness):
    """
    How many panels warp less than the tolerance allows, and what share that is.
    """
    values = list(flatness.values())

    num_buildable = 0
    for value in values:
        if value <= 1.0:
            num_buildable += 1

    return num_buildable, 100.0 * num_buildable / len(values)


def distance_to_target(mesh, mesh_target):
    """
    Mean and largest distance between the free vertices and the target.
    """
    distances = []
    for vertex in mesh.vertices_free():
        x, y, z = mesh.vertex_coordinates(vertex)
        xt, yt, zt = mesh_target.vertex_coordinates(vertex)
        distances.append(sqrt((x - xt) ** 2 + (y - yt) ** 2 + (z - zt) ** 2))

    return sum(distances) / len(distances), max(distances)
