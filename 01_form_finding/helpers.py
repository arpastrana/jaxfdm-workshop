"""
Helpers for form-finding.
"""
import jax
import jax.numpy as jnp

from jax_fdm.datastructures import FDNetwork


def compute_vector_norm(vector):
    """
    Euclidean length of a vector.
    """
    squared = vector * vector
    length_squared = jnp.sum(squared)
    length = jnp.sqrt(length_squared)
    return length


def compute_vectors_norm(vectors):
    """
    Euclidean length of each row, as a column.
    """
    lengths = jax.vmap(compute_vector_norm, in_axes=0, out_axes=0)(vectors)
    lengths = jnp.expand_dims(lengths, axis=1)
    return lengths


def _as_xyz(point):
    """
    Convert a point to an XYZ coordinate.
    """
    x = float(point[0])
    y = float(point[1])
    z = 0.0
    if len(point) > 2:
        z = float(point[2])

    return [x, y, z]


def _as_scalar(value):
    """
    Convert a value to a scalar.
    """
    if getattr(value, "ndim", 0) > 0:
        value = value[0]

    return float(value)


def create_chain_network(supports, x, loads=None, forces=None, residuals=None):
    """
    Create a JAX FDM network from a chain of free nodes between two supports.
    """
    network = FDNetwork()

    # The chain runs support, free nodes in order, support
    points = [supports[0]]
    for position in x:
        points.append(position)
    points.append(supports[1])

    for key, point in enumerate(points):
        px, py, pz = _as_xyz(point)
        network.add_node(key, x=px, y=py, z=pz)

    last = len(points) - 1
    network.node_support(0)
    network.node_support(last)

    for key in range(1, last):
        if loads is not None:
            lx, ly, lz = _as_xyz(loads[key - 1])
            network.node_load(key, [lx, ly, lz])

        if residuals is not None:
            # JAX FDM stores rx, ry, rz as load - internal, so negate it before drawing
            rx, ry, rz = _as_xyz(residuals[key - 1])
            network.node_attributes(key, ["rx", "ry", "rz"], [-rx, -ry, -rz])

    for key in range(last):
        edge = network.add_edge(key, key + 1)

        if forces is not None:
            force = _as_scalar(forces[key])
            length = network.edge_length(edge)
            network.edge_attribute(edge, "force", force)
            network.edge_forcedensity(edge, force / length)

    return network


def create_network(supports, x, load=None, forces=None, residual=None):
    """
    Create a JAX FDM network from a free node.
    """
    network = FDNetwork()

    x, y, z = _as_xyz(x)
    network.add_node(0, x=x, y=y, z=z)

    if load is not None:
        px, py, pz = _as_xyz(load)
        network.node_load(0, [px, py, pz])

    if residual is not None:
        # JAX FDM stores rx, ry, rz as load - internal, so negate it before drawing
        rx, ry, rz = _as_xyz(residual)
        network.node_attributes(0, ["rx", "ry", "rz"], [-rx, -ry, -rz])

    for i, support in enumerate(supports):
        key = i + 1
        x, y, z = _as_xyz(support)
        network.add_node(key, x=x, y=y, z=z)
        network.node_support(key)
        network.add_edge(0, key)

        if forces is not None:
            force = _as_scalar(forces[i])
            length = network.edge_length((0, key))
            network.edge_attribute((0, key), "force", force)
            network.edge_forcedensity((0, key), force / length)

    return network
