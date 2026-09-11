"""
Helpers to build networks of arches.
"""
from jax_fdm.datastructures import FDNetwork


# Nodes closer than this many decimal places are treated as the same node
DECIMALS = 6


def create_arch_network(span, num_segments):
    """
    Build a flat arch of straight segments, centered on the origin and running along X.
    """
    network = FDNetwork()

    previous = None
    for i in range(num_segments + 1):
        x = -0.5 * span + span * i / num_segments
        node = network.add_node(x=x, y=0.0, z=0.0)
        if previous is not None:
            network.add_edge(previous, node)
        previous = node

    return network


def node_point(network, node):
    """
    Round a node position so that two nodes in the same place compare equal.
    """
    x, y, z = network.node_coordinates(node)

    return (round(x, DECIMALS), round(y, DECIMALS), round(z, DECIMALS))


def fuse_networks(first, second):
    """
    Copy the second network into the first, reusing the nodes they have in common.

    Returns the fused network and the keys of the nodes the two networks share.
    """
    network = first.copy()

    # Look up nodes by position, so a node in both networks is found not duplicated
    keys_by_point = {}
    for node in network.nodes():
        keys_by_point[node_point(network, node)] = node

    shared = []
    keys_by_node = {}
    for node in second.nodes():
        point = node_point(second, node)
        if point in keys_by_point:
            shared.append(keys_by_point[point])
        else:
            x, y, z = second.node_coordinates(node)
            keys_by_point[point] = network.add_node(x=x, y=y, z=z)
        keys_by_node[node] = keys_by_point[point]

    for u, v in second.edges():
        network.add_edge(keys_by_node[u], keys_by_node[v])

    return network, shared
