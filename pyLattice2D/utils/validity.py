import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from pyLattice2D.utils.math import intersect


def is_connected(edge_list):
    '''
    Function testing whether a graph contains disconnected subgraphs. Allows isolated nodes.

    Input
    edge_list: list of edges [[1,2], [2,3], ...]

    Output
    Boolean value that returns True if the graph contains no disconnected subgraphs.

    '''
    num_nodes = len(set(np.array(edge_list).flatten()))

    graph = nx.Graph()
    for s,t in edge_list:
        nx.add_path(graph, [s,t])

    reachable_nodes = nx.shortest_path(graph, np.min(edge_list)).keys()

    if len(reachable_nodes) == num_nodes:
        return True
    else:
        return False

def no_crossings(edge_list, coordinates):
    '''
    Function that checks if there are no edge crossings.

    Inputs
    edge_list: list of edges [[0,1], [2,3], ...]
    coordinates: list of node coordinates [[x0,y0], [x1,y1], ...]

    Returns False if two edges cross.
    '''
    # check all edges against each other, but stop early if an intersecting pair was found
    for i in range(len(edge_list)):
        for j in range(i+1,len(edge_list)):

            overlapping_node = None
            if edge_list[i][0] in edge_list[j]:
                overlapping_node = edge_list[i][0]
            elif edge_list[i][1] in edge_list[j]:
                overlapping_node = edge_list[i][1]
            # special case: check if there is an overlapping node
            if overlapping_node is not None:
                first_point, last_point = list(set(np.array([edge_list[i], edge_list[j]]).flatten()).difference({overlapping_node}))
                vec0 = coordinates[first_point]-coordinates[overlapping_node]
                vec1 = coordinates[overlapping_node] - coordinates[last_point]
                # special case: check if the edges lie above each other (or if the node is a corner)
                if cosine(vec0,vec1) == 2:
                    return False
                else:
                    continue

            # standard case: check if line segments cross
            else:
                p1 = coordinates[edge_list[i][0]]
                p2 = coordinates[edge_list[i][1]]
                p3 = coordinates[edge_list[j][0]]
                p4 = coordinates[edge_list[j][1]]

                if intersect(p1, p2, p3, p4):
                    return False

    return True

def is_graph_valid(edge_list, coordinates):
    '''
    Checks both whether a graph contains disconnected subgraphs
    and whether edges physically cross.

    Inputs
    edge_list: list of edges [[0,1], [2,3], ...]
    coordinates: list of node coordinates [[x0,y0], [x1,y1], ...]

    Returns True if the graph is valid.
    '''
    return is_connected(edge_list) and no_crossings(edge_list, coordinates)
