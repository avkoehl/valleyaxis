import networkx as nx
from networkx import Graph
from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.geometry import MultiLineString

# TODO: consider not using networkx for this, use shapely 


def fix_skeleton(skeleton, surface):
    # Convert to single linestring if possible (helps with connectivity)
    skeleton = linemerge(skeleton)
    if isinstance(skeleton, LineString):
        skeleton = MultiLineString([skeleton])

    # Create graph
    G = Graph()
    for line in skeleton.geoms:
        coords = list(map(tuple, line.coords))
        G.add_edges_from(zip(coords[:-1], coords[1:]))

    # if no disconnected components, return original fix_skeleton
    if nx.is_connected(G):
        return skeleton
    else:
        reconnect_disconnected(G, surface)


def recconnect_disconnected(graph, surface):
    # convert to graph
    # see if there are any disconnected components
    # if there are, find the shortest geodesic path between them
    # connect them and return the new skeleton
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    # get centroids of each subgraph
    # for each disconnected, find the closest other subgraph based on centroid distance

    # use geodesic to connect 
    # repeat until fully connected
    for g in subgraphs:
        # find the closest 
    pass
