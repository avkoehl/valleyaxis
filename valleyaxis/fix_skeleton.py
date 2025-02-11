import numpy as np
import networkx as nx
from networkx import Graph
from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point


def fix_skeleton(skeleton, surface):
    # Convert to single linestring if possible (helps with connectivity)
    skeleton = linemerge(skeleton)
    if isinstance(skeleton, LineString):
        skeleton = MultiLineString([skeleton])

    # Create graph
    G = mutliline_to_nxgraph(skeleton)

    # if no disconnected components, return original fix_skeleton
    if nx.is_connected(G):
        return skeleton
    else:
        connections = connect_components(G, surface)
        for _, _, _, line in connections:
            skeleton = skeleton.union(line)
        return skeleton


def mutliline_to_nxgraph(multiline):
    G = Graph()
    for line in multiline.geoms:
        coords = list(map(tuple, line.coords))
        G.add_edges_from(zip(coords[:-1], coords[1:]))
    return G


def nxgraph_to_multiline(G):
    lines = []
    for c in nx.connected_components(G):
        lines.append(LineString(c))
    return MultiLineString(lines)


def connect_components(graph, surface):
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    point_groups = []
    for subgraph in subgraphs:
        lines = nxgraph_to_multiline(subgraph)
        point_groups.append(multiline_to_points(lines))

    connections = connect_point_groups(point_groups, surface)
    return connections


def connect_point_groups(point_groups, surface):
    """
    Connect multiple groups of points using a minimum spanning tree approach.
    Each connection is the shortest path between any two points in different groups.

    Args:
        point_groups: List of lists, where each inner list contains shapely Points

    Returns:
        list: List of tuples (point1, point2, distance) representing the connections
    """

    # Initialize structures for Kruskal's algorithm
    class UnionFind:
        def __init__(self):
            self.parent = {}

        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                return x
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px != py:
                self.parent[px] = py
                return True
            return False

    # Find all pairwise distances between groups
    edges = []
    for i in range(len(point_groups)):
        for j in range(i + 1, len(point_groups)):
            input_points = point_groups[i]
            target_points = point_groups[j]

            try:
                input_point, target_point, ls = surface.nearest_points(
                    input_points, target_points
                )
                edges.append((ls.length, line, i, j, input_point, target_point))
            except ValueError:
                continue

    if not edges:
        raise ValueError("No valid connections found between point groups")

    # Sort edges by distance
    edges.sort()

    # Run Kruskal's algorithm
    uf = UnionFind()
    connections = []
    groups_connected = set()

    for distance, line, group1, group2, point1, point2 in edges:
        if uf.union(group1, group2):
            connections.append((point1, point2, distance, line))
            groups_connected.add(group1)
            groups_connected.add(group2)

            # Check if all groups are connected
            if len(groups_connected) == len(point_groups):
                break

    if len(groups_connected) < len(point_groups):
        raise ValueError("Unable to connect all point groups")

    return connections


def multiline_to_points(multiline, spacing=None):
    """
    Convert a MultiLineString or LineString to a set of Points.

    Args:
        multiline: shapely MultiLineString or LineString
        spacing: Optional float, if provided, interpolates points along lines at this spacing
                If None, only returns vertices

    Returns:
        list of shapely Points
    """
    # Convert single LineString to MultiLineString if needed
    if isinstance(multiline, LineString):
        multiline = MultiLineString([multiline])

    points = []

    for line in multiline.geoms:
        # Add vertices
        coords = list(line.coords)
        points.extend([Point(x, y) for x, y in coords])

        # If spacing is provided, interpolate points along the line
        if spacing is not None:
            # Get total length of line
            length = line.length

            # Calculate number of points to interpolate
            num_points = int(length / spacing)

            if num_points > 0:
                # Generate evenly spaced distances along the line
                distances = np.linspace(0, length, num_points + 2)[1:-1]

                # Interpolate points at these distances
                for distance in distances:
                    point = line.interpolate(distance)
                    points.append(Point(point.x, point.y))

    # Remove duplicates by converting to set of coordinates and back to Points
    unique_coords = set((p.x, p.y) for p in points)
    unique_points = [Point(x, y) for x, y in unique_coords]

    return unique_points
