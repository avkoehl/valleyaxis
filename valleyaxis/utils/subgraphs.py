import networkx as nx
from shapely.geometry import Point
import geopandas as gpd

from valleyaxis.utils.network import lines_to_network


def split_flowlines(flowlines):
    graph = lines_to_network(flowlines)

    outlets = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    if len(outlets) == 1:
        return flowlines

    flowlines["network_id"] = None
    for i, outlet in enumerate(outlets):
        upstream = nx.ancestors(graph, outlet)
        upstream.add(outlet)
        subgraph = graph.subgraph(upstream)
        streams = list(
            set(data["streamID"] for u, v, data in subgraph.edges(data=True))
        )
        flowlines.loc[streams, "network_id"] = i
    return flowlines


def find_channel_heads_and_outlets(flowlines_gdf):
    """
    Find channel head points and outlet points from flowlines GeoDataFrame.
    assumes within each flowline the coordinates are ordered from upstream to downstream
    """
    # Get all endpoints
    endpoints = []
    startpoints = []

    for line in flowlines_gdf.geometry:
        coords = list(line.coords)
        endpoints.append(tuple(coords[-1]))  # downstream end
        startpoints.append(tuple(coords[0]))  # upstream end

    startpoints_set = set(startpoints)
    endpoints_set = set(endpoints)

    # Channel heads are startpoints that aren't endpoints of other lines
    # outlet points are endpoints that aren't startpoints of other lines
    results = []
    for ind, line in flowlines_gdf.iterrows():
        start = tuple(line.geometry.coords[0])
        end = tuple(line.geometry.coords[-1])
        if start not in endpoints_set:
            result = {
                "geometry": Point(start),
                "type": "inflow",
                "flowline_id": ind,
            }
            results.append(result)
            continue
        if end not in startpoints_set:
            result = {
                "geometry": Point(end),
                "type": "outflow",
                "flowline_id": ind,
            }
            results.append(result)
    results = gpd.GeoDataFrame(results, crs=flowlines_gdf.crs)
    return results
