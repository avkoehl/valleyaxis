from shapely.geometry import Point
import geopandas as gpd


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
    for _, line in flowlines_gdf.iterrows():
        start = tuple(line.geometry.coords[0])
        end = tuple(line.geometry.coords[-1])
        if start not in endpoints_set:
            result = {
                "geometry": Point(start),
                "type": "inflow",
                "flowline_id": line["STRM_VAL"],
            }
            results.append(result)
            continue
        if end not in startpoints_set:
            result = {
                "geometry": Point(end),
                "type": "outflow",
                "flowline_id": line["STRM_VAL"],
            }
            results.append(result)

    return gpd.GeoDataFrame(results, crs=flowlines_gdf.crs)
