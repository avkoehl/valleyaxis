import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon, LineString
from skimage.graph import MCP_Geometric
import xarray as xr
from tqdm import tqdm
import networkx as nx
from shapelysmooth import taubin_smooth
from shapelysmooth import chaikin_smooth

from valleyaxis.utils.subgraphs import split_flowlines
from valleyaxis.utils.subgraphs import find_channel_heads_and_outlets
from valleyaxis.utils.strahler import compute_stream_order
from valleyaxis.utils.geom import remove_holes


def check_inputs(dem, flowlines, floor):
    # dem is rioxarray raster
    if not isinstance(dem, xr.DataArray):
        raise ValueError("dem must be a rioxarray raster")
    # flowlines is gpd.GeoDataFrame of linestrings
    if not isinstance(flowlines, gpd.GeoDataFrame):
        raise ValueError("flowlines must be a geopandas GeoDataFrame")

    if not all(flowlines.geometry.geom_type == "LineString"):
        raise ValueError("flowlines must be a GeoDataFrame of LineStrings")

    # floor is a single polygon
    if not isinstance(floor, Polygon):
        raise ValueError("floor must be a single Shapely Polygon")

    # dem and flowlines must have the same CRS
    if dem.rio.crs != flowlines.crs:
        raise ValueError("dem and flowlines must have the same CRS")


def valley_centerlines(
    dem,
    flowlines,
    floor,
    f1=1000,
    a=4.25,
    f2=3000,
    b=3.5,
    maxarea=None,
    smooth_output=True,
):
    check_inputs(dem, flowlines, floor)
    flowlines = split_flowlines(flowlines)
    floor = remove_holes(floor, maxarea=maxarea)

    results = []
    for network_id, lines in flowlines.groupby("network_id"):
        channel_nodes = find_channel_heads_and_outlets(lines)
        if sum(channel_nodes["type"] == "outflow") != 1:
            raise ValueError("There must be exactly one outlet point in the flowlines.")

        outlet = channel_nodes.loc[
            channel_nodes[channel_nodes["type"] == "outflow"].index[0], "geometry"
        ]
        inflow_points = channel_nodes.loc[channel_nodes["type"] == "inflow", "geometry"]
        centerlines = _centerlines(dem, inflow_points, outlet, floor, f1, a, f2, b)
        centerlines = compute_stream_order(centerlines)
        centerlines["network_id"] = network_id
        results.append(centerlines)

    centerlines = pd.concat(results, ignore_index=True)
    if smooth_output:
        centerlines["geometry"] = centerlines.geometry.apply(
            lambda x: chaikin_smooth(taubin_smooth(x))
        )

    return centerlines


def _centerlines(
    dem: xr.DataArray,
    inflow_points: gpd.GeoSeries,
    outlet_point: Point,
    floor: Polygon,
    f1: float = 1000,
    a: float = 4.25,
    f2: float = 3000,
    b: float = 3.5,
) -> gpd.GeoSeries:
    """
    Extract valley centerlines using a cost-distance approach.

    Parameters
    ----------
    dem : Union[xr.DataArray, str]
        Digital elevation model as xarray DataArray or path to GeoTIFF
    inflow_points : gpd.GeoSeries[Point]
        Series of inflow points
    outlet_point : Point
        Outlet point
    floor : Polygon
        Valley floor polygon as Shapely Polygon or GeoDataFrame
    f1 : float, optional
        Distance penalty factor, by default 1000
    a : float, optional
        Distance penalty exponent, by default 4.25
    f2 : float, optional
        Elevation penalty factor, by default 3000
    b : float, optional
        Elevation penalty exponent, by default 3.5

    Returns
    -------
    gpd.GeoSeries[LineString]
        Series of valley centerlines as Shapely LineString

    References
    ----------
    Kienholz et al. (2014) - https://tc.copernicus.org/articles/8/503/2014/
    """
    # Create penalty surface
    penalty = _create_penalty_surface(dem, floor, f1, a, f2, b)
    mcp = MCP_Geometric(penalty.data)

    # convert inflow points and outlet point to pixel coordinates
    transform = rasterio.transform.AffineTransformer(penalty.rio.transform())
    out_row, out_col = transform.rowcol(outlet_point.x, outlet_point.y)
    inlet_cells = []
    for point in inflow_points:
        row, col = transform.rowcol(point.x, point.y)
        inlet_cells.append((row, col))

    # create segments
    paths = trace_paths(mcp, inlet_cells, out_row, out_col)
    graph = paths_to_nxdigraph(paths)
    segments = graph_segments(graph)
    lines = segments_to_linestrings(segments, transform, dem.rio.crs)
    return lines


def trace_paths(mcp, inlet_cells, out_row, out_col):
    paths = []
    for row, col in tqdm(inlet_cells):
        costs, traceback = mcp.find_costs(
            starts=[[row, col]], ends=[[out_row, out_col]]
        )
        path = mcp.traceback([out_row, out_col])
        paths.append(path)
    sorted_paths = sorted(paths, key=len, reverse=True)
    return sorted_paths


def allocate_basins(mcp, start_cells):
    # start cells are the centerline points ?
    pass


def paths_to_nxdigraph(paths):
    G = nx.DiGraph()
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])

    # identify junctions, inflow points, and outlet points
    for node in G.nodes():
        # inflow nodes
        if G.in_degree(node) == 0 and G.out_degree(node) > 0:
            G.nodes[node]["inflow"] = True

        # junction
        if G.in_degree(node) > 1 and G.out_degree(node) > 0:
            G.nodes[node]["junction"] = True

        # outlet
        if G.in_degree(node) > 0 and G.out_degree(node) == 0:
            G.nodes[node]["outlet"] = True

    return G


def graph_segments(G):
    # flowline is from inflow to junction, or, junction to junction, or junction to outlet, or inflow to outlet
    # start nodes are all nodes with inflow is True or junction is True
    start_nodes = [
        n for n, d in G.nodes(data=True) if d.get("inflow") or d.get("junction")
    ]

    segments = []
    for start in start_nodes:
        if G.out_degree(start) == 0:
            continue

        # start new segment
        segment = [start]
        current = list(G.successors(start))[0]  # only one out edge

        while (current not in start_nodes) and (G.out_degree(current) > 0):
            segment.append(current)
            current = list(G.successors(current))[0]
        segment.append(current)  # add the last node (junction or outlet)
        segments.append(segment)
    return segments


def segments_to_linestrings(segments, transform, crs):
    records = []
    for i, segment in enumerate(segments):
        coords = []
        for pixel in segment:
            x, y = transform.xy(pixel[0], pixel[1])
            coords.append((x, y))
        line = LineString(coords)
        records.append(
            {
                "ID": i,
                "geometry": line,
            }
        )
    gdf = pd.DataFrame.from_records(records)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

    return gdf


def _create_penalty_surface(
    dem: xr.DataArray, floor: Polygon, f1: float, a: float, f2: float, b: float
) -> xr.DataArray:
    """
    Create penalty surface combining distance and elevation components.

    Parameters
    ----------
    dem : xr.DataArray
        Digital elevation model
    floor : Polygon
        Valley floor polygon
    f1, a : float
        Distance penalty parameters
    f2, b : float
        Elevation penalty parameters

    Returns
    -------
    xr.DataArray
        Combined penalty surface
    """
    # Clip DEM to valley floor
    clipped_dem = dem.rio.clip([floor])

    # Create mask of valid pixels
    polygon_mask = np.zeros_like(clipped_dem)
    polygon_mask[np.isfinite(clipped_dem)] = 1

    # Calculate distance component
    pixel_size = clipped_dem.rio.resolution()[0]
    distance = distance_transform_edt(polygon_mask) * pixel_size

    # Calculate distance penalty
    d_max = np.nanmax(distance)
    dist_penalty = (d_max - distance) / d_max * f1
    dist_penalty = np.power(dist_penalty, a)

    # Calculate elevation penalty
    z_min = np.nanmin(clipped_dem)
    z_max = np.nanmax(clipped_dem)
    elev_penalty = (clipped_dem - z_min) / (z_max - z_min) * f2
    elev_penalty = np.power(elev_penalty, b)

    # Combine penalties
    penalty = dist_penalty + elev_penalty

    # Fill NaN values with high cost
    penalty = penalty.fillna(-1)

    return penalty
