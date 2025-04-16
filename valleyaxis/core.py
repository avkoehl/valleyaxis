import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon
from skimage.graph import MCP_Geometric
import xarray as xr
from tqdm import tqdm
import numba


def valley_centerlines(
    dem: xr.DataArray,
    inflow_points: gpd.GeoSeries,
    outlet_point: Point,
    floor: Polygon,
    f1: float = 1000,
    a: float = 4.25,
    f2: float = 3000,
    b: float = 3.5,
) -> np.ndarray:
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
    np.ndarray
        Binary array where 1 indicates centerline pixels

    References
    ----------
    Kienholz et al. (2014) - https://tc.copernicus.org/articles/8/503/2014/
    """
    # Create penalty surface
    penalty = _create_penalty_surface(dem, floor, f1, a, f2, b)

    # Initialize MCP calculator
    mcp = MCP_Geometric(penalty.data)

    # Initialize results array
    results = penalty.copy()
    results.data = np.zeros_like(penalty.data, dtype=np.int32)

    # Convert outlet coordinates to array indices
    transform = rasterio.transform.AffineTransformer(penalty.rio.transform())
    out_row, out_col = transform.rowcol(outlet_point.x, outlet_point.y)

    # getpaths from each inflow point to outlet point
    paths = []
    inflow_nodes = []
    for inflow in tqdm(inflow_points):
        row, col = transform.rowcol(inflow.x, inflow.y)
        inflow_nodes.append((row, col))
        costs, traceback = mcp.find_costs(
            starts=[[row, col]], ends=[[out_row, out_col]]
        )
        path = mcp.traceback([out_row, out_col])
        paths.append(path)

    # now need to trace the paths from the inflow to the outlet, mark when passing a junction
    # mark edge between inflow -> junction and and between any junctions and with outlet
    sorted_paths = sorted(paths, key=len, reverse=True)
    j = find_junctions(sorted_paths, penalty.shape)
    # give each junction a unique id
    junc_id = 1
    for junc in j:
        results[junc] = junc_id
        junc_id += 1

    # create graph from the paths
    # from each inflow AND junction, find path to next junction or outlet
    # that gets labeled after the junction or inflow node that starts it
    # make a linestring
    flattend_paths = [elem]

    for path in paths:
        for pixel in path:
            if pixel in inflow_nodes:
                results[pixel] = 2
            if pixel in junctions:
                results[pixel] = 3
            if pixel == (out_row, out_col):
                results[pixel] = 4
            else:
                results[pixel] = 1
    return results


def label_skeleton(sorted_paths, shape):
    skeleton = np.zeros(shape, dtype=np.int32)
    for path in sorted_paths:
        for pixel in path:
            if skeleton[pixel] == 0:
                skeleton[pixel] = 1
    return skeleton


def find_junctions(sorted_paths, shape):
    visited = np.zeros(shape, dtype=np.int32)
    junctions = []
    for path in sorted_paths:
        count = 0
        for pixel in path:
            count += 1
            if visited[pixel] == 0:
                visited[pixel] = 1
            else:
                junctions.append(pixel)
                break
    return junctions


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
