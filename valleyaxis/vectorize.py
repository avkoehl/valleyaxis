# binary skeleton -> gpd.Geoseries of segments binary floors -> gpd.GeoDataFrame of floors polygons
import numba
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString
from skimage.morphology import skeletonize


def vectorize_skeleton(skeleton, outlet):
    transform = rasterio.transform.AffineTransformer(skeleton.rio.transform())
    outlet_row, outlet_col = transform.rowcol(outlet.x, outlet.y)
    skeleton.data = skeletonize(skeleton.data)
    labeled, segments = _label_skeleton_numba(skeleton.data, outlet_row, outlet_col)
    # need better definition of junction

    lines = []
    for segment in segments:
        coords = [transform.xy(node[0], node[1]) for node in segment]
        linestring = LineString(coords)
        lines.append(linestring)
    gdf = gpd.GeoDataFrame(geometry=lines, crs=skeleton.rio.crs)
    return gdf, labeled


@numba.njit
def _label_skeleton_numba(skeleton, outlet_row, outlet_col):
    labeled = np.zeros(skeleton.shape, dtype=np.int32)

    # get all start nodes
    start_nodes, junctions = _get_nodes_numba(skeleton)
    start_nodes = [node for node in start_nodes if node != (outlet_row, outlet_col)]

    visited = np.zeros(skeleton.shape, dtype=np.bool)
    segments = []
    label = 1
    for start_node in start_nodes:
        segment = [start_node]
        visited[start_node] = True
        current_node = start_node
        labeled[start_node] = label

        while True:
            neighbors = _get_neighbors_numba(skeleton, current_node[0], current_node[1])
            # remove visited neighbors in numba friendly way
            neighbors = [n for n in neighbors if not visited[n]]
            if len(neighbors) == 0:
                break
            if len(neighbors) == 1:
                next_node = neighbors[0]
                if visited[next_node]:
                    break
                if next_node in junctions:
                    # add the junction to start nodes
                    # and remove it from the list of junctions
                    junctions.remove(next_node)
                    start_nodes.append(next_node)
                    break
                segment.append(next_node)
                visited[next_node] = True
                labeled[next_node] = label
                current_node = next_node
            else:  # somehow at a junction
                break

        # add the segment to the list
        if len(segment) > 1:
            segments.append(segment)
        label += 1
    return labeled, segments


@numba.njit
def _get_neighbors_numba(array, row, col):
    """
    Get the neighbors of a given node in the skeleton.
    """
    neighbors = []
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if (
                (r, c) != (row, col)
                and 0 <= r < array.shape[0]
                and 0 <= c < array.shape[1]
            ):
                if array[r, c] == 1:
                    neighbors.append((r, c))
    return neighbors


@numba.njit
def _get_nodes_numba(skeleton_array):
    start_nodes = []
    junctions = []
    # get start nodes, end nodes, and junctions
    for row in range(skeleton_array.shape[0]):
        for col in range(skeleton_array.shape[1]):
            if skeleton_array[row, col] == 1:
                neighbors = _get_neighbors_numba(skeleton_array, row, col)
                if len(neighbors) == 1:
                    start_nodes.append((row, col))
                if len(neighbors) > 2:
                    junctions.append((row, col))
    return start_nodes, junctions
