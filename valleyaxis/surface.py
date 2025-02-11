import geopandas as gpd
from numba import njit
import numpy as np
import xarray as xr
import rioxarray
from rasterio import features
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString, Point


class Surface:
    """
    Calculate geodesic (shortest) paths over a DEM surface constrained by a polygon.

    Attributes:
        dem: rioxarray DataArray representing the digital elevation model
        polygon: shapely Polygon defining the boundary constraint
        graph: scipy sparse matrix representing the surface graph
        transform: affine transform between pixel and world coordinates
        node_ids: numpy array mapping pixel positions to graph node IDs

    Methods:
        trace_path: find the shortest path between two points over the DEM surface as a LineString
    """

    def __init__(self, dem, polygon, enforce_uphill=False, enforce_downhill=False):
        """
        Initialize the Geodesic calculator.

        Args:
            dem: rioxarray DataArray of the digital elevation model
            polygon: shapely Polygon defining the boundary constraint
            enforce_uphill: bool, if True only allows uphill movement
            enforce_downhill: bool, if True only allows downhill movement
        """
        self.dem = dem
        self.polygon = polygon
        self.transform = dem.rio.transform()

        # Create the graph
        self.graph = create_surface_graph(
            dem=dem,
            polygon=polygon,
            enforce_uphill=enforce_uphill,
            enforce_downhill=enforce_downhill,
        )

        # Store node IDs for coordinate conversion
        self.node_ids = np.arange(dem.size).reshape(dem.shape)

    def nearest_points(self, input_points, target_points):
        """
        Find the nearest points on the surface between two sets of points.
        returns the nearest points on the surface for each input and target point and
        the path between them
        """
        # convert input_points and target_points to pixel coordinates
        input_px = [self._world_to_pixel(p) for p in input_points]
        target_px = [self._world_to_pixel(p) for p in target_points]
        input_nodes = [self.node_ids[px[0], px[1]] for px in input_px]
        target_nodes = [self.node_ids[px[0], px[1]] for px in target_px]

        # Calculate shortest path
        distances, predecessors = shortest_path(
            csgraph=self.graph,
            directed=True,
            indices=input_nodes,
            return_predecessors=True,
        )

        # extract distances to target node_ids
        target_dists = distances[:, target_nodes]

        # find the minimum distance and keep the indices
        min_input_idx, min_target_idx = divmod(target_dists.argmin(), len(target_nodes))
        min_distance = target_dists[min_input_idx, min_target_idx]

        if np.isinf(min_distance):
            return None, None, None
        else:
            # Reconstruct path
            path_nodes = []
            current_node = target_nodes[min_target_idx]
            while current_node != input_nodes[min_input_idx]:
                if current_node == -9999:
                    raise ValueError("No valid path exists between the points")
                path_nodes.append(current_node)
                # update current node
                current_node = predecessors[0, current_node]  # TODO: danger!
            path_nodes.append(input_nodes[min_input_idx])
            path_nodes.reverse()

            # Convert node IDs back to world coordinates
            path_coords = []
            for node in path_nodes:
                px_row, px_col = np.where(self.node_ids == node)
                world_point = self._pixel_to_world((px_row[0], px_col[0]))
                path_coords.append(world_point)

            path = LineString(path_coords)
            return input_points[min_input_idx], target_points[min_target_idx], path

    def trace_path(self, start_point: Point, end_point: Point) -> LineString:
        """
        Find the shortest path between two points over the DEM surface.

        Args:
            start_point: shapely Point in world coordinates
            end_point: shapely Point in world coordinates

        Returns:
            shapely LineString representing the path in world coordinates
        """
        # Convert world coordinates to pixel coordinates
        start_px = self._world_to_pixel(start_point)
        end_px = self._world_to_pixel(end_point)

        # Get node IDs for start and end points
        start_node = self.node_ids[start_px[0], start_px[1]]
        end_node = self.node_ids[end_px[0], end_px[1]]

        # Calculate shortest path
        _, predecessors = djikstra(
            csgraph=self.graph,
            directed=True,
            indices=start_node,
            return_predecessors=True,
        )

        # Reconstruct path
        path_nodes = []
        current_node = end_node
        while current_node != start_node:
            if current_node == -9999:  # No path found
                raise ValueError("No valid path exists between the points")
            path_nodes.append(current_node)
            current_node = predecessors[current_node]
        path_nodes.append(start_node)
        path_nodes.reverse()

        # Convert node IDs back to world coordinates
        path_coords = []
        for node in path_nodes:
            px_row, px_col = np.where(self.node_ids == node)
            world_point = self._pixel_to_world((px_row[0], px_col[0]))
            path_coords.append(world_point)

        return LineString(path_coords)

    def _world_to_pixel(self, point: Point) -> tuple:
        """
        Convert world coordinates to pixel coordinates.

        Args:
            point: shapely Point in world coordinates

        Returns:
            tuple (row, col) of pixel coordinates
        """
        col, row = ~self.transform * (point.x, point.y)
        col = int(round(col))
        row = int(round(row))
        return row, col

    def _pixel_to_world(self, pixel: tuple) -> tuple:
        """
        Convert pixel coordinates to world coordinates.

        Args:
            pixel: tuple (row, col) of pixel coordinates

        Returns:
            tuple (x, y) of world coordinates
        """
        x, y = self.transform * (pixel[1], pixel[0])
        return (x, y)


def create_surface_graph(dem, polygon, enforce_uphill=False, enforce_downhill=False):
    walls = rasterize_polygon_bounds(polygon, dem)
    graph = dem_to_graph(dem.data, walls.data, enforce_uphill, enforce_downhill)
    return graph


def dem_to_graph(dem, walls, enforce_uphill=False, enforce_downhill=False):
    data, ids = _create_graph_data_numba(dem, walls, enforce_uphill, enforce_downhill)
    graph = csr_matrix(data, shape=(ids.size, ids.size))
    return graph


def rasterize_polygon_bounds(polygon, dem):
    # Convert polygon to GeoDataFrame with same CRS as DEM
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=dem.rio.crs)
    boundary = gdf.geometry.boundary

    # Create empty raster matching DEM
    raster = xr.zeros_like(dem)

    # Rasterize the boundary
    shapes = [(geom, 1) for geom in boundary.geometry]
    features.rasterize(
        shapes=shapes,
        out=raster.data,
        transform=dem.rio.transform(),
        dtype=np.uint8,
        all_touched=True,
    )

    # Create rioxarray DataArray with same coordinates and attributes as DEM
    result = xr.DataArray(
        raster.data,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            "transform": dem.rio.transform(),
            "crs": dem.rio.crs,
            "res": dem.rio.resolution(),
        },
    )

    return result


@njit
def _create_graph_data_numba(dem, walls, enforce_uphill, enforce_downhill):
    nrows, ncols = dem.shape
    ids = np.arange(dem.size).reshape(dem.shape)
    row_inds = []
    col_inds = []
    data = []
    for row in range(nrows):
        for col in range(ncols):
            start = ids[row, col]

            if walls is not None:
                if walls[row, col]:
                    continue

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = row + dx
                    ny = col + dy
                    end = ids[nx, ny]

                    if walls is not None:
                        if walls[nx, ny]:
                            continue

                    if 0 <= nx < nrows and 0 <= ny < ncols:
                        cost = 1 if dx == 0 or dy == 0 else 1.41
                        cost *= dem[nx, ny] - dem[row, col]

                        if cost < 0 and enforce_uphill:
                            continue

                        if cost > 0 and enforce_downhill:
                            continue

                        data.append(np.abs(cost))
                        row_inds.append(start)
                        col_inds.append(end)

    return (data, (row_inds, col_inds)), ids
