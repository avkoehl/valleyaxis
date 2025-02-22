# https://tc.copernicus.org/articles/8/503/2014/tc-8-503-2014.pdf

# penalty = [(max(d) - d) / max(d)) * f1 ] ^ a + [(z - min(z)) / (max(z) - min(z)) * f2 ] ^ b
import rioxarray as rxr
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

from valleyaxis.channel_nodes import find_channel_heads_and_outlets


dem = rxr.open_rasterio("./data/conditioned_dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/flowlines.shp")
floor = gpd.read_file("./data/floor.shp").geometry.iloc[0]
points = find_channel_heads_and_outlets(flowlines)


def create_penalty_image(dem, floor, f1, a, f2, b):
    clipped_dem = dem.rio.clip([floor])
    polygon_mask = np.zeros_like(clipped_dem)
    polygon_mask[np.isfinite(clipped_dem)] = 1

    pixel_size = clipped_dem.rio.resolution()[0]
    distance = distance_transform_edt(polygon_mask) * pixel_size

    d_max = np.nanmax(distance)
    dist_penalty = (d_max - distance) / d_max * f1
    dist_penalty = np.power(dist_penalty, a)

    z_min = np.nanmin(clipped_dem)
    z_max = np.nanmax(clipped_dem)
    elev_penalty = (clipped_dem - z_min) / (z_max - z_min) * f2
    elev_penalty = np.power(elev_penalty, b)

    penalty = dist_penalty + elev_penalty
    return penalty


penalty = create_penalty_image(dem, floor, 1000, 4.25, 3000, 3.5)
penalty = penalty.fillna(-1)


mcp = MCP_Geometric(penalty.data)


results = penalty.copy()
results.data = np.zeros_like(penalty, dtype=np.int32)

out_point = points[points["type"] == "outflow"].geometry.values[0]
out_col, out_row = ~penalty.rio.transform() * (out_point.x, out_point.y)
out_col = int(out_col)
out_row = int(out_row)

for ind, point in points.iterrows():
    if point["type"] == "outflow":
        continue
    print(point["flowline_id"])
    col, row = ~penalty.rio.transform() * (point.geometry.x, point.geometry.y)
    col = int(col)
    row = int(row)
    costs, traceback = mcp.find_costs(starts=[[row, col]], ends=[[out_row, out_col]])
    path = mcp.traceback([out_row, out_col])
    for p in path:
        results.data[p[0], p[1]] = 1
