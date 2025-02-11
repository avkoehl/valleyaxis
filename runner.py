import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd

from valleyaxis.surface import Surface
from valleyaxis.voronoi_skeleton import voronoi_skeleton
from valleyaxis.channel_nodes import find_channel_heads_and_outlets
from valleyaxis.fix_skeleton import fix_skeleton
from valleyaxis.extend_skeleton import extend_skeleton

# load data:
# dem raster
# flowlines geodataframe with id column
# valley floor polygon

# setup config

dem = rxr.open_rasterio("./data/conditioned_dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/flowlines.shp")
floor = gpd.read_file("./data/floor.shp").geometry.iloc[0]

surface = Surface(dem, floor)
channel_nodes = find_channel_heads_and_outlets(flowlines)


# add simply params
skeleton = voronoi_skeleton(
    floor,
    num_points=5000,
    simplify_polygon=5,
    simplify_skeleton=1,
)

# fixed_skeleton = fix_skeleton(skeleton, surface)

extended = extend_skeleton(skeleton, surface, channel_nodes)

# pruned = prune_skeleton(extended, channel_nodes)

fig, ax = plt.subplots()
gpd.GeoSeries([floor], crs=dem.rio.crs).plot(ax=ax, facecolor="none", edgecolor="black")
gpd.GeoSeries(extended, crs=dem.rio.crs).plot(ax=ax, color="blue")
gpd.GeoSeries(skeleton, crs=dem.rio.crs).plot(ax=ax, color="red")
# gpd.GeoSeries(fixed_skeleton, crs=dem.rio.crs).plot(ax=ax, color="green")
plt.show()
