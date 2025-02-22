import matplotlib.pyplot as plt

import geopandas as gpd
import rioxarray as rxr

from valleyaxis.channel_nodes import find_channel_heads_and_outlets
from valleyaxis import valley_centerlines


# load a sample dem, valley floor, and flowlines
dem = rxr.open_rasterio("./data/conditioned_dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/flowlines.shp")
floor = gpd.read_file("./data/floor.shp").geometry.iloc[0]

channel_nodes = find_channel_heads_and_outlets(flowlines)
outlet = channel_nodes.loc[
    channel_nodes[channel_nodes["type"] == "outflow"].index[0], "geometry"
]
inlets = channel_nodes.loc[channel_nodes["type"] == "inflow", "geometry"]

centerline = valley_centerlines(dem, inlets, outlet, floor)

fig, ax = plt.subplots()
centerline.plot(ax=ax)
gpd.GeoSeries([floor], crs=dem.rio.crs).plot(ax=ax, facecolor="none", edgecolor="black")
plt.show()
