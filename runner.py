import geopandas as gpd
import rioxarray as rxr
from valleyaxis.channel_nodes import find_channel_heads_and_outlets
from valleyaxis import valley_centerlines
from valleyaxis.vectorize import vectorize_skeleton

# Load input data
dem = rxr.open_rasterio("./sample_data/1805000202-dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./sample_data/1805000202-flowlines.shp")
floor = gpd.read_file("./sample_data/floor.shp").geometry[0]

# Find channel heads and outlets from flowlines
channel_nodes = find_channel_heads_and_outlets(flowlines)
outlet = channel_nodes.loc[
    channel_nodes[channel_nodes["type"] == "outflow"].index[0], "geometry"
]
inlets = channel_nodes.loc[channel_nodes["type"] == "inflow", "geometry"]

# Extract centerlines
skeleton = valley_centerlines(dem, inlets, outlet, floor)
df, labeled = vectorize_skeleton(skeleton, outlet)
