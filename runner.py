import geopandas as gpd
import rioxarray as rxr
from valleyaxis.core import valley_centerlines

# Load input data
dem = rxr.open_rasterio("./sample_data/1805000202-dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./sample_data/1805000202-flowlines.shp")
floor = gpd.read_file("./sample_data/floor.shp").geometry[0].buffer(0.01)

centerlines = valley_centerlines(dem, flowlines, floor)
