import numpy as np
import rasterio
from shapely.geometry import Polygon
import geopandas as gpd


def floor_raster_to_polygon(raster_file):
    with rasterio.open(raster_file) as src:
        image = src.read(1)
        crs = src.crs

        # assume anywhere value is not zero or nan is a polygon
        image = np.where(image != 0, 1, 0)
        image = image.astype(np.uint8)

        transform = src.transform
        polygons = []
        for shape, value in rasterio.features.shapes(image, transform=transform):
            if value == 1:
                polygons.append(Polygon(shape["coordinates"][0]))

    df = gpd.GeoDataFrame(polygons, columns=["geometry"], crs=crs)
    polygon = df.union_all().buffer(0.01)
    if isinstance(polygon, Polygon):
        return polygon
    else:
        raise ValueError("Unable to create a single polygon from raster data.")


def remove_holes(polygon, maxarea=None):
    if maxarea is None:
        return Polygon(polygon.exterior.coords)

    holes = [hole for hole in polygon.interiors if hole.area <= maxarea]
    return Polygon(polygon.exterior.coords, holes=holes)
