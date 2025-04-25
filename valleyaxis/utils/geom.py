import numpy as np
import rasterio
from shapely.geometry import Polygon
import geopandas as gpd
from warnings import warn as warning


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
                exterior = shape["coordinates"][0]
                if len(shape["coordinates"]) > 1:
                    interior = shape["coordinates"][1:]
                else:
                    interior = []

                polygons.append(Polygon(exterior, interior))

    df = gpd.GeoDataFrame(polygons, columns=["geometry"], crs=crs)
    polygon = df.union_all().buffer(0.01)
    if isinstance(polygon, Polygon):
        return polygon
    else:
        # if returns multipolygon, explode it and return list of polygons
        if polygon.geom_type == "MultiPolygon":
            polygons = [poly for poly in polygon.geoms if poly.is_valid]
            if len(polygons) == 1:
                return polygons[0]
            else:
                warning("More than one polygon found, returning list of polygons")
                return polygons


def remove_holes(polygon, maxarea=None):
    if maxarea is None:
        return Polygon(polygon.exterior.coords)

    holes = [hole for hole in polygon.interiors if hole.area <= maxarea]
    return Polygon(polygon.exterior.coords, holes=holes)
