from shapely.ops import nearest_points


def extend_skeleton(skeleton, surface, channel_nodes):
    # skeleton is a MultiLineString
    # geodesic is Geodesic object
    # channel_nodes is a GeoDataFrame with geometry column containing Point objects

    for point in channel_nodes.geometry:
        nearest_point_on_skeleton = nearest_points(skeleton, point)[0]
        path = surface.trace_path(point, nearest_point_on_skeleton)
        if path is not None:
            skeleton = skeleton.union(path)

    return skeleton
