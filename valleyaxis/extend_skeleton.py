from tqdm import tqdm

from valleyaxis.fix_skeleton import multiline_to_points


def extend_skeleton(skeleton, surface, channel_nodes):
    # skeleton is a MultiLineString
    # geodesic is Geodesic object
    # channel_nodes is a GeoDataFrame with geometry column containing Point objects

    for point in tqdm(channel_nodes.geometry):
        skeleton_points = multiline_to_points(skeleton, spacing=10)
        _, _, path = surface.nearest_points([point], skeleton_points)
        if path is not None:
            skeleton = skeleton.union(path)

    return skeleton
