import numpy as np
from scipy.spatial import Voronoi
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.geometry import LineString
from tqdm import tqdm


def voronoi_skeleton(polygon, num_points, simplify_polygon, simplify_skeleton):
    """
    Generate a Voronoi skeleton (medial axis) for a polygon.
    """
    # Sample points along the boundary
    polygon = polygon.simplify(simplify_polygon)
    points = generate_boundary_points(polygon, num_points)
    vor = Voronoi(points)

    # Extract Voronoi lines
    lines = []
    print("Extracting Voronoi lines...")
    for p1, p2 in tqdm(vor.ridge_vertices):
        if p1 >= 0 and p2 >= 0:  # Skip infinite ridges
            line = LineString([vor.vertices[p1], vor.vertices[p2]])
            if polygon.contains(line):
                lines.append(line)

    # Merge lines and clean up
    skeleton = unary_union(lines)
    skeleton = skeleton.intersection(polygon)
    skeleton = skeleton.simplify(simplify_skeleton)

    return skeleton


def generate_boundary_points(polygon: Polygon, num_points: int) -> np.ndarray:
    """
    Generate equally spaced points along polygon boundary.

    Args:
        polygon: Shapely Polygon object
        num_points: Number of points to generate

    Returns:
        numpy array of points with shape (n, 2)
    """
    # Get the boundary as a LineString
    boundary = polygon.exterior

    # Calculate total length and spacing
    total_length = boundary.length
    spacing = total_length / num_points

    # Generate points at equal distances
    points = []
    for i in range(num_points):
        # Calculate distance along the boundary for this point
        distance = i * spacing

        # Use linear referencing to get point at this distance
        point = boundary.interpolate(distance)
        points.append((point.x, point.y))

    return np.array(points)
