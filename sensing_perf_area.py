import warnings
from time import time
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import quadpy
import scipy.integrate
import scipy.spatial
from scipy.spatial import Voronoi

import LazyVinh.convex_polygon_intersection as LazyVinh

warnings.filterwarnings('error', '.*extremely bad integrand behavior.*')

def f(point1: np.ndarray, point2: np.ndarray) -> float:
    R = 0.45
    dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2).flat[0] # np.sqrt returns np.ndarray
    if dist <= R: return 1
    else: return 0

def phi(point: np.ndarray) -> float:
    CENTER = [[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]]
    result = 0
    for c in CENTER:
        result = result + 5 * np.exp(-6 * ((point[0] - c[0])**2 + (point[1] - c[1])**2))
    return result.flat[0] # np.exp changed it into ndarray

def is_polygon_ccw(polygon : np.ndarray) -> bool:
    """
    Args:
        polygon: a convex polygon in either CW or CCW order;
            shape of ndarray is (npoints, 2)
    
    Returns:
        a boolean value whether polygon is in CCW order or not
    """

    # only need to check three consecutive points
    if len(polygon) < 3:
        return True
    else:
        a = np.subtract(polygon[1], polygon[0])
        b = np.subtract(polygon[2], polygon[0])
        return np.cross(a, b) > 0

def is_point_q_closer_to_p_than_r(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """
    Args:
        p: the "pivot" point
        q, r: point to be compared
            p, q, and r's shape of ndarray is (2,)
    Returns:
        True if distance(p, q) is strictly smaller than distance(p, r)
    """

    return np.abs(np.linalg.norm(q - p)) < np.abs(np.linalg.norm(r - p))

def get_vector_angle(vector: np.ndarray) -> float:
    """
    Args:
        vector: nonzero 2d vector
            shape of ndarray is (2,)
    Returns:
        a number in range of 0 to 2pi inclusive
            for example, vector(1, 0) returns 0 and vector(-1, 0) returns pi
    """

    length = np.abs(np.linalg.norm(vector))
    assert(length > 0)
    vector = vector / length
    angle = np.arccos(np.clip(np.dot(vector, np.array([1, 0])), -1.0, 1.0)).flat[0] # np.arccos returns np.ndarray
    if vector[1] < 0:
        angle = 2 * np.pi - angle
    return angle

def get_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Get the smaller angle between vector v1 and v2
    Args:
        v1, v2: nonzero 2d vector
            shape of ndarray is (2,)
    
    Returns:
        the smaller angle between v1 and v2
    """
    
    length1 = abs(np.linalg.norm(v1))
    length2 = abs(np.linalg.norm(v2))
    assert(length1 > 0 and length2 > 0)
    v1 = v1 / length1
    v2 = v2 / length2
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)).flat[0] # np.arccos returns np.ndarray

def get_rotated_vector(vector: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation, vector)

def get_intersection_line_segments(segment1, segment2) -> Optional[np.ndarray]:
    """
    Args:
        segment1, segment2: each defined as (endpoint1, endpoint2)
            shape of ndarray is (2, 2)
    
    Returns:
        a point if segment intersects; None otherwise

    From: https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
    """

    x1, y1 = segment1[0]
    x2, y2 = segment1[1]
    x3, y3 = segment2[0]
    x4, y4 = segment2[1]
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return np.array([x,y])

# TODO merge p1, p2 argument to segment
def get_intersection_circle_and_line_segment(circle_center: np.ndarray,
                                             circle_radius: np.ndarray,
                                             p1: np.ndarray,
                                             p2: np.ndarray,
                                             full_line: bool = True,
                                             tangent_tol: float = 1e-9) -> List[Tuple[float, float]]: # type: ignore
    """
    Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    Args:
        circle_center: The (x, y) location of the circle center
        circle_radius: The radius of the circle
        p1: The (x, y) location of the first point of the segment
        p2: The (x, y) location of the second point of the segment
        full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    
    Returns:
        a list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    From: https://stackoverflow.com/a/59582674/12607236
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = p1, p2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def get_intersection_polygon(polygon1: np.ndarray, polygon2: np.ndarray) -> np.ndarray:
    """
    Args:
        polygon1, polygon2: convex polygon in CCW order
            shape of ndarray is (npoints, 2)
    
    Returns:
        points of the polygon in CCW order.
            shape of ndarray is (npoints, 2)
    """

    intersection = LazyVinh.intersect(polygon1, polygon2)
    return np.array(intersection)

def get_integral_over_convex_polygon(polygon: np.ndarray, function: Callable[[np.ndarray], float], function_degree: int = 12) -> float:
    """
    Args:
        polygon: a convex polygon in CCW order
            shape of ndarray is (npoints, 2)
        function: a function to be integrated over the polygon
            shape of arg ndarray is (2,)
        function_degree: the highest polynomial degree of the function, used to determine what integration algorithm will be used
    
    Returns:
        the resulting integration
    """
    
    # quadpy needs a function to accept array of testValues. this function will help that
    def function_helper(test_values):
        result = np.zeros(test_values.shape[1])
        for i in range(test_values.shape[1]):
            result[i] = function(test_values[:,i])
        return result
    
    scheme = quadpy.t2.get_good_scheme(function_degree)
    result = 0
    if len(polygon) >= 3:
        for i in range(1, len(polygon) - 1):
            result = result + scheme.integrate(lambda x: function_helper(x), [polygon[0], polygon[i], polygon[i+1]])
    return result

def get_integral_over_arc(arc_center: np.ndarray, arc_radius: float, arc_angle_from: float, arc_angle_to: float, function: Callable[[np.ndarray], float]) -> float:
    """
    Args:
        arc_center, arc_radius, arc_angle_from, arc_angle_to: define the properties of the arc
            arc_center shape of ndarray is (2,)
            arc_angle... is in radian
        function: the function to be integrated
    
    Returns:
        the resulting integral
    """

    def function_helper(angle):
        direction = get_rotated_vector(np.array([arc_radius, 0]), angle)
        return function(arc_center + direction)
    return scipy.integrate.quad(function_helper, arc_angle_from, arc_angle_to, limit=100)[0] # quad returns tuple (result, error)

# TODO create a normalize() function instead
def get_unit_normal_at_point_in_arc(arc_center: np.ndarray, arc_point: np.ndarray) -> np.ndarray:
    """
    Args:
        arc_center: the center of the arc
        arc_point: the point in question

    Returns:
        a unit normal at arc_point in an arc with arc_center as its center
    """

    vector = arc_point - arc_center
    normal = vector / np.linalg.norm(vector)
    return normal

def get_near_points(points: np.ndarray, index_point: int, radius: float) -> np.ndarray:
    """
    Args:
        points: the list of points
            shape of ndarray is (npoints, 2)
        indexPoint: the index of the point inside points of what we are interested in
        radius: the maximum allowed distance between two points to be considered as "near"
    
    Returns:
        list of points that is close to the indexPoint, without the point itself
            shape of ndarray is (n_near_points, 2)
    """

    result = list()
    for i in range(len(points)):
        if i != index_point:
            distance = np.abs(np.linalg.norm(points[i] - points[index_point]))
            if distance <= radius:
                result.append(points[i])
    return np.array(result)

def get_voronoi_partitions(points: np.ndarray, border_polygon: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        points: set of point for the voronoi partition
            shape of ndarray is (npoints, 2)
        border_polygon: "border" of the voronoi partition in CCW order
            shape of ndarray is (n_border_points, 2)
    
    Returns:
        a list of polygons, where the ith polygon is the voronoi partition of the ith point in CCW order
            shape of ndarray is (n_polygon_points, 2)
    """

    BORDER_COORD = 1e6 # for now, let's assume 1e6 is big enough
    # add 4 border points so that the voronoi partition shape will be limited
    new_points = np.append(points, [[-BORDER_COORD, 0], [BORDER_COORD, 0], [0, -BORDER_COORD], [0, BORDER_COORD]], 0)
    
    voronoi = Voronoi(new_points)
    voronoi_partition = []
    for i in range(len(points)): # discarding the 4 border points
        region_index = voronoi.point_region[i]
        region_points = []
        for j in voronoi.regions[region_index]:
            assert(j != -1), "Region " + new_points[i] + " is not bounded"
            region_points.append(voronoi.vertices[j])
        if not(is_polygon_ccw(region_points)): # scipy's voronoi region order is either CW or CCW, so we need to fix it to become CCW
            region_points.reverse()
        region_points = np.array(region_points)
        voronoi_partition.append(get_intersection_polygon(region_points, border_polygon))
    
    return voronoi_partition

def get_limited_voronoi_partition(point: np.ndarray,
                                  near_points: np.ndarray,
                                  half_radius: float,
                                  border_polygon: np.ndarray,
                                  angle_error_tolerance: float = 1e-8) -> List[Union[Tuple[int, np.ndarray, np.ndarray], Tuple[int, float, float]]]:
    """
    Args:
        point: a single point
            shape of ndarray is (2,)
        near_points: points which are not further than radius from point
        half_radius: the radius of the ball. in other words, half of the maximum allowed distance between two points to be considered as neighbor
        border_polygon: "border" of the voronoi partition in CCW order
            shape of ndarray is (n_border_points, 2)
    
    Returns:
        list of line segments and arcs that describe this point's limited voronoi partition
        a line segment and arc is differentiated by the first value of the list, where line segment is 0 and arc is 1
            a line segment entry is defined by [0, endPoint1, endPoint2]
            an arc entry is defined by [1, angleStart, angleEnd], where angleStart <= angleEnd
    """

    point = np.copy(point) # to avoid modifying client's variable
    neighbor_points = list()
    if len(near_points) < 2:
        for near_point in near_points:
            neighbor_points.append(np.copy(near_point))
    else:
        all_points = np.append([point], near_points, 0)
        delaunay = scipy.spatial.Delaunay(all_points)
        neighbor_points_index = set()
        assert(point[0] == delaunay.points[0][0] and point[1] == delaunay.points[0][1])
        for triangle in delaunay.simplices:
            if 0 in triangle:
                for index in triangle:
                    if index != 0:
                        neighbor_points_index.add(index)
        neighbor_points = [delaunay.points[index] for index in neighbor_points_index]

    uncut_segments = list() # the "voronoi" segments created by these neighbor points. some of these segments might overlap each other
    for neighbor_point in neighbor_points:
        # finding each segment's endpoints at the circle(point, radius)
        direction = neighbor_point - point
        direction_angle = get_vector_angle(direction)
        distance_to_mid_point = np.abs(np.linalg.norm(direction / 2))
        delta_angle = np.arccos(distance_to_mid_point / half_radius).flat[0]
        # endPoint1 to endPoint2 is in CCW order
        endpoint1 = point + get_rotated_vector(np.array([half_radius, 0]), direction_angle - delta_angle)
        endpoint2 = point + get_rotated_vector(np.array([half_radius, 0]), direction_angle + delta_angle)
        uncut_segments.append([endpoint1, endpoint2])
    for i in range(len(border_polygon)): # for border polygon
        endpoint1 = np.copy(border_polygon[i-1])
        endpoint2 = np.copy(border_polygon[i])
        # we regard this "line segment" as an "infinity line", because all uncutSegments have their endPoints at the circle
        intersections = get_intersection_circle_and_line_segment(point, half_radius, endpoint1, endpoint2)
        if len(intersections) == 2:
            uncut_segments.append([intersections[0], intersections[1]])
        else:
            assert(len(intersections) == 0 or len(intersections) == 1), "Unexpected number of intersections"
    uncut_segments = np.array(uncut_segments)

    cut_segments = list()
    cut_segments_angle_range = list() # there are no "jumping" range
    for segment in uncut_segments:
        endpoint1 = segment[0]
        endpoint2 = segment[1]
        should_insert = True
        for other_segment in uncut_segments:
            if not np.array_equal(segment, other_segment):
                intersection = get_intersection_line_segments(segment, other_segment)
                intersection1 = get_intersection_line_segments([point, endpoint1], other_segment)
                intersection2 = get_intersection_line_segments([point, endpoint2], other_segment)
                if (intersection1 is not None) and (intersection2 is not None):
                    should_insert = False
                    break
                elif intersection is not None:
                    if intersection1 is not None:
                        endpoint1 = intersection
                    if intersection2 is not None:
                        endpoint2 = intersection
        if should_insert:
            cut_segments.append([endpoint1, endpoint2])
            angle_range_from = get_vector_angle(endpoint1 - point)
            angle_range_to = get_vector_angle(endpoint2 - point)
            if angle_range_from <= angle_range_to:
                cut_segments_angle_range.append([angle_range_from, angle_range_to])
            else: # avoiding "jumping" range
                cut_segments_angle_range.append([angle_range_from, 2 * np.pi])
                cut_segments_angle_range.append([0, angle_range_to])

    limited_voronoi_partition = list()
    for segment in cut_segments:
        limited_voronoi_partition.append((0, segment[0], segment[1]))

    # comparator function for sorting cutSegmentsAngleRange
    # the "smallest" range is a range where it has the smallest start point
    def compare(item):
        return item[0]
    cut_segments_angle_range.sort(key=compare)
    prev_angle = 0
    for i in range(len(cut_segments_angle_range)):
        if prev_angle >= 2 * np.pi: break
        isBigEnough = (abs(prev_angle - cut_segments_angle_range[i][0]) > angle_error_tolerance)
        if prev_angle < cut_segments_angle_range[i][0] and isBigEnough:
            limited_voronoi_partition.append([1, prev_angle, cut_segments_angle_range[i][0]])
        prev_angle = max(prev_angle, cut_segments_angle_range[i][1])
    if prev_angle < 2 * np.pi:
        limited_voronoi_partition.append((1, prev_angle, 2 * np.pi))

    return limited_voronoi_partition

def h(points: np.ndarray, voronoi_partitions: List[np.ndarray]) -> float:
    """The sensing performance function

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        voronoi_partitions: voronoi partition of points after intersected with a certain border polygon
            the ith element is the ith point's voronoi partition in CCW order
            shape of ndarray is (n_ith_point_polygon, 2)
    
    Returns:
        the value of sensing performance.
    """

    result = 0
    for i in range(len(points)):
        def combined_function(point: np.ndarray) -> float:
            return f(point, points[i]) * phi(point)
        result = result + get_integral_over_convex_polygon(voronoi_partitions[i], combined_function)
    return result

def dhdp_area(point: np.ndarray, limited_voronoi_partition: np.ndarray, half_radius: float) -> float:
    """
    The dH/dp function for the Area case f(p) = {1 if |p - q| <= radius; 0 otherwise}
    This function runs for one single point. Thus, for this Area case, it is spatially distributed

    Args:
        point: position of the agent
            shape of ndarray is (2,)
        limited_voronoi_partition: limited voronoi partition of the point after intersected with a certain border polygon
            refer to get_limited_voronoi_partition function's returns for detailed explanation;
            shape of ndarray is (npolygonpoints, 2)
        half_radius: the radius of the ball. in other words, half of the maximum allowed distance between two points to be considered as neighbor

    Returns:
        the dHdp value for the point
    """

    def function_x(p):
        normal = get_unit_normal_at_point_in_arc(point, p)
        return normal[0] * phi(p)
    def function_y(p):
        normal = get_unit_normal_at_point_in_arc(point, p)
        return normal[1] * phi(p)
    
    result = np.zeros((2))
    for border in limited_voronoi_partition:
        if border[0] == 1:
            result[0] = result[0] + get_integral_over_arc(point, half_radius, border[1], border[2], function_x)
            result[1] = result[1] + get_integral_over_arc(point, half_radius, border[1], border[2], function_y)
    return result

def epsilon(point: np.ndarray, limited_voronoi_partition: np.ndarray) -> float:
    return 0.002
    median = np.median(limited_voronoi_partition, 0)
    return np.abs(median - np.array(point)) / 3

def simulate(points: np.ndarray, radius: float, border_polygon: np.ndarray, max_simulation_time: float = 30) -> np.ndarray:
    """Run the deployment simulation to a set of agent points to increase overall sensing performance

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        radius: the maximum allowed distance between two points to be considered as neighbor
        border_polygon: "border" of the voronoi partition in CCW order
            shape of ndarray is (n_border_points, 2)
        max_simulation_time: max simulation time
    
    Returns:
        the position of the agents after deployment
            shape of ndarray is (npoints, 2)
    """

    voronoi_partition = get_voronoi_partitions(points, border_polygon)
    print("Initial H value:", h(points, voronoi_partition))
    draw_limited_voronoi_partitions(points, radius, border_polygon)

    time_start = time()
    while time() - time_start < max_simulation_time:
        # update points
        new_points = np.zeros(points.shape)
        for i in range(len(points)):
            limitedVoronoiPartition = get_limited_voronoi_partition(points[i], get_near_points(points, i, radius), radius / 2, border_polygon)
            dHdp = dhdp_area(points[i], limitedVoronoiPartition, radius / 2)
            new_points[i] = points[i] + epsilon(points[i], limitedVoronoiPartition) * dHdp * points[i]
        points = new_points
    
    voronoi_partition = get_voronoi_partitions(points, border_polygon)
    print("Final H value:", h(points, voronoi_partition))
    draw_limited_voronoi_partitions(points, radius, border_polygon)

def draw_voronoi_partitions(points: np.ndarray, voronoi_partitions: List[np.ndarray]) -> None:
    """Draw the given points and voronoi_partitions using matplotlib

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        voronoi_partitions: voronoi partition of points after intersected with a certain border polygon
            the ith element is the ith point's voronoi partition in CCW order
            shape of ndarray is (n_ith_point_polygon, 2)

    Returns:
        None
    """

    for i in range(len(voronoi_partitions)):
        plt.plot(points[i][0], points[i][1], 'ro')
        plt.plot(np.append(voronoi_partitions[i][:,0], voronoi_partitions[i][0,0]), np.append(voronoi_partitions[i][:,1], voronoi_partitions[i][0,1]))
    plt.show()

def draw_limited_voronoi_partitions(points: np.ndarray, radius: float, border_polygon: np.ndarray) -> None:
    """Draw the given points and limited voronoi partition using matplotlib

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        radius: the maximum allowed distance between two points to be considered as neighbor
        border_polygon: "border" of the voronoi partition in CCW order
            shape of ndarray is (n_border_points, 2)

    Returns:
        None
    """

    # draw partition
    limited_voronoi_partitions = list()
    for i in range(len(points)):
        limited_voronoi_partition = get_limited_voronoi_partition(points[i], get_near_points(points, i, radius), radius / 2, border_polygon)
        limited_voronoi_partitions.append(limited_voronoi_partition)

    fig, ax = plt.subplots()
    for i in range(len(points)):
        ax.plot(points[i][0], points[i][1], 'ro')
        for current_partition in limited_voronoi_partitions[i]:
            if current_partition[0] == 0: # a line segment
                x = [current_partition[1][0], current_partition[2][0]]
                y = [current_partition[1][1], current_partition[2][1]]
                ax.plot(x, y)
            else: # an arc
                arc = matplotlib.patches.Arc(points[i], radius, radius, 0, np.rad2deg(current_partition[1]), np.rad2deg(current_partition[2]))
                ax.add_patch(arc)

    border_polygon = np.array(border_polygon)
    border_polygon = np.append(border_polygon, [border_polygon[0]], 0)
    ax.plot(border_polygon[:,0], border_polygon[:,1])

    plt.axis('scaled')
    plt.show()

points = np.array([[0.515, 0.8], [0.585, 0.995], [0.735, 1.1], [0.76, 0.845], [0.88, 0.755], [0.885, 0.55], [0.93, 1.07], [1.11, 0.4], [1.155, 1.135], [1.21, 0.68], [1.345, 0.48], [1.38, 0.565], [2.235, 0.75], [2.29, 0.73], [2.405, 0.95], [2.4, 0.7]])
border_polygon = np.array([[0, 0], [2.125, 0], [2.9325, 1.5], [2.975, 1.6], [2.9325, 1.7], [2.295, 2.1], [0.85, 2.3], [0.17, 1.2]])
radius = 0.45

simulate(points, radius, border_polygon, 20)