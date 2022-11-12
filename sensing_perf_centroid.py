from time import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import quadpy
from scipy.spatial import Voronoi

from LazyVinh.convex_polygon_intersection import intersect


def f(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def Phi(point):
    center = [[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]]
    result = 0
    for c in center:
        result = result + 5 * np.exp(-6 * ((point[0] - c[0])**2 + (point[1] - c[1])**2))
    return result

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

def get_intersection(polygon1: np.ndarray, polygon2: np.ndarray) -> np.ndarray:
    """
    Args:
        polygon1, polygon2: convex polygon in CCW order
            shape of ndarray is (npoints, 2)
    
    Returns:
        points of the polygon in CCW order.
            shape of ndarray is (npoints, 2)
    """

    intersection = intersect(polygon1, polygon2)
    return np.array([np.array(points) for points in intersection])

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

def get_voronoi_partition(points: np.ndarray, border_polygon: np.ndarray) -> List[np.ndarray]: # type: ignore
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
        voronoi_partition.append(get_intersection(region_points, border_polygon))
    
    return voronoi_partition

def h(points: np.ndarray, voronoi_partitions: List[np.ndarray]) -> float: # type: ignore
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
            return f(point, points[i]) * Phi(point)
        result = result + get_integral_over_convex_polygon(voronoi_partitions[i], combined_function)
    return result

def dhdp_centroid(points: np.ndarray, voronoi_partitions: List[np.ndarray]) -> np.ndarray: # type: ignore
    """
    The dH/dp function for the Centroid case f(p) = -(|p - q| ** 2)
    This function runs for all points, not one single points. Thus, for this Centroid case, it is not spatially distributed

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        voronoi_partitions: voronoi partition of points after intersected with a certain border polygon
            the ith element is the ith point's voronoi partition in CCW order
            shape of ndarray is (n_ith_point_polygon, 2)

    Returns:
        the dHdp value for all points (respectively)
    """

    result = []
    for i in range(len(points)):
        mass = get_integral_over_convex_polygon(voronoi_partitions[i], Phi)
        def phi_x(point: np.ndarray) -> float:
            return Phi(point) * point[0]
        center_of_mass_x = get_integral_over_convex_polygon(voronoi_partitions[i], phi_x) / mass
        def phi_y(point: np.ndarray) -> float:
            return Phi(point) * point[1]
        center_of_mass_y = get_integral_over_convex_polygon(voronoi_partitions[i], phi_y) / mass
        result.append([2 * mass * (center_of_mass_x - points[i][0]), 2 * mass * (center_of_mass_y - points[i][1])])
    return np.array(result)

def epsilon(point, voronoi_partition):
    return 0.1
    median = np.median(voronoi_partition, 0)
    return np.abs(median - np.array(point)) / 3

def simulate(points: np.ndarray, border_polygon: np.ndarray, max_simulation_time: float = 30) -> np.ndarray:
    """Run the deployment simulation to a set of agent points to increase overall sensing performance

    Args:
        points: position of the agents
            shape of ndarray is (npoints, 2)
        border_polygon: "border" of the voronoi partition in CCW order
            shape of ndarray is (n_border_points, 2)
        max_simulation_time: max simulation time
    
    Returns:
        the position of the agents after deployment
            shape of ndarray is (npoints, 2)
    """

    points = np.copy(points) # to avoid modifying the client's variable
    voronoi_partition = get_voronoi_partition(points, border_polygon)
    print("Initial H value:", h(points, voronoi_partition))
    draw_voronoi_partitions(points, voronoi_partition)

    timeStart = time()
    while time() - timeStart < max_simulation_time:
        dhdp = dhdp_centroid(points, voronoi_partition)
        # update points
        for i in range(len(points)):
            points[i] = points[i] + epsilon(points[i], voronoi_partition[i]) * dhdp[i] * points[i]
        # update voronoiPartition
        voronoi_partition = get_voronoi_partition(points, border_polygon)

    print("Final H value:", h(points, voronoi_partition))
    draw_voronoi_partitions(points, voronoi_partition)
    return points

def draw_voronoi_partitions(points: np.ndarray, voronoi_partitions: List[np.ndarray]) -> None: # type: ignore
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

points = np.array([[0.515, 0.8], [0.585, 0.995], [0.735, 1.1], [0.76, 0.845], [0.88, 0.755], [0.885, 0.55], [0.93, 1.07], [1.11, 0.4], [1.155, 1.135], [1.21, 0.68], [1.345, 0.48], [1.38, 0.565], [2.235, 0.75], [2.29, 0.73], [2.405, 0.95], [2.4, 0.7]])
borderPolygon = np.array([[0, 0], [2.125, 0], [2.9325, 1.5], [2.975, 1.6], [2.9325, 1.7], [2.295, 2.1], [0.85, 2.3], [0.17, 1.2]])
simulate(points, borderPolygon, 20)