import numpy as np
from time import time

def f(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def Phi(point):
    from math import exp

    center = [[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]]
    result = 0
    for c in center:
        result = result + 5 * exp(-6 * ((point[0] - c[0])**2 + (point[1] - c[1])**2))
    return result

# input: a convex polygon where only the (i)th and (i+1)th point is connected (first and last points are also connected); list[npoints, 2]
#   *note that the polygon must be either CW or CCW
# return: a boolean value whether the points order is CCW or not; boolean
def isPolygonCCW(polygon):
    # only need to check three consecutive points
    if len(polygon) < 3:
        return True
    else:
        a = np.subtract(polygon[1], polygon[0])
        b = np.subtract(polygon[2], polygon[0])
        return np.cross(a, b) > 0

# input: p1 and p2 is convex polygon in CCW order; list[npoints, 2]
# return: list of points of the polygon in CCW order; ndarray shape(nresultpoints, 2)
def getIntersection(p1, p2):
    from imports.LazyVinh.convex_polygon_intersection import intersect

    intersection = intersect(p1, p2)
    return np.array([np.array(points) for points in intersection])

# input:
#   a convex polygon in CCW order; list[npoints, 2]
#   a function to be integrated over the polygon; a function that accepts exactly 1 argument
#   functionDegree is the highest polynomial degree of the function, used to determine what integration algorithm will be used; an integer
# return: the resulting integral duh
def getIntegralOverConvexPolygon(polygon, function, functionDegree = 12):
    import quadpy

    # quadpy needs a function to accept array of testValues. this function will help that
    def funcHelper(testValues):
        result = np.zeros(testValues.shape[1])
        for i in range(testValues.shape[1]):
            result[i] = function(testValues[:,i])
        return result
    
    scheme = quadpy.t2.get_good_scheme(functionDegree)
    result = 0
    if len(polygon) >= 3:
        for i in range(1, len(polygon) - 1):
            result = result + scheme.integrate(lambda x: funcHelper(x), [polygon[0], polygon[i], polygon[i+1]])
    return result

# inputs: borderPolygon is the "border" of the voronoi partition in CCW order; list[npoints, 2]
# return: a list of polygons, where the ith polygon is the voronoi partition of the ith point in CCW order; list[npoints] consisting of ndarrays shape(n[i], 2)
def getVoronoiPartition(points, borderPolygon):
    from scipy.spatial import Voronoi

    borderCoord = 1e6 # for now, let's assume 1e6 is big enough

    timeStart = time()
    newPoints = points.copy()
    newPoints.append([-borderCoord, 0])
    newPoints.append([borderCoord, 0])
    newPoints.append([0, -borderCoord])
    newPoints.append([0, borderCoord])
    
    voronoi = Voronoi(newPoints)
    voronoiPartition = []
    for i in range(len(points)):
        regionIndex = voronoi.point_region[i]
        regionPoints = []
        for j in voronoi.regions[regionIndex]:
            assert(j != -1), "Region " + newPoints[i] + " is not bounded"
            regionPoints.append(voronoi.vertices[j])
        if not(isPolygonCCW(regionPoints)): # scipy's voronoi region order is either CW or CCW, so we need to fix it to become CCW
            regionPoints.reverse()
        voronoiPartition.append(getIntersection(regionPoints, borderPolygon))
    
    # print("getVoronoiPartition execution time:", time() - timeStart)
    return voronoiPartition

def H(points, voronoiPartition):    
    timeStart = time()
    result = 0
    for i in range(len(points)):
        def combinedFunction(point):
            return f(point, points[i]) * Phi(point)
        result = result + getIntegralOverConvexPolygon(voronoiPartition[i], combinedFunction)
    # print("H execution time:", time() - timeStart)
    return result

def dHdpCentroid(points, voronoiPartition):
    result = []
    for i in range(len(points)):
        mass = getIntegralOverConvexPolygon(voronoiPartition[i], Phi)
        def PhiX(point):
            return Phi(point) * point[0]
        centerOfMassX = getIntegralOverConvexPolygon(voronoiPartition[i], PhiX) / mass
        def PhiY(point):
            return Phi(point) * point[1]
        centerOfMassY = getIntegralOverConvexPolygon(voronoiPartition[i], PhiY) / mass
        result.append([2 * mass * (centerOfMassX - points[i][0]), 2 * mass * (centerOfMassY - points[i][1])])
    return np.array(result)

def epsilon(point, currentVoronoiPartition):
    return 0.1
    median = np.median(currentVoronoiPartition, 0)
    return np.abs(median - np.array(point)) / 3

def simulate(points, borderPolygon, maxTime = 30):
    voronoiPartition = getVoronoiPartition(points, borderPolygon)
    print("Initial H value:", H(points, voronoiPartition))
    printVoronoiPartition(points, voronoiPartition)
    timeStart = time()
    while time() - timeStart < maxTime:
        dHdp = dHdpCentroid(points, voronoiPartition)
        # update points
        for i in range(len(points)):
            points[i] = np.array(points[i]) + epsilon(points[i], voronoiPartition[i]) * dHdp[i] * np.array(points[i])
        # update voronoiPartition
        voronoiPartition = getVoronoiPartition(points, borderPolygon)
    print("Final H value:", H(points, voronoiPartition))
    printVoronoiPartition(points, voronoiPartition)

def printVoronoiPartition(points, voronoiPartition):
    import matplotlib.pyplot as plt

    timeStart = time()
    for i in range(len(voronoiPartition)):
        plt.plot(points[i][0], points[i][1], 'ro')
        plt.plot(np.append(voronoiPartition[i][:,0], voronoiPartition[i][0,0]), np.append(voronoiPartition[i][:,1], voronoiPartition[i][0,1]))
    # print("printVoronoiPartition execution time:", time() - timeStart)
    plt.show()

points = [[0.515, 0.8], [0.585, 0.995], [0.735, 1.1], [0.76, 0.845], [0.88, 0.755], [0.885, 0.55], [0.93, 1.07], [1.11, 0.4], [1.155, 1.135], [1.21, 0.68], [1.345, 0.48], [1.38, 0.565], [2.235, 0.75], [2.29, 0.73], [2.405, 0.95], [2.4, 0.7]]
borderPolygon = [[0, 0], [2.125, 0], [2.9325, 1.5], [2.975, 1.6], [2.9325, 1.7], [2.295, 2.1], [0.85, 2.3], [0.17, 1.2]]
simulate(points, borderPolygon, 20)
# print(H(points, getVoronoiPartition(points, borderPolygon)))
# printVoronoiPartition(points, getVoronoiPartition(points, borderPolygon))