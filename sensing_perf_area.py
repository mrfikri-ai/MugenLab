import numpy as np
from time import time

import warnings
warnings.filterwarnings('error', '.*extremely bad integrand behavior.*')

def f(p1, p2):
    import math

    R = 0.45
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    if dist <= R: return 1
    else: return 0

def Phi(point):
    from math import exp

    center = [[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]]
    result = 0
    for c in center:
        result = result + 5 * exp(-6 * ((point[0] - c[0])**2 + (point[1] - c[1])**2))
    return result

def isPolygonCCW(polygon):
    # input: a convex polygon where only the (i)th and (i+1)th point is connected (first and last points are also connected); list[npoints, 2]
    #   *note that the polygon must be either CW or CCW
    # return: a boolean value whether the points order is CCW or not; boolean
    # only need to check three consecutive points
    if len(polygon) < 3:
        return True
    else:
        a = np.subtract(polygon[1], polygon[0])
        b = np.subtract(polygon[2], polygon[0])
        return np.cross(a, b) > 0

def isPointQCloserToP(p, q, r):
    # input: p is the "pivot", q and r is the one to be compared; ndarray shape(2)
    # return: True if distance(p, q) is smaller than distance(q, r); boolean
    return np.abs(np.linalg.norm(q - p)) < np.abs(np.linalg.norm(r - p))

def getVectorAngle(v):
    # return: a number in range of 0 to 2pi inclusive; float
    lenV = np.abs(np.linalg.norm(v))
    assert(lenV > 0)
    v = v / lenV
    angle = np.arccos(np.clip(np.dot(v, np.array([1, 0])), -1.0, 1.0))
    if v[1] < 0:
        angle = 2 * np.pi - angle
    return angle

def getAngleBetweenVectors(v1, v2):
    # input: v1 and v2 are nonzero ndarray; ndarray shape(2)
    #   if v2 is not specified, then v2 = [1, 0]
    lenV1 = abs(np.linalg.norm(v1))
    lenV2 = abs(np.linalg.norm(v2))
    assert(lenV1 > 0 and lenV2 > 0)
    v1 = v1 / lenV1
    v2 = v2 / lenV2
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def getRotatedVector(vector, angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation, vector)

def getIntersectionLineSegment(s1, s2):
    # credit: https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
    # input: s1 and s2 are line segment; list[2, 2]
    # return: if exist, the intersection point, otherwise None; list[2] or None

    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]
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
    return [x,y]

def getIntersectionCircleAndLineSegment(circleCenter, circleRadius, p1, p2, full_line=True, tangent_tol=1e-9):
    """ From: https://stackoverflow.com/a/59582674/12607236
    Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = p1, p2, circleCenter
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circleRadius ** 2 * dr ** 2 - big_d ** 2

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

def getIntersectionPolygon(p1, p2):
    # input: p1 and p2 is convex polygon in CCW order; list[npoints, 2]
    # return: list of points of the polygon in CCW order; ndarray shape(nresultpoints, 2)
    from imports.LazyVinh.convex_polygon_intersection import intersect

    intersection = intersect(p1, p2)
    return np.array([np.array(points) for points in intersection])

def getIntegralOverConvexPolygon(polygon, function, functionDegree = 12):
    # input:
    #   a convex polygon in CCW order; list[npoints, 2]
    #   a function to be integrated over the polygon; a function that accepts exactly 1 argument
    #       the argument format is list[2]
    #   functionDegree is the highest polynomial degree of the function, used to determine what integration algorithm will be used; an integer
    # return: the resulting integral duh
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

def getIntegralOverArc(arcCenter, arcRadius, arcAngleFrom, arcAngleTo, function):
    # inputs:
    #   arc... define the properties of the arc. arcAngle... is in radian; float
    #       arcCenter format is list[2]
    #       arcRadius, arcAngleFrom, arcAngleTo format are float
    #   function is the function to be integrated; a function that accepts exactly 1 argument
    #       the argument format is list[2]
    # return: the resulting integral duh; float
    from scipy.integrate import quad

    def functionHelper(angle):
        point = np.array(arcCenter)
        direction = getRotatedVector(np.array([arcRadius, 0]), angle)
        return function(point + direction)
    return quad(functionHelper, arcAngleFrom, arcAngleTo, limit=100)[0]

def getUnitNormalAtPointInArc(arcCenter, arcPoint):
    # return: a unit normal at arcPoint in an arc with arcCenter as its center; list[2]
    vector = np.array(arcPoint) - np.array(arcCenter)
    normal = vector / np.linalg.norm(vector)
    return normal

def getNearPoints(points, indexPoint, radius):
    # input:
    #   points is the list of points; list[npoints, 2]
    #   indexPoint is the index of the point inside points of what we are interested in; integer
    #   radius is the maximum of distance between points[indexPoint] and point[i]; float
    # return: list of points that is close to the indexPoint, without the point itself; list[nresult, 2]

    result = list()
    for i in range(len(points)):
        if i != indexPoint:
            distance = np.abs(np.linalg.norm(np.array(points[i]) - np.array(points[indexPoint])))
            if distance <= radius:
                result.append(points[i].copy())
    return result

def getVoronoiPartition(points, borderPolygon):
    # inputs: borderPolygon is the "border" of the voronoi partition in CCW order; list[npoints, 2]
    # return: a list of polygons, where the ith polygon is the voronoi partition of the ith point in CCW order; list[npoints] consisting of ndarrays shape(n[i], 2)
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
        voronoiPartition.append(getIntersectionPolygon(regionPoints, borderPolygon))
    
    # print("getVoronoiPartition execution time:", time() - timeStart)
    return voronoiPartition

def getLimitedVoronoiPartition(point, nearPoints, radius, borderPolygon, angleErrorTolerance = 1e-8):
    # inputs:
    #   point; list[2]
    #   nearPoints; list[npoints, 2]
    #   radius; float
    # return: list of line segments and arcs that describe this point's limited voronoi partition; list[nobjects, 3]
    #   a line segment and arc is differentiated by the first value of the list, where line segment is 0 and arc is 1
    #       a line segment entry is defined by [0, endPoint1, endPoint2]; endPoint1 and endPoint2 is ndarray shape(2)
    #       an arc entry is defined by [1, angleStart, angleEnd]; angleStart and angleEnd is float
    #           angleStart <= angleEnd is always true
    import scipy.spatial

    point = np.array(point)
    neighborPoints = list()
    if len(nearPoints) < 2:
        for nearPoint in nearPoints:
            neighborPoints.append(np.array(nearPoint))
    else:
        allPoints = np.append([point], nearPoints, 0)
        delaunay = scipy.spatial.Delaunay(allPoints)
        neighborPointsIndex = set()
        assert(point[0] == delaunay.points[0][0] and point[1] == delaunay.points[0][1])
        for triangle in delaunay.simplices:
            if 0 in triangle:
                for index in triangle:
                    if index != 0:
                        neighborPointsIndex.add(index)
        neighborPoints = [delaunay.points[index] for index in neighborPointsIndex]

    uncutSegments = [] # the "voronoi" segments created by these neighbor points. some of these segments might overlap each other
    for neighborPoint in neighborPoints:
        # finding each segment's endpoints at the circle(point, radius)
        direction = neighborPoint - point
        directionAngle = getVectorAngle(direction)
        distanceToMidPoint = np.abs(np.linalg.norm(direction / 2))
        deltaAngle = np.arccos(distanceToMidPoint / radius)
        # endPoint1 to endPoint2 is in CCW order
        endPoint1 = point + getRotatedVector(np.array([radius, 0]), directionAngle - deltaAngle)
        endPoint2 = point + getRotatedVector(np.array([radius, 0]), directionAngle + deltaAngle)
        uncutSegments.append([endPoint1, endPoint2])
    for i in range(len(borderPolygon)): # for border polygon
        endPoint1 = np.array(borderPolygon[i-1])
        endPoint2 = np.array(borderPolygon[i])
        # we regard this "line segment" as an "infinity line", because all uncutSegments have their endPoints at the circle
        intersections = getIntersectionCircleAndLineSegment(point, radius, endPoint1, endPoint2)
        if len(intersections) == 2:
            uncutSegments.append([intersections[0], intersections[1]])
        else:
            assert(len(intersections) == 0 or len(intersections) == 1), "Unexpected number of intersections"
    uncutSegments = np.array(uncutSegments)
    
    cutSegments = list()
    cutSegmentsAngleRange = list() # there are no "jumping" range
    for segment in uncutSegments:
        endPoint1 = segment[0]
        endPoint2 = segment[1]
        shouldInsert = True
        for otherSegment in uncutSegments:
            if not np.array_equal(segment, otherSegment):
                intersection = getIntersectionLineSegment(segment, otherSegment)
                intersection1 = getIntersectionLineSegment([point, endPoint1], otherSegment)
                intersection2 = getIntersectionLineSegment([point, endPoint2], otherSegment)
                if (intersection1 is not None) and (intersection2 is not None):
                    shouldInsert = False
                    break
                elif intersection is not None:
                    if intersection1 is not None:
                        endPoint1 = intersection
                    if intersection2 is not None:
                        endPoint2 = intersection
        if shouldInsert:
            cutSegments.append([endPoint1, endPoint2])
            angleRangeFrom = getVectorAngle(endPoint1 - point)
            angleRangeTo = getVectorAngle(endPoint2 - point)
            if angleRangeFrom <= angleRangeTo:
                cutSegmentsAngleRange.append([angleRangeFrom, angleRangeTo])
            else: # avoiding "jumping" range
                cutSegmentsAngleRange.append([angleRangeFrom, 2 * np.pi])
                cutSegmentsAngleRange.append([0, angleRangeTo])

    limitedVoronoiPartition = list()
    for segment in cutSegments:
        limitedVoronoiPartition.append([0, segment[0], segment[1]])

    # comparator function for sorting cutSegmentsAngleRange
    # the "smallest" range is a range where it has the smallest start point
    def compare(item):
        return item[0]
    cutSegmentsAngleRange.sort(key=compare)
    prevAngle = 0
    indexAngleRange = 0
    while prevAngle < 2 * np.pi and indexAngleRange < len(cutSegmentsAngleRange):
        isBigEnough = (abs(prevAngle - cutSegmentsAngleRange[indexAngleRange][0]) > angleErrorTolerance)
        if prevAngle < cutSegmentsAngleRange[indexAngleRange][0] and isBigEnough:
            limitedVoronoiPartition.append([1, prevAngle, cutSegmentsAngleRange[indexAngleRange][0]])
        prevAngle = max(prevAngle, cutSegmentsAngleRange[indexAngleRange][1])
        indexAngleRange = indexAngleRange + 1
    if prevAngle < 2 * np.pi:
        limitedVoronoiPartition.append([1, prevAngle, 2 * np.pi])

    return limitedVoronoiPartition

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

def dHdpArea(point, limitedVoronoiPartition, radius):
    def functionX(p):
        normal = getUnitNormalAtPointInArc(point, p)
        return normal[0] * Phi(p)
    def functionY(p):
        normal = getUnitNormalAtPointInArc(point, p)
        return normal[1] * Phi(p)
    
    result = np.zeros((2))
    for border in limitedVoronoiPartition:
        if border[0] == 1:
            result[0] = result[0] + getIntegralOverArc(point, radius, border[1], border[2], functionX)
            result[1] = result[1] + getIntegralOverArc(point, radius, border[1], border[2], functionY)
    return result

def epsilon(point, currentVoronoiPartition): # currently not used. look inside simulate() function for epsilon
    return 0.03
    median = np.median(currentVoronoiPartition, 0)
    return np.abs(median - np.array(point)) / 3

def simulate(points, radius, borderPolygon, maxTime = 30):
    voronoiPartition = getVoronoiPartition(points, borderPolygon)
    print("Initial H value:", H(points, voronoiPartition))
    drawLimitedVoronoiPartition(points, radius, borderPolygon)
    timeStart = time()
    while time() - timeStart < maxTime:
        # update points
        newPoints = [None] * len(points)
        for i in range(len(points)):
            limitedVoronoiPartition = getLimitedVoronoiPartition(points[i], getNearPoints(points, i, radius), radius / 2, borderPolygon)
            dHdp = dHdpArea(points[i], limitedVoronoiPartition, radius / 2)
            epsilon = 0.002
            newPoints[i] = np.array(points[i]) + epsilon * dHdp * np.array(points[i])
        points = newPoints
    print("Final H value:", H(points, voronoiPartition))
    drawLimitedVoronoiPartition(points, radius, borderPolygon)

def printVoronoiPartition(points, voronoiPartition):
    import matplotlib.pyplot as plt

    timeStart = time()
    for i in range(len(voronoiPartition)):
        plt.plot(points[i][0], points[i][1], 'ro')
        plt.plot(np.append(voronoiPartition[i][:,0], voronoiPartition[i][0,0]), np.append(voronoiPartition[i][:,1], voronoiPartition[i][0,1]))
    # print("printVoronoiPartition execution time:", time() - timeStart)
    plt.show()

def drawLimitedVoronoiPartition(points, radius, borderPolygon):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    # draw partition
    limitedVoronoiPartitions = list()
    for i in range(len(points)):
        limitedVoronoiPartition = getLimitedVoronoiPartition(points[i], getNearPoints(points, i, radius), radius / 2, borderPolygon)
        limitedVoronoiPartitions.append(limitedVoronoiPartition)

    fig, ax = plt.subplots()
    for i in range(len(points)):
        ax.plot(points[i][0], points[i][1], 'ro')
        for currentPartition in limitedVoronoiPartitions[i]:
            if currentPartition[0] == 0: # a line segment
                x = [currentPartition[1][0], currentPartition[2][0]]
                y = [currentPartition[1][1], currentPartition[2][1]]
                ax.plot(x, y)
            else: # an arc
                arc = Arc(points[i], radius, radius, 0, np.rad2deg(currentPartition[1]), np.rad2deg(currentPartition[2]))
                ax.add_patch(arc)

    borderPolygon = np.array(borderPolygon)
    borderPolygon = np.append(borderPolygon, [borderPolygon[0]], 0)
    ax.plot(borderPolygon[:,0], borderPolygon[:,1])

    # fig.set_figheight(6)
    # fig.set_figwidth(6)
    # # plt.xlim(-3, 7)
    # # plt.ylim(-3, 7)
    # plt.xlim(0, 3)
    # plt.ylim(0, 3)
    plt.axis('scaled')
    plt.show()

points = [[0.515, 0.8], [0.585, 0.995], [0.735, 1.1], [0.76, 0.845], [0.88, 0.755], [0.885, 0.55], [0.93, 1.07], [1.11, 0.4], [1.155, 1.135], [1.21, 0.68], [1.345, 0.48], [1.38, 0.565], [2.235, 0.75], [2.29, 0.73], [2.405, 0.95], [2.4, 0.7]]
borderPolygon = [[0, 0], [2.125, 0], [2.9325, 1.5], [2.975, 1.6], [2.9325, 1.7], [2.295, 2.1], [0.85, 2.3], [0.17, 1.2]]
radius = 0.45
# points = [[0, 0], [1, 3], [5, 3], [2, 4], [0, 5]]
# borderPolygon = [[0,0], [10, 0], [10, 10], [0, 10]]

simulate(points, radius, borderPolygon, 20)

# print(H(points, getVoronoiPartition(points, borderPolygon)))
# printVoronoiPartition(points, getVoronoiPartition(points, borderPolygon))