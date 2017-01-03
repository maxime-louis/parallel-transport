# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import *
import math
from mpl_toolkits.mplot3d import Axes3D
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt


def invcotang(x):
    if x==0:
        return math.pi/2
    elif x<0:
        return math.pi + np.arctan(1/x)
    else:
        return np.arctan(1/x)

def localChartTo3D(x):
    dimension = len(x)
    if (dimension != 2):
        raise ValueError("Dimension error")
    x3D = np.zeros(3)
    x3D[0] = np.sin(x[0])*np.cos(x[1])
    x3D[1] = np.sin(x[0])*np.sin(x[1])
    x3D[2] = np.cos(x[0])
    return x3D

def chartVelocityTo3D(x,v):
    dimension = len(x)
    out = np.zeros(3)
    M = np.array([[np.cos(x[0])*np.cos(x[1]), -np.sin(x[0])*np.sin(x[1])],
         [np.cos(x[0])*np.sin(x[1]), np.sin(x[0])*np.cos(x[1])],
         [-np.sin(x[0]), 0]])
    aux = np.matmul(M,v)
    for k,elt in enumerate(aux):
        out[k] = elt
    return out

#Returns the position on the geodesic at time t with gamma(0)=x, gamma'(0)=v
def computeGeodesic(x3D, v3D, t):
    v3DNorm = np.linalg.norm(v3D)
    if v3DNorm < 1e-20:
        return x3D
    return np.cos(t*v3DNorm)*x3D + np.sin(t*v3DNorm)*v3D/v3DNorm

def to2D(x):
    assert len(x)==3, "Not the right dimension of input !"
    if abs(np.linalg.norm(x) - 1)>=1e-4:
        print(np.linalg.norm(x), "Not of norm 1")
    theta = np.arctan(x[1]/x[0])
    phi = np.arccos(x[2])
    return np.array([phi,theta])


def toLatituteLongitude(x):
    theta = x[0]
    phi = x[1]
    latitude = 90. - theta * 180./math.pi
    longitude = phi*180./math.pi
    return np.array([latitude, longitude])

def toSpherical(x):
    latitude = x[0]
    longitude = x[1]
    theta = (90 - latitude)*math.pi/180.
    phi = longitude*math.pi/180.
    return np.array([theta, phi])

def inverseProblemLine(x,y):
    #Get the latitude longitudes
    xl, yl = toLatituteLongitude(x), toLatituteLongitude(y)
    geod = Geodesic(1., 0.)
    g = geod.InverseLine(xl[0], xl[1], yl[0], yl[1])
    return g


# x = [0.2, 1.6]
# y = [2., 0.1]
# line = inverseProblemLine(x,y)
# print(line.s13)
# pos = line.Position(1., Geodesic.STANDARD)
# lat = pos['lat2']
# lon = pos['lon2']
# print toSpherical([lat, lon])

def SchildsLadder(x,v,w,number_of_time_steps):
    x3D = localChartTo3D(x)
    v3D = chartVelocityTo3D(x, v)
    dimension = 2 #Dimension of the manifold
    delta = 1./number_of_time_steps
    #To store the computed values of trajectory and transport
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    for i in range(number_of_time_steps+1):
        xtraj[i] = to2D(computeGeodesic(x3D, v3D, delta * i))
    pwtraj[0] = w
    time  = 0.
    for k in range(number_of_time_steps):
        print("\n")
        # Get P0, P1
        P0 = xtraj[k]
        P1 = xtraj[k+1]
        # Compute the first geodesic to find P2, from P0 with initial tangent vector wk
        P2 = to2D(computeGeodesic(localChartTo3D(P0), chartVelocityTo3D(P0, pwtraj[k]), delta))
        # Compute the geodesic linking P1 to P2 and its midpoint P3
        invLine = inverseProblemLine(P2, P1)
        print("P0", P0)
        print("P1", P1)
        print("P2", P2)
        pos = invLine.Position(invLine.s13/2., Geodesic.STANDARD)
        P3Latitude = pos['lat2']
        P3Longitude = pos['lon2']
        P3 = toSpherical([P3Latitude, P3Longitude])
        print("P3", P3)
        # Compute the geodesic linking P0 to P3 and go twice further to get P4, wk+1 is the riemannian
        invLine = inverseProblemLine(P0, P3)
        pos = invLine.Position(invLine.s13 * 2.)
        P4Latitude = pos['lat2']
        P4Longitude = pos['lon2']
        P4 = toSpherical([P4Latitude, P4Longitude])
        print("P4", P4)
        # logarithm of the geodesic connecting P1 to P4.
        invLine = inverseProblemLine(P1, P4)
        P1Obj = invLine.Position(0.)
        aziRa = P1Obj['azi1'] * math.pi/180.
        direction = np.array([-1.*np.cos(aziRa), np.sin(aziRa)])
        directionNorm = np.cos(aziRa)**2 + np.sin(-1.*np.cos(aziRa))**2*np.sin(aziRa)**2
        print("norm", directionNorm)
        normalizedDirection = direction#/ directionNorm
        transported = normalizedDirection * invLine.s13  /delta
        print("azimuth", aziRa, "transported", transported)
        precision = np.linalg.norm(computeGeodesic(localChartTo3D(P1), chartVelocityTo3D(P1, transported), delta) - localChartTo3D(P4))
        print("precision", precision)
        # assert precision<1e-2, precision
        pwtraj[k+1] = transported
    return xtraj, pwtraj

x = [2., 1.]
v = [0.01, 0.05]
w = [0.,0.1]
steps = 1
xtraj, pwtraj = SchildsLadder(x,v,w,steps)

x3D = localChartTo3D(x)
v3D = chartVelocityTo3D(x, v)
w3D = chartVelocityTo3D(x, w)

n = np.linalg.norm(v3D)
x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

proj = np.dot(v3D,w3D)/np.dot(v3D, v3D)
projOrtho = np.dot(np.cross(x3D,v3D), w3D)/np.dot(v3D,v3D)
truepw3D = proj * v3DFinal + projOrtho*np.cross(x3D, v3D)
print("true pw :", truepw3D)
print("Estimated :", chartVelocityTo3D(to2D(x3DFinal), pwtraj[-1]))
# x1 = [1.,1.]
# x2 = [1.5, 1.]
# invLine = inverseProblemLine(x1, x2)
# x1Obj = invLine.Position(0.)
# aziRa = x1Obj['azi1'] * math.pi/180.
# print("azimuth", aziRa, "distance", invLine.s13)
# v = np.array([-np.cos(aziRa), -np.sin(aziRa)]) * invLine.s13
# print("v", v)
# print(computeGeodesic(localChartTo3D(x1), chartVelocityTo3D(x1, v), 1.), localChartTo3D(x2))
