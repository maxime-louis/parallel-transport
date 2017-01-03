# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import *
import math
from mpl_toolkits.mplot3d import Axes3D
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt

def render_sphere(points, support = [], vectors=[]):
    #Set colours and render
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    xx, yy, zz = np.hsplit(points, 3)
    ax.scatter(xx,yy,zz,color="k",s=20)
    if len(vectors)>=1:
        assert len(support) == len(vectors), "Lengths don't match !"
        u, v, w = np.hsplit(vectors, 3)
        xxx, yyy, zzz = np.hsplit(support, 3)
        ax.quiver(xxx, yyy ,zzz , u, v, w,length = 0.3, pivot = "tail")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")
    plt.tight_layout()
    # plt.savefig("Graphs/ParallelTransportOnSphere.pdf")
    plt.show()

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
    phi = np.arctan(x[1]/x[0])
    if phi<=0:
        phi += math.pi#To keep phi where it belongs.
    theta = np.arccos(x[2])#Lies between 0 and pi : ok.
    return np.array([theta,phi])#This respects the convention


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

def SchildsLadder(x,v,w,number_of_time_steps, verbose = False, factor = 1.):
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
    maxprecision = 0.
    for k in range(number_of_time_steps):
        if verbose:
            print("\n")
        # Get P0, P1
        P0 = xtraj[k]
        P1 = xtraj[k+1]
        # Compute the first geodesic to find P2, from P0 with initial tangent vector wk
        P2 = to2D(computeGeodesic(localChartTo3D(P0), chartVelocityTo3D(P0, pwtraj[k]), delta*factor))
        # Compute the geodesic linking P1 to P2 and its midpoint P3
        invLine = inverseProblemLine(P2, P1)
        if verbose:
            print("P0", P0)
            print("P1", P1)
            print("P2", P2)
        pos = invLine.Position(invLine.s13/2., Geodesic.STANDARD)
        P3Latitude = pos['lat2']
        P3Longitude = pos['lon2']
        P3 = toSpherical([P3Latitude, P3Longitude])
        if verbose:
            print("P3", P3)
        # Compute the geodesic linking P0 to P3 and go twice further to get P4,
        invLine = inverseProblemLine(P0, P3)
        pos = invLine.Position(invLine.s13 * 2.)
        P4Latitude = pos['lat2']
        P4Longitude = pos['lon2']
        P4 = toSpherical([P4Latitude, P4Longitude])
        if verbose:
            print("P4", P4)
        #wk+1 is the riemannian logarithm of the geodesic connecting P1 to P4.
        invLine = inverseProblemLine(P1, P4)
        P1Obj = invLine.Position(0.)
        aziRa = P1Obj['azi1'] * math.pi/180.
        direction = np.array([-1.*np.cos(aziRa), np.sin(aziRa)])
        # directionNorm = np.cos(aziRa)**2 + np.sin(-1.*np.cos(aziRa))**2*np.sin(aziRa)**2
        # print("norm", directionNorm)
        normalizedDirection = direction#/ directionNorm
        transported = normalizedDirection * invLine.s13
        precision = np.linalg.norm(computeGeodesic(localChartTo3D(P1), chartVelocityTo3D(P1, transported), 1.) - localChartTo3D(P4))
        assert precision<=1e-15, "Large error in the inverse problem"
        if precision >= maxprecision:
            maxprecision = precision
        if verbose:
            print("precision", precision)
        # assert precision<1e-2, precision
        pwtraj[k+1] = transported/(delta*factor)
    # print("Largest error in inverse problem : ", maxprecision)
    return xtraj, pwtraj

def GetErrors():
    x = [math.pi/2.,5.1]
    x3D = localChartTo3D(x)
    v = [0., 1.]
    v3D = chartVelocityTo3D(x, v)
    w = [1.,0.]
    w3D = chartVelocityTo3D(x, w)

    n = np.linalg.norm(v3D)
    x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
    v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

    proj = np.dot(v3D,w3D)/np.dot(v3D, v3D)
    projOrtho = np.dot(np.cross(x3D,v3D), w3D)/np.dot(v3D,v3D)
    truepw3D = proj * v3DFinal + projOrtho*np.cross(x3D, v3D)

    errors = []
    nb = [int(i*10) for i in np.linspace(10,100,20)]
    inverseNb = [1./elt for elt in nb]

    for step in nb:
        xtraj, pwtraj = SchildsLadder(x,v,w,step)
        xtraj3D = np.array([localChartTo3D(elt) for elt in xtraj])
        pwtraj3D = np.array([chartVelocityTo3D(xtraj[i], pwtraj[i]) for i in range(len(xtraj))])
        errors.append(np.linalg.norm(pwtraj3D[-1] - truepw3D))
        # print("true pw :", truepw3D)
        # print("Estimated :", chartVelocityTo3D(to2D(x3DFinal), pwtraj[-1]))
        print("Error :", errors[-1], "Steps :", step)
    return nb, errors


def ErrorAsFunctionOfDelta():
    x = [math.pi/2.,5.1]
    x3D = localChartTo3D(x)
    v = [0., 1.]
    v3D = chartVelocityTo3D(x, v)
    w = [1.,0.]
    w3D = chartVelocityTo3D(x, w)

    n = np.linalg.norm(v3D)
    x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
    v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

    proj = np.dot(v3D,w3D)/np.dot(v3D, v3D)
    projOrtho = np.dot(np.cross(x3D,v3D), w3D)/np.dot(v3D,v3D)
    truepw3D = proj * v3DFinal + projOrtho*np.cross(x3D, v3D)

    nbSteps = 50
    factors = np.linspace(0.0001,0.0002,50)
    errors = []
    for fact in factors:
        xtraj, pwtraj = SchildsLadder(x,v,w,nbSteps,factor=fact)
        xtraj3D = np.array([localChartTo3D(elt) for elt in xtraj])
        pwtraj3D = np.array([chartVelocityTo3D(xtraj[i], pwtraj[i]) for i in range(len(xtraj))])
        errors.append(np.linalg.norm(pwtraj3D[-1] - truepw3D))
        # print("true pw :", truepw3D)
        # print("Estimated :", chartVelocityTo3D(to2D(x3DFinal), pwtraj[-1]))
        print("Error :", errors[-1], "factor :", fact)
    return factors, errors

abscisse, errors = GetErrors()
# st = [1./elt for elt in abscisse]
# plt.plot(st, errors)
# plt.xlim(xmin = 0)
# plt.ylim(ymin = 0)
#plt.savefig("Graphs/Schildserror.pdf")
# plt.show()









# P1 = [1.5,5]
# P2 = [0.7,6.]
# P13D = localChartTo3D(P1)
# P23D = localChartTo3D(P2)
# print(P13D, P23D)
# invLine = inverseProblemLine(P1, P2)
#
# nbSteps = 100
# traj = np.zeros((nbSteps, 3))
# for i,t in enumerate(np.linspace(0., invLine.s13, nbSteps)):
#     obj = invLine.Position(t)
#     lon = obj['lon2']
#     lat = obj['lat2']
#     traj[i] = localChartTo3D(toSpherical([lat, lon]))
#
# P1Obj = invLine.Position(0.)
# aziRa = P1Obj['azi1'] * math.pi/180.
# direction = np.array([-1.*np.cos(aziRa), np.sin(aziRa)])
# direction3D = chartVelocityTo3D(P1, direction)
# support = np.array([P13D])
# vector = np.array([direction3D])
#
# render_sphere(traj, support, vector)
