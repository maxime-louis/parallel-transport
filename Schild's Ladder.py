# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import *
import math
from mpl_toolkits.mplot3d import Axes3D
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
from scipy import linalg

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
    norm = np.linalg.norm(v3D)
    return np.cos(t*norm) * x3D + np.sin(t*norm) * v3D/norm

def trueParallelTransport(x,v,w,t):
    v3D = chartVelocityTo3D(x, v)
    x3D = localChartTo3D(x)
    w3D = chartVelocityTo3D(x, w)
    n = np.linalg.norm(v3D)
    if n<1e-10:
        return w3D
    squaredNorm = np.dot(v3D, v3D)
    x3DFinal = computeGeodesic(x3D,v3D,t)
    v3DFinal = computeGeodesicVelocity(x3D,v3D,t)
    sign = np.sign(np.dot(w3D, np.cross(x3D,v3D)))
    truepw3D = v3DFinal * np.dot(w3D,v3D) /(squaredNorm) + sign*np.sqrt(np.dot(w3D, w3D) - np.dot(w3D,v3D)**2/squaredNorm) * np.cross(x3DFinal, v3DFinal/n)
    return truepw3D

def computeGeodesicVelocity(x3D,v3D,t):
    norm = np.linalg.norm(v3D)
    return -np.sin(t*norm)*x3D*norm + np.cos(t*norm) * v3D

def to2D(x):
    assert len(x)==3, "Not the right dimension of input !"
    assert(np.linalg.norm(x) -1 )<=1e-4, "Not of norm 1 %f" % (np.linalg.norm(x))
    phi = np.arctan(x[1]/x[0])
    if (x[0]<=0):
        phi += math.pi
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

#VERIFIED
def toSphericalVector(x3D,v3D):
    x2D = to2D(x3D)
    theta = x2D[0]
    phi = x2D[1]
    M = np.array([[np.cos(theta)*np.cos(phi),-np.sin(phi)* np.sin(theta),  np.cos(phi)*np.sin(theta)],
                  [np.cos(theta)*np.sin(phi), np.sin(theta)* np.cos(phi), np.sin(phi)*np.sin(theta)],
                  [-1.*np.sin(theta), 0 , np.cos(theta)]])
    invM = linalg.inv(M)
    vSpherical = np.matmul(invM, v3D)
    thetaCoord = vSpherical[0]
    phiCoord = vSpherical[1]
    assert abs(vSpherical[2]) < 1e-10, "Watch out, it does not seem to be tangent to the sphere : %f" % vSpherical[2]
    return np.array([thetaCoord, phiCoord])

def VerifyCoordinates():
    for i in range(1000):
        pos = np.random.rand(2)*math.pi
        v = np.random.rand(2) * 10.
        pos3D = localChartTo3D(pos)
        v3D = chartVelocityTo3D(pos, v)
        vrebuilt = toSphericalVector(pos3D, v3D)
        posrebuilt = to2D(pos3D)
        assert(np.linalg.norm(pos - posrebuilt))<=1e-10, ":o"
        if np.linalg.norm(vrebuilt - v)>=1e-14:
            print(v)
            print(vrebuilt)
            print("")

def inverseProblemLine(x,y):
    #Get the latitude longitudes
    xl, yl = toLatituteLongitude(x), toLatituteLongitude(y)
    geod = Geodesic(1., 0.)
    g = geod.InverseLine(xl[0], xl[1], yl[0], yl[1])
    return g

def getDistanceAndLog(x, y, x3D, y3D, verbose=False):
    w = np.cross(x3D, y3D)
    v = -1.*np.cross(x3D, w)
    norm = np.linalg.norm(v)
    normalizedV = v/norm
    mine = 10.
    optimalT = 0.
    xl, yl = toLatituteLongitude(x), toLatituteLongitude(y)
    geod = Geodesic(1., 0.)
    g = geod.InverseLine(xl[0], xl[1], yl[0], yl[1])
    distance = g.s13
    for t in np.linspace(0., 2*math.pi, 1000):
        d = np.linalg.norm(computeGeodesic(x3D, normalizedV, distance) - y3D)
        if d <mine:
            mine = d
            optimalT = t
    if verbose:
        print("mine", mine, "optimalT", optimalT)
        print("distance:",distance)
    sphericalNormalizedV = toSphericalVector(x3D, normalizedV)
    out = sphericalNormalizedV * distance
    return out

def metric(x,v,w):
    return w[0]*v[0] + np.sin(x[0])**2. * w[1]*v[1]

def SchildsLadder(x,v,w,number_of_time_steps, verbose = False, factor = 1.):
    x3D = localChartTo3D(x)
    v3D = chartVelocityTo3D(x, v)
    dimension = 2 #Dimension of the manifold
    delta = 1./number_of_time_steps
    #To store the computed values of trajectory and transport
    pwtraj = np.zeros((number_of_time_steps+1, 2))
    xtraj = np.zeros((number_of_time_steps+1, 3))
    for i in range(number_of_time_steps+1):
        xtraj[i] = computeGeodesic(x3D, v3D, delta * i)
    pwtraj[0] = w
    time  = 0.
    for k in range(number_of_time_steps):
        # Get P0, P1
        P03D = xtraj[k]
        P13D = xtraj[k+1]
        P0 = to2D(P03D)
        P1 = to2D(P13D)
        # Compute the first geodesic to find P2, from P0 with initial tangent vector wk
        P23D = computeGeodesic(P03D, chartVelocityTo3D(P0, pwtraj[k]), delta*factor)
        P2 = to2D(P23D)
        # Compute the geodesic linking P1 to P2 and its midpoint P3
        invLine = inverseProblemLine(P2, P1)
        pos = invLine.Position(invLine.s13/2., Geodesic.STANDARD)
        P3Latitude = pos['lat2']
        P3Longitude = pos['lon2']
        P3 = toSpherical([P3Latitude, P3Longitude])
        # Compute the geodesic linking P0 to P3 and go twice further to get P4,
        invLine = inverseProblemLine(P0, P3)
        pos = invLine.Position(invLine.s13 * 2.)
        P4Latitude = pos['lat2']
        P4Longitude = pos['lon2']
        P4 = toSpherical([P4Latitude, P4Longitude])
        P43D = localChartTo3D(P4)
        #wk+1 is the riemannian logarithm of the geodesic connecting P1 to P4.
        v = getDistanceAndLog(P1, P4, P13D, P43D, verbose = verbose)
        pwtraj[k+1] = v/delta
    return xtraj, pwtraj

def GetErrors():
    #Initial conditions
    # x = [math.pi/2.+1.5,0.8]
    # v = np.array([1., -1.])
    # w = v
    # vortho = np.array([-v[1], v[0]/np.sin(x[0])**2])
    # w = v + vortho
    x = [math.pi/4,0.]
    v = np.array([2.9616, 1.4810])/2.
    alpha = [ 1.4808 ,  0.37025]
    vortho = np.array([-v[1], v[0]/np.sin(x[0])**2])
    w=[alpha[1], -alpha[0]]
    #3D equivalents
    x3D = localChartTo3D(x)
    v3D = chartVelocityTo3D(x, v)
    w3D = chartVelocityTo3D(x, w)
    x3DFinal = computeGeodesic(x3D,v3D,1.)
    pw3D = trueParallelTransport(x,v,w,1.)
    #Steps and corresponding errors
    errors = []
    nb = [i for i in range(10,200,3)]
    inverseNb = [1./elt for elt in nb]
    print("Real transport :", pw3D)
    for step in nb:
        xtraj3D, pwtraj = SchildsLadder(x,v,w,step, verbose = False)
        pwestimate3D = chartVelocityTo3D(to2D(xtraj3D[-1]), pwtraj[-1])
        errors.append(np.linalg.norm(pwestimate3D - pw3D)/np.linalg.norm(w))
        print("")
        print("Predicted previous time step :", chartVelocityTo3D(to2D(xtraj3D[-2]), pwtraj[-2]))
        print("Error :", np.linalg.norm(pwestimate3D - pw3D)/np.linalg.norm(w), "Steps :", step, "Predicted : ", pwestimate3D)
    return nb, errors


def ErrorAsFunctionOfDelta():
    x = [math.pi/2.+1.,5.1]
    x3D = localChartTo3D(x)
    v = [0., 1.]
    v3D = chartVelocityTo3D(x, v)
    w = [1.,0.]
    w3D = chartVelocityTo3D(x, w)

    n = np.linalg.norm(v3D)
    x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
    pw3D = trueParallelTransport(x, v, w, 1.)

    nbSteps = 50
    factors = np.linspace(0.0001,0.0002,50)
    errors = []
    for fact in factors:
        xtraj, pwtraj = SchildsLadder(x,v,w,nbSteps,factor=fact)
        xtraj3D = np.array([localChartTo3D(elt) for elt in xtraj])
        pwtraj3D = np.array([chartVelocityTo3D(xtraj[i], pwtraj[i]) for i in range(len(xtraj))])
        errors.append(np.linalg.norm(pwtraj3D[-1] - truepw3D)/np.linalg.norm(pw3D))
        # print("true pw :", truepw3D)
        # print("Estimated :", chartVelocityTo3D(to2D(x3DFinal), pwtraj[-1]))
        print("Error :", errors[-1], "factor :", fact, "Predicted : ", pwtraj3D[-1])
    return factors, errors


nb, errors = GetErrors()
np.save("errorSchild",errors)
st = [1./elt for elt in nb]
plt.plot(st, errors)
plt.xlim(xmin = 0)
plt.ylim(ymin = 0)
# plt.savefig("Graphs/Schildserror.pdf")
plt.show()
