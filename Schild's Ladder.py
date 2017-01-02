# -*- coding: utf-8 -*-
import numpy as np
import render_sphere as rs
import matplotlib.pyplot as plt
from scipy.optimize import *
import math

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
    return np.cos(t*v3DNorm)*x3D + np.sin(t*v3DNorm)*v3D/v3DNorm

def to2D(x):
    assert len(x)==3, "Not the right dimension of input !"
    if abs(np.linalg.norm(x) - 1)>=1e-4:
        print np.linalg.norm(x), "Not of norm 1"
    theta = np.arctan(x[1]/x[0])
    phi = np.arccos(x[2])
    return np.array([phi,theta])

#Returns a, phi0 such that the geodesic cot(theta) = a*cos(phi-phi0) is the great circle we are looking for.
def computeLogarithm(x2D, y2D):
    phi1, theta1 = x2D[0], x2D[1]
    phi2, theta2 = y2D[0], y2D[1]
    def F(param):
        return [1./np.tan(theta1) - param[0] * np.cos(phi1-param[1]), 1./np.tan(theta2) - param[0]*np.cos(phi2-param[1])]
    vs = broyden1(F,np.zeros(2), f_tol = 1e-10)
    print(vs)
    return vs

def getGeodesicFromAPhi(a,phi0,t):
    thetaOut = t
    phiOut = phi0 + np.arccos(1/(a*np.tan(t)))
    return np.array([phiOut, thetaOut])

x = [0.2,0.2]
y = [1.2,1.2]
x3D = localChartTo3D(x)
y3D = localChartTo3D(y)
v = computeLogarithm(x, y)
a, phi0 = v[0], v[1]
theta1 = x[1]
theta2 = y[1]
epsilon = 1e-6
epsilonpos = getGeodesicFromAPhi(a, phi0, theta1 + epsilon)
epsilonneg = getGeodesicFromAPhi(a, phi0, theta1 - epsilon)
print getGeodesicFromAPhi(a, phi0, theta2), y
tangentVector = (epsilonpos - epsilonneg) /(2*epsilon)
tangentVector3D = chartVelocityTo3D(x,tangentVector)
print(tangentVector, tangentVector3D)
# for t in np.linspace(0,1.,10000):
#     print np.linalg.norm(y3D - computeGeodesic(x3D, tangentVector3D, t))



def SchildsLadder(x,v,w,number_of_time_steps):
    pass
    # dimension = len(x) #Dimension of the manifold
    # delta = 1./number_of_time_steps
    # #To store the computed values of trajectory and transport
    # pwtraj = np.zeros((number_of_time_steps+1, dimension))
    # xtraj[0] = x
    # pwtraj[0] = w
    # time  = 0.
    # # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    # for k in range(number_of_time_steps):
        #Get P0, P1
        #Compute the first geodesic to find P2, from P0 with initial tangent vector wk
        #Compute the geodesic linking P1 to P2 and its midpoint P3
        #Compute the geodesic linking P0 to P3 and go twice further to get P4, wk+1 is the riemannian
        #logarithm of the geodesic connecting P1 to P4.
