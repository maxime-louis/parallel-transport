# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

def render_sphere(xtraj, alphatraj, pwtraj):
    xtraj3D = np.array([localChartTo3D(elt) for elt in xtraj])
    pwtraj3D = np.array([chartVelocityTo3D(xtraj[i], pwtraj[i]) for i in xrange(len(xtraj))])
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    xx, yy, zz = np.hsplit(xtraj3D, 3)
    #number of vectors we represent :
    nbPoints = 20
    pwtraj3DForPlot = np.array([pwtraj3D[int(len(pwtraj3D)*i*1./20)] for i in range(20)])
    xtraj3DForPlot = np.array([xtraj3D[int(len(xtraj3D)*i*1./20)] for i in range(20)])
    xvec, yvec, zvec = np.hsplit(xtraj3DForPlot, 3)
    u, v, w = np.hsplit(pwtraj3DForPlot, 3)

    #Set colours and render
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.scatter(xx,yy,zz,color="k",s=20)
    ax.quiver(xvec, yvec ,zvec , u, v, w,length = 0.1, pivot = "tail")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
