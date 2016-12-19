# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

n = 3
dimSym = n*(n+1)/2
corresp = {}



corresp = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]


def flatten(x):
    a,b = x.shape
    assert (a==n),"Wrong dimension"
    assert (b==n),"Wrong dimension"
    dimSym = n*(n+1)/2
    xFlattened = np.zeros(dimSym)
    for i,(a,b) in enumerate(corresp):
        xFlattened[i] = x[a,b]
    return xFlattened

def reconstruct(x):
    assert (len(x) == n*(n+1)/2), "Wrong dimension of flattened array"
    xReconstructed = np.zeros((n,n))
    for i,(a,b) in enumerate(corresp):
        xReconstructed[a,b] = x[i]
        xReconstructed[b,a] = x[i]
    return xReconstructed

#A basis for symmetric matrices (could be written more generically...)
E0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
E1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
E2 = np.array([[0,0,0],[0,0,0],[0,0,1]])
E3 = np.array([[0,1,0],[1,0,0],[0,0,0]])
E4 = np.array([[0,0,1],[0,0,0],[1,0,0]])
E5 = np.array([[0,0,0],[0,0,1],[0,1,0]])
E = [E0,E1,E2,E3,E4,E5]
EFlat = [flatten(elt) for elt in E]

def isPositiveDefinite(m):
    w,_ = np.linalg.eig(m)
    for i in range(n):
        E = np.zeros(n)
        E[i] = 1
        if (np.dot(E, np.matmul(m, E))) < 1e-15:
            return False
    return True,w

def plotMatrix(m):
    assert isPositiveDefinite(m), "Can't plot a non spd matrix"
    w,x = np.linalg.eig(m)

    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

def generateRandomInvertible():
    #We take the exponential of a random matrix.
    B = np.random.rand(n,n)
    expB = linalg.expm(B)
    assert (abs(np.linalg.det(expB))>1e-10),"Generated a non invertible matrix ! !"
    return expB

def generateRandomSPD():
    B = generateRandomInvertible()
    return np.matmul(np.transpose(B),B)

def generateRandomSymmetric():
    m = np.random.rand(n,n)
    for i in range(n):
        for j in range(n):
            if i<=j:
                m[i,j] = m[j,i]
    return m

def metric(sigmaFlat, vFlat, wFlat):
    sigma = reconstruct(sigmaFlat)
    v = reconstruct(vFlat)
    w = reconstruct(wFlat)
    if (abs(np.linalg.det(sigma))<1e-10):
        print "Determinant is :", np.linalg.det(sigma), "So the matrix does not look invertible :", sigma
    sqrtSigma = linalg.sqrtm(sigma)
    inverseSigma = linalg.inv(sigma)
    fac1 = np.matmul(inverseSigma, v)
    fac2 = np.matmul(inverseSigma, w)
    fac3 = np.matmul(fac1, fac2)
    out = 0.5*np.trace(fac3)
    return out

def getMetricMatrix(xFlat):
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            if (g[j,i] != 0):
                g[i,j] = g[j,i]
            else:
                g[i,j] = metric(xFlat, EFlat[i], EFlat[j])
    return g

def trueGeodesic(x0, v0, t):
    sqrtX0 = linalg.sqrtm(x0)
    sqrtinverseX0 = linalg.sqrtm(linalg.inv(x0))
    ex = linalg.expm(t * np.matmul(np.matmul(sqrtinverseX0, v0),sqrtinverseX0))
    out = np.matmul(np.matmul(sqrtX0, ex), sqrtX0)
    return out

def trueParallelTransport(p0, v0, w):
    invP0 = linalg.inv(p0)
    ex1 = linalg.expm(0.5 * np.matmul(v0,invP0))
    ex2 = linalg.expm(0.5 * np.matmul(invP0, v0))
    out = np.matmul(np.matmul(ex1, w), ex2)
    return out

def co_vector_from_vector(xFlat, wFlat, g):
    cowFlat = np.matmul(g, wFlat)
    return cowFlat

def vector_from_co_vector(xFlat, alphaFlat, g):
    inverseG = linalg.inv(g)
    wFlat = np.matmul(inverseG, alphaFlat)
    return wFlat

#Returns the kth component (which is a matrix) of the gradient of the inverse of the metric
def gradientInverseMetric(k, xMat, xMatInverse, g, inverseMetric):
    #The derivative of the metric with respect to the kth coordinate
    kMetricGradient = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            kDerivativeXMatInverse = - 1. * np.matmul(np.matmul(xMatInverse, E[k]), xMatInverse)
            fac1 = np.matmul(np.matmul(kDerivativeXMatInverse, E[i]), np.matmul(xMatInverse, E[j]))
            fac2 = np.matmul(np.matmul(xMatInverse, E[i]), np.matmul(kDerivativeXMatInverse, E[j]))
            #This is the variation of the ij-th coord of the metric with respect to the kth coordinate
            kMetricGradient[i,j] = 0.5* np.trace(fac1 + fac2)
    #The variation of the inverse of the metric with respect to the kth coordinate
    kInverseMetricGradient = -1.* np.matmul(np.matmul(inverseMetric, kMetricGradient), inverseMetric)
    return kInverseMetricGradient

def hamiltonian_equation(xFlat, alphaFlat, g, inverseG):
    assert (np.linalg.norm(np.matmul(g,inverseG) - np.eye(6))<1e-6), "Not the right inverse"
    xMat = reconstruct(xFlat)
    xMatInverse = linalg.inv(xMat)
    det = np.linalg.det(g)
    Fx = np.matmul(inverseG, alphaFlat)
    Falpha = np.zeros(dimSym)
    for k in xrange(dimSym):
        #This is the derivative of the inverse of the metric with respect to the k-th coordinate.
        aux = np.matmul(gradientInverseMetric(k, xMat, xMatInverse, g, inverseG), alphaFlat)
        Falpha[k] = - 0.5 * np.dot(alphaFlat, aux)
    return Fx, Falpha

def checkGradient():
    epsilon = 1e-4
    x = np.random.rand(6)
    xMat = reconstruct(x)
    xMatInverse = linalg.inv(xMat)
    g = getMetricMatrix(x)
    inverseG = linalg.inv(g)
    currentMax = 0.
    for k in xrange(dimSym):
        analytic = gradientInverseMetric(k, xMat, xMatInverse,g, inverseG)
        estimated = (linalg.inv(getMetricMatrix(x + epsilon * EFlat[k])) - linalg.inv(getMetricMatrix(x - epsilon * EFlat[k])))/(2*epsilon)
        for i in xrange(6):
            for j in xrange(6):
                print(analytic[i,j])
                if abs(analytic[i,j]-estimated[i,j]) > currentMax:
                    currentMax = analytic[i,j] - estimated[i,j]
    print "Maximum difference found for gradient of the inverse metric :", currentMax

def checkAnalyticalGeodesic(x0, v0):
    epsilon = 1e-6
    x1 = trueGeodesic(x0, v0, epsilon)
    x2 = trueGeodesic(x0,v0, 2*epsilon)
    v1 = (x1-x0)/epsilon
    v2 = (x2-x1)/epsilon
    g = getMetricMatrix(flatten(x0))
    alpha1 = co_vector_from_vector(flatten(x0), flatten(v1), g)
    g = getMetricMatrix(flatten(x1))
    alpha2 = co_vector_from_vector(flatten(x1), flatten(v2), g)
    alphadot = (alpha2-alpha1)/epsilon
    # _,Falpha = hamiltonian_equation(x0, )



hamilts = []

#Takes vectors as input, expressed in the E basis.
def parallel_transport(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    #initialisation
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5, 1]
    time  = 0.
    print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        time = time + delta
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        g = getMetricMatrix(xcurr)
        det = np.linalg.det(g)
        if (det<1e-12):
            print "The metric does not look invertible :", det
        invg = linalg.inv(g)
        velocity = vector_from_co_vector(xcurr, alphacurr, g)
        # if (k % 20 == 0):
            # print ""
            # print "step :",k, "squared norm : ", metric(xcurr, pwtraj[k], pwtraj[k]), "wcurr :", pwtraj[k]
            # print "Iteration : ", k
            # print "Hamiltonian value :", 0.5 * np.dot(alphacurr, np.matmul(invg, alphacurr))
            # print "Velocity norm :", metric(xcurr, velocity, velocity)
            # print "Scalar product velocity|transported : ", metric(xcurr, pwtraj[k], velocity)
        b, eigs = isPositiveDefinite(reconstruct(xcurr))
        if not(b):
            print "Matrix is not positive definite ! :", eigs
        hamilts.append(0.5 * np.dot(alphacurr, np.matmul(invg, alphacurr)))
        #Compute the position of the next point on the geodesic
        for step in RK_Steps:
            met = getMetricMatrix(xcurr)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            for step in RK_Steps:
                met = getMetricMatrix(xPerturbed)
                metInverse = linalg.inv(met)
                Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed, met, metInverse)
                xPerturbed = xtraj[k] + step * delta * Fx
                alphaPerturbed = alphaPk + step * delta * Falpha
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        pAnalytic = trueGeodesic(x0, v0, time)
        if (i%20==0):
            print "Error : ", np.linalg.norm(xcurr - flatten(pAnalytic))
        g = getMetricMatrix(xcurr)
        velocity = vector_from_co_vector(xcurr, alphacurr, g)
        # print "Velocity :", metric(xcurr, velocity, velocity)
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj


x0 = np.eye(3)
v0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
w = generateRandomSymmetric()



#Get the flat versions
x0Flat = flatten(x0)
v0Flat = flatten(v0)
wFlat = flatten(w)


g = getMetricMatrix(x0Flat)
alpha = co_vector_from_vector(x0Flat, v0Flat, g)

assert isPositiveDefinite(x0)
print "Initial point : ", x0
print "Initial velocity : ", v0Flat


pFinal = trueGeodesic(x0, v0, 1.)
wFinal = trueParallelTransport(x0, v0, w)

for i in xrange(1,3):
    xtraj, alphatraj, pwtraj = parallel_transport(x0Flat, alpha, wFlat, 100*i)


    print "Final point estimation :\n ", reconstruct(xtraj[-1])
    print "Analytic point estimation :\n ", pFinal
    print "Final transported vector estimation :\n ", reconstruct(pwtraj[-1])
    print "Analytic Estimation : \n", wFinal
    print "L2 error :", np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))



    # plt.plot(hamilts)
    # plt.show()
