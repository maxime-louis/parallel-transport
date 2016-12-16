# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

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

#A basis of the symmetric matrices (could be written more generically...)
E0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
E1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
E2 = np.array([[0,0,0],[0,0,0],[0,0,1]])
E3 = np.array([[0,1,0],[1,0,0],[0,0,0]])
E4 = np.array([[0,0,1],[0,0,0],[1,0,0]])
E5 = np.array([[0,0,0],[0,0,1],[0,1,0]])
E = [E0,E1,E2,E3,E4,E5]
EFlat = [flatten(elt) for elt in E]

def isPositiveDefinite(m):
    for i in range(n):
        E = np.zeros(n)
        E[i] = 1
        if (np.dot(E, np.matmul(m, E))) < 1e-15:
            return False
    return True

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
    out = np.trace(fac3)
    return out

def getMetricMatrix(xFlat):
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(i, dimSym):
            g[i,j] = metric(xFlat, EFlat[i], EFlat[j])
    for i in xrange(dimSym):
        for j in xrange(0, i):
            g[i,j] = g[j,i]
    return g

def trueGeodesic(p0, v0):
    sqrtP0 = linalg.sqrtm(p0)
    sqrtinverseP0 = linalg.sqrtm(linalg.inv(p0))
    ex = linalg.expm(np.matmul(np.matmul(sqrtinverseP0, v0),sqrtinverseP0))
    out = np.matmul(np.matmul(sqrtP0, ex), sqrtP0)
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
    #We evaluate the metric first:
    inverseG = linalg.inv(g)
    wFlat = np.matmul(inverseG, alphaFlat)
    return wFlat

#Returns te kth component (which is a vector) of the gradient of the inverse of the metric
def gradientInverseMetric(k, xFlat, g, inverseMetric):
    xMat = reconstruct(xFlat)
    xMatInverse = linalg.inv(xMat)
    #The derivative of the metric with respect to the kth coordinate
    kMetricGradient = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            kDerivativeXMatInverse = - 1. * np.matmul(np.matmul(xMatInverse, E[k]), xMatInverse)
            fac1 = np.matmul(np.matmul(kDerivativeXMatInverse, E[i]), np.matmul(xMatInverse, E[j]))
            fac2 = np.matmul(np.matmul(xMatInverse, E[i]), np.matmul(kDerivativeXMatInverse, E[j]))
            #This is the variation of the ij-th coord of the metric with respect to the kth coordinate
            kMetricGradient[i,j] = np.trace(fac1 + fac2)
    #The variation of the inverse of the metric with respect to the kth coordinate
    kInverseMetricGradient = -1. * np.matmul(np.matmul(inverseMetric, kMetricGradient), inverseMetric)
    return kInverseMetricGradient


def hamiltonian_equation(xFlat, alphaFlat):
    g = getMetricMatrix(xFlat)
    if (abs(np.linalg.det(g))<1e-10):
        print "Determinant of the metric is :", np.linalg.det(g), "So the metric does not look invertible at this point :", xFlat
    inverseG = linalg.inv(g)
    Fx = np.matmul(inverseG, alphaFlat)
    Falpha = np.zeros(dimSym)
    for k in xrange(dimSym):
        Falpha[k] = - 0.5 * np.dot(alphaFlat, np.matmul(gradientInverseMetric(k,xFlat, g, inverseG), alphaFlat))
    return Fx, Falpha

def checkGradient():
    epsilon = 1e-4
    x = np.random.rand(6)
    g = getMetricMatrix(x)
    inverseG = linalg.inv(g)
    for k in xrange(dimSym):
        analytic = gradientInverseMetric(k, x, g, inverseG)
        estimated = (linalg.inv(getMetricMatrix(x + epsilon * EFlat[k])) - linalg.inv(getMetricMatrix(x - epsilon * EFlat[k])))/(2*epsilon)
        for i in xrange(6):
            for j in xrange(6):
                if (analytic[i,j]>1e-4 or estimated[i,j]>1e-4):
                    print analytic[i,j], estimated[i,j]



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
    initialNorm = np.sqrt(metric(x,w,w))
    initialCrossProductWithVelocity = metric(x,initialVelocity,w)
    RK_Steps = [0.5, 1]
    time  = 0.
    print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    print "InitialNorm :", initialNorm
    print "initial cross product with velocity :", initialCrossProductWithVelocity
    for k in range(number_of_time_steps):
        time = time + delta
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        g = getMetricMatrix(xcurr)
        print("Hamiltonian value :", 0.5 * np.dot(alphacurr, np.matmul(g, alphacurr)))

        print "step :",k, "squared norm : ", metric(xcurr, pwtraj[k], pwtraj[k]), "wcurr :", pwtraj[k]
        velocity = vector_from_co_vector(xcurr, alphacurr, g)
        print "scalar product with velocity :", metric(xcurr, velocity, pwtraj[k])
        assert isPositiveDefinite(reconstruct(xcurr)), "Matrix is not positive definite !"

        #Compute the position of the next point on the geodesic
        for i,step in enumerate(RK_Steps):
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
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
                Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
                xPerturbed = xtraj[k] + step * delta * Fx
                alphaPerturbed = alphaPk + step * delta * Falpha
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj


checkGradient()
#
# p0 = generateRandomSPD()
# v0 = generateRandomSymmetric()/10.
# w = generateRandomSymmetric()
#
# #Get the flat versions
# p0Flat = flatten(p0)
# v0Flat = flatten(v0)
# wFlat = flatten(w)
# g = getMetricMatrix(p0Flat)
# alphaFlat = co_vector_from_vector(p0Flat, v0Flat, g)
#
# pFinal = trueGeodesic(p0, v0)
# wFinal = trueParallelTransport(p0, v0, w)
#
#
#
# xtraj, alphatraj, pwtraj = parallel_transport(p0Flat, v0Flat, wFlat, 300)
# print wFinal
# print pwtraj[-1]
