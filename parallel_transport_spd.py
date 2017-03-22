# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
n = 3
dimSym = n*(n+1)/2
corresp = {}



corresp = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]

#Converts a symmetric matrix into its representation in the E basis
def flatten(x):
    a,b = x.shape
    assert (a==n),"Wrong dimension"
    assert (b==n),"Wrong dimension"
    dimSym = n*(n+1)/2
    xFlattened = np.zeros(dimSym)
    for i,(a,b) in enumerate(corresp):
        xFlattened[i] = x[a,b]
    return xFlattened

#Reconstruct a symmetric matrix from its E representation
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

#Evaluate the spd metric at the point sigmaFlat with tangent vectors vFlat and wFlat
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

#Get the metric matrix.
def getMetricMatrix(xFlat):
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            if (g[j,i] != 0):
                g[i,j] = g[j,i]
            else:
                g[i,j] = metric(xFlat, EFlat[i], EFlat[j])
    return g

#Compute the geodesic, analytically.
def trueGeodesic(x0, v0, t):
    sqrtX0 = linalg.sqrtm(x0)
    sqrtinverseX0 = linalg.sqrtm(linalg.inv(x0))
    ex = linalg.expm(t * np.matmul(np.matmul(sqrtinverseX0, v0),sqrtinverseX0))
    out = np.matmul(np.matmul(sqrtX0, ex), sqrtX0)
    return out

#Compute analytical parallel transport
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

#Geodesic evolution equation
def hamiltonian_equation(xFlat, alphaFlat, g, inverseG):
    assert (np.linalg.norm(np.matmul(g,inverseG) - np.eye(6))<1e-10), "Not the right inverse"
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

#Verify the gradient of the inverse metric.
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

#Takes vectors as input, expressed in the E basis. VERIFIED.
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
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5,1.]
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        metr = getMetricMatrix(xcurr)
        velocity = vector_from_co_vector(xcurr, alphacurr, metr)
        # print "parallel vector norm :", metric(xcurr, pwtraj[k], pwtraj[k])
        # print "velocity norm :", metric(xcurr, velocity, velocity)
        #Compute the position of the next point on the geodesic
        for step in RK_Steps:
            met = getMetricMatrix(xcurr)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
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
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr
    return xtraj, alphatraj, pwtraj

def parallel_transport_conservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    initialNormSquared = metric(x, w, w)
    initialScalarProduct = metric(x,w,initialVelocity)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5,1.]
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        metr = getMetricMatrix(xcurr)
        velocity = vector_from_co_vector(xcurr, alphacurr, metr)
        # print "Scalar product with velocity :", metric(xcurr, velocity, pwtraj[k])
        # print "w norm :", metric(xcurr, pwtraj[k], pwtraj[k])
        for step in RK_Steps:
            met = getMetricMatrix(xcurr)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
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
        #Here we add a step to enforce the conservation.
        metr = getMetricMatrix(xcurr)
        uncorrectedEstimate = Jacobi/(epsilon*delta)
        currVelocity = vector_from_co_vector(xcurr, alphacurr, metr)
        currVelocityNormSquared = metric(xcurr, currVelocity, currVelocity)
        currwNormSquared = metric(xcurr, uncorrectedEstimate, uncorrectedEstimate)
        currScalarProd = metric(xcurr, currVelocity, uncorrectedEstimate)
        p = np.zeros(3)
        p[0] = (currScalarProd - 2 * currVelocityNormSquared + currwNormSquared * currVelocityNormSquared**2/currScalarProd**2)
        p[1] = 2*initialScalarProduct*(1 - currVelocityNormSquared * currwNormSquared/(currScalarProd**2))
        p[2] = initialScalarProduct**2*currwNormSquared/currScalarProd**2 - initialNormSquared
        roots = np.roots(p)
        alpha = np.min(roots)
        beta = (initialScalarProduct-alpha * currVelocityNormSquared)/currScalarProd
        pwtraj[k+1] = alpha * currVelocity + beta * uncorrectedEstimate
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr
    return xtraj, alphatraj, pwtraj

#VERIFIED
def parallel_transport_RK4(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        met = getMetricMatrix(xcurr)
        k1, l1 = hamiltonian_equation(xcurr, alphacurr, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon/2.*k1)
        k2, l2 = hamiltonian_equation(xcurr + epsilon/2. * k1, alphacurr + epsilon/2. * l1, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon/2.*k2)
        k3, l3 = hamiltonian_equation(xcurr + epsilon/2. * k2, alphacurr + epsilon/2. * l2, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon*k3)
        k4, l4 = hamiltonian_equation(xcurr + epsilon * k3, alphacurr + epsilon * l3, met, linalg.inv(met))
        xcurr = xcurr + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
        alphacurr = alphacurr + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            met = getMetricMatrix(xPerturbed)
            k1, l1 = hamiltonian_equation(xPerturbed, alphaPerturbed, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon/2.*k1)
            k2, l2 = hamiltonian_equation(xPerturbed + epsilon/2. * k1, alphaPerturbed + epsilon/2. * l1, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon/2.*k2)
            k3, l3 = hamiltonian_equation(xPerturbed + epsilon/2. * k2, alphaPerturbed + epsilon/2. * l2, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon*k3)
            k4, l4 = hamiltonian_equation(xPerturbed + epsilon * k3, alphaPerturbed + epsilon * l3, met, linalg.inv(met))
            xPerturbed = xPerturbed + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

#VERIFIED
def parallel_transport_order3Jacobi(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5,1.]
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        velocity = vector_from_co_vector(xcurr, alphacurr, getMetricMatrix(xcurr))
        # print "velocity norm :", metric(xcurr, velocity, velocity)
        # print "w norm :", metric(xcurr, pwtraj[k], pwtraj[k])
        # print "scalar product :", metric(xcurr, pwtraj[k], velocity)
        for step in RK_Steps:
            met = getMetricMatrix(xcurr)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [-2,-1,1,2]
        xP = []
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
            xP.append(xPerturbed)
            #Update the estimate
        Jacobi = 1./12.*(xP[0]-8*xP[1]+8*xP[2]-xP[3])
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj
#VERIFIED
def parallel_transport_order3Jacobi_RK4(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    initialSquaredNorm = metric(x, w, w)
    alphatraj[0] = alpha
    pwtraj[0] = w
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        met = getMetricMatrix(xcurr)
        k1, l1 = hamiltonian_equation(xcurr, alphacurr, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon/2.*k1)
        k2, l2 = hamiltonian_equation(xcurr + epsilon/2. * k1, alphacurr + epsilon/2. * l1, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon/2.*k2)
        k3, l3 = hamiltonian_equation(xcurr + epsilon/2. * k2, alphacurr + epsilon/2. * l2, met, linalg.inv(met))
        met = getMetricMatrix(xcurr + epsilon*k3)
        k4, l4 = hamiltonian_equation(xcurr + epsilon * k3, alphacurr + epsilon * l3, met, linalg.inv(met))
        xcurr = xcurr + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
        alphacurr = alphacurr + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [-2,-1,1,2]
        xP = []
        #For each perturbation, compute the perturbed geodesic
        for pert in perturbations:
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            met = getMetricMatrix(xPerturbed)
            k1, l1 = hamiltonian_equation(xPerturbed, alphaPerturbed, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon/2.*k1)
            k2, l2 = hamiltonian_equation(xPerturbed + epsilon/2. * k1, alphaPerturbed + epsilon/2. * l1, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon/2.*k2)
            k3, l3 = hamiltonian_equation(xPerturbed + epsilon/2. * k2, alphaPerturbed + epsilon/2. * l2, met, linalg.inv(met))
            met = getMetricMatrix(xPerturbed + epsilon*k3)
            k4, l4 = hamiltonian_equation(xPerturbed + epsilon * k3, alphaPerturbed + epsilon * l3, met, linalg.inv(met))
            xPerturbed = xPerturbed + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
            xP.append(xPerturbed)
            #Update the estimate
        Jacobi = 1./12.*(xP[0]-8*xP[1]+8*xP[2]-xP[3])
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_RK1(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        met = getMetricMatrix(xcurr)
        metInverse = linalg.inv(met)
        Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
        xcurr = xtraj[k] +  delta * Fx
        alphacurr = alphatraj[k] +  delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            met = getMetricMatrix(xPerturbed)
            Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed, met, linalg.inv(met))
            xPerturbed = xtraj[k] +  delta * Fx
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

#VERIFIED, marche alors qu'il ne devrait pas marcher !
def parallel_transport_RK1Geodesic(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5,1.]
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        met = getMetricMatrix(xtraj[k])
        metInverse = linalg.inv(met)
        Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
        xcurr = xtraj[k] +  delta * Fx
        alphacurr = alphatraj[k] +  delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
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
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

#VERIFIED
def parallel_transport_RK1Jacobi(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    initialG = getMetricMatrix(x)
    initialVelocity = vector_from_co_vector(x, alpha, initialG)
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    RK_Steps = [0.5,1.]
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        for step in RK_Steps:
            met = getMetricMatrix(xcurr)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr, met, metInverse)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            met = getMetricMatrix(xPerturbed)
            metInverse = linalg.inv(met)
            Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed, met, metInverse)
            xPerturbed = xtraj[k] + step * delta * Fx
            alphaPerturbed = alphaPk + step * delta * Falpha
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def FitLinear(nb, errors, color):
    regr = linear_model.LinearRegression(fit_intercept=True)
    nbForFit = [[elt] for elt in nb[-30:]]
    errorsForFit = [[elt] for elt in errors[-30:]]
    regr.fit(nbForFit, errorsForFit)
    print("regression coefficients :", regr.coef_, regr.intercept_)
    assert(regr.intercept_ < 1e-1), "Does not seem to converge !"
    nbForFit = [[elt] for elt in np.linspace(0,0.005,100)]
    plt.plot(nbForFit, regr.predict(nbForFit), color=color)

w0 = np.array([[1.,0.2,0],[0.2,1,0],[0,0,0]])
x0 = np.eye(3)
print "x0", x0
v0 = np.array([[1,1,0],[1,0,0],[0,0,0]])/5.
orthov0 = np.array([[0,0,0],[0,0,0],[0,0,1]])
colors = ['b','g','r','c','k','y']
ws  = [orthov0 + v0]#, 5*v0]
# for i in range(3):
#     ws.append(generateRandomSymmetric())
#     print(ws[-1])
errrrors = []
sps = []
for i,w0 in enumerate(ws):
    pFinal = trueGeodesic(x0, v0, 1.)
    wFinal = trueParallelTransport(x0, v0, w0)
    #Get the flat versions
    x0Flat, v0Flat, wFlat = flatten(x0), flatten(v0), flatten(w0)
    #get the initial momentum
    initialMetric = getMetricMatrix(x0Flat)
    alpha = co_vector_from_vector(x0Flat, v0Flat, initialMetric)
    steps = [int(elt)*3 for elt in np.linspace(3,180,10)]
    nb = [1./elt for elt in steps]
    errors = []
    sp = metric(x0Flat, v0Flat, wFlat)/np.sqrt(metric(x0Flat, wFlat, wFlat)*metric(x0Flat, v0Flat, v0Flat))
    sps.append(sp)
    print("sp", sp)
    for step in steps:
        xtraj, alphatraj, pwtraj = parallel_transport(x0Flat, alpha, wFlat, step)
        errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
        print "RK2",errors[-1], "Steps", step
    errrrors.append(errors)
np.save("Data2SingleRenorm/RK2traj", xtraj)
np.save("Data2SingleRenorm/RK2alphatraj", alphatraj)
np.save("Data2SingleRenorm/RK2pwtraj", pwtraj)
np.save("Data2SingleRenorm/RK2errors", np.array(errors))
np.save("Data2SingleRenorm/RK2steps", steps)
color="royalblue"
nb = [1./elt for elt in np.load("Data2SingleRenorm/RK2steps.npy")]
errors = np.load("Data2SingleRenorm/RK2errors.npy")
print(errors)
plt.scatter(nb, errors, alpha=0.4, color=color, label="Runge-Kutta 2")
FitLinear(nb, errors, color)
for i,err in enumerate(errrrors):
    plt.scatter(nb, errrrors[i], alpha=0.7, color=colors[i], label = "Runge-Kutta 2 " + str(sps[i])[:5])

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_RK1(x0Flat, alpha, wFlat, step)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print("RK1",errors[-1])
np.save("Data2SingleRenorm/RK1traj", xtraj)
np.save("Data2SingleRenorm/RK1alphatraj", alphatraj)
np.save("Data2SingleRenorm/RK1pwtraj", pwtraj)
np.save("Data2SingleRenorm/RK1errors", np.array(errors))
np.save("Data2SingleRenorm/RK1steps", steps)
errors = np.load("Data2SingleRenorm/RK1errors.npy")
plt.scatter(nb, errors, alpha=0.7, color="green", label = "Runge-Kutta 1")

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_RK4(x0Flat, alpha, wFlat, step)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "RK4", errors[-1]
np.save("Data2SingleRenorm/RK4traj", xtraj)
np.save("Data2SingleRenorm/RK4alphatraj", alphatraj)
np.save("Data2SingleRenorm/RK4pwtraj", pwtraj)
np.save("Data2SingleRenorm/RK4errors", np.array(errors))
np.save("Data2SingleRenorm/RK4steps", steps)
color="brown"
errors = np.load("Data2SingleRenorm/RK4errors.npy")
plt.scatter(nb, errors, alpha=0.7, color=color, label = "Runge-Kutta 4")
FitLinear(nb, errors, color)
errors = []
for nbSteps in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi(x0Flat, alpha, wFlat, nbSteps)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "five point method", errors[-1]
np.save("Data2SingleRenorm/order3RK2traj", xtraj)
np.save("Data2SingleRenorm/order3RK2alphatraj", alphatraj)
np.save("Data2SingleRenorm/order3RK2pwtraj", pwtraj)
np.save("Data2SingleRenorm/order3RK2errors", np.array(errors))
np.save("Data2SingleRenorm/order3RK2steps", steps)
color="peru"
errors = np.load("Data2SingleRenorm/order3RK2errors.npy")
plt.scatter(nb, errors, alpha=0.7, color=color, label = "Five Point Method and Runge-Kutta 2")
FitLinear(nb, errors, color)

errors = []
for nbSteps in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi_RK4(x0Flat, alpha, wFlat, nbSteps)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "RK4 with 5 points : ", errors[-1]
np.save("Data2SingleRenorm/order3RK4traj", xtraj)
np.save("Data2SingleRenorm/order3RK4alphatraj", alphatraj)
np.save("Data2SingleRenorm/order3RK4pwtraj", pwtraj)
np.save("Data2SingleRenorm/order3RK4errors", np.array(errors))
np.save("Data2SingleRenorm/order3RK4steps", steps)
color="green"
errors = np.load("Data2SingleRenorm/order3RK4errors.npy")
plt.scatter(nb, errors, alpha=0.7, color="green", label = "Five Point Method and Runge-Kutta 4")
FitLinear(nb, errors, color)
errors = []
for nbSteps in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_RK1Geodesic(x0Flat, alpha, wFlat, nbSteps)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "RK1 for the geodesic only : ", errors[-1]
np.save("Data2SingleRenorm/RK1Geodesictraj", xtraj)
np.save("Data2SingleRenorm/RK1Geodesicalphatraj", alphatraj)
np.save("Data2SingleRenorm/RK1Geodesicpwtraj", pwtraj)
np.save("Data2SingleRenorm/RK1Geodesicerrors", np.array(errors))
np.save("Data2SingleRenorm/RK1Geodesicsteps", steps)
errors = np.load("Data2SingleRenorm/RK1Geodesicerrors.npy")
plt.scatter(nb, errors, alpha=0.7, color="orange", label = "Runge-Kutta 1 for the main geodesic")

errors = []
for nbSteps in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_RK1Jacobi(x0Flat, alpha, wFlat, nbSteps)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "RK1 for jacobi only : ", errors[-1]
np.save("Data2SingleRenorm/RK1jacobitraj", xtraj)
np.save("Data2SingleRenorm/RK1jacobialphatraj", alphatraj)
np.save("Data2SingleRenorm/RK1jacobipwtraj", pwtraj)
np.save("Data2SingleRenorm/RK1jacobierrors", np.array(errors))
np.save("Data2SingleRenorm/RK1jacobisteps", steps)
errors = np.load("Data2SingleRenorm/RK1jacobierrors.npy")
plt.scatter(nb, errors, alpha=0.7, color="yellow", label = "RK1 jacobi")

errors = []
for nbSteps in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_conservation(x0Flat, alpha, wFlat, nbSteps)
    errors.append(np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0))
    print "Conservation : ", errors[-1], "\n"
np.save("Data2SingleRenorm/Conservationxtraj", xtraj)
np.save("Data2SingleRenorm/Conservationalphatraj", alphatraj)
np.save("Data2SingleRenorm/Conservationpwtraj", pwtraj)
np.save("Data2SingleRenorm/Conservationerrors", np.array(errors))
np.save("Data2SingleRenorm/Conservationsteps", steps)
color="red"
errors = np.load("Data2SingleRenorm/Conservationerrors.npy")
plt.scatter(nb, errors, alpha=0.7, color=color, label = "Enforcing conservations")
FitLinear(nb, errors, color)
plt.xlabel("1/N")
plt.legend(loc='upper left', prop={'size':12})
plt.xlim(xmin=0)
plt.ylim(ymin=0)
# plt.savefig("/Users/maxime.louis/Documents/Paper Parallel transport/figures/ErrorsSPD.pdf")
plt.show()
