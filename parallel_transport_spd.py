# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print("Computing on ", num_cores, " cores")

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

def RK2Step(x, alpha, epsilon):
    xOut = x
    alphaOut = alpha
    RK_Steps = [0.5,1.]
    for step in RK_Steps:
        met = getMetricMatrix(xOut)
        metInverse = linalg.inv(met)
        Fx, Falpha = hamiltonian_equation(xOut, alphaOut, met, metInverse)
        xOut = x + epsilon * step * Fx
        alphaOut = alpha + epsilon * step * Falpha
    return xOut, alphaOut

def RK4Step(x, alpha, epsilon):
    xOut = x
    alphaOut = alpha
    met = getMetricMatrix(x)
    k1, l1 = hamiltonian_equation(xOut, alphaOut, met, linalg.inv(met))
    met = getMetricMatrix(xOut + epsilon/2.*k1)
    k2, l2 = hamiltonian_equation(xOut + epsilon/2. * k1, alphaOut + epsilon/2. * l1, met, linalg.inv(met))
    met = getMetricMatrix(xOut + epsilon/2.*k2)
    k3, l3 = hamiltonian_equation(xOut + epsilon/2. * k2, alphaOut + epsilon/2. * l2, met, linalg.inv(met))
    met = getMetricMatrix(xOut + epsilon*k3)
    k4, l4 = hamiltonian_equation(xOut + epsilon * k3, alphaOut + epsilon * l3, met, linalg.inv(met))
    xOut = x + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
    alphaOut = alpha + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
    return xOut, alphaOut

#Takes vectors as input, expressed in the E basis. VERIFIED.
def parallel_transport_double_RK2_noConservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK2Step(xtraj[k], alphatraj[k], epsilon)
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            xPerturbed, alphaPerturbed = RK2Step(xtraj[k], alphatraj[k] + pert * epsilon * betacurr, epsilon)
            Jacobi = Jacobi + Weights[i] * xPerturbed
        prop = Jacobi / (epsilon * h)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop#np.sqrt(initialSquaredNorm/normProp) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr
    return xtraj, alphatraj, pwtraj

#Takes vectors as input, expressed in the E basis. VERIFIED.
def parallel_transport_single_RK2_noConservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK2Step(xtraj[k], alphatraj[k], epsilon)
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        xPerturbed, alphaPerturbed = RK2Step(xtraj[k], alphatraj[k] + epsilon * betacurr, epsilon)
        #Update the estimate
        Jacobi = (xPerturbed - xcurr)/epsilon
        prop = Jacobi / (h)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr
    return xtraj, alphatraj, pwtraj

#Takes vectors as input, expressed in the E basis. VERIFIED.
def parallel_transport_single_RK4_conservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK4Step(xtraj[k], alphatraj[k], epsilon)
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        xPerturbed, alphaPerturbed = RK4Step(xtraj[k], alphatraj[k] + epsilon * betacurr, epsilon)
        #Update the estimate
        Jacobi = (xPerturbed - xcurr)/epsilon
        metr = getMetricMatrix(xcurr)
        uncorrectedEstimate = Jacobi/(epsilon*h)
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

#Takes vectors as input, expressed in the E basis. VERIFIED.
def parallel_transport_single_RK2_conservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK2Step(xtraj[k], alphatraj[k], epsilon)
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        xPerturbed, alphaPerturbed = RK2Step(xtraj[k], alphatraj[k] + epsilon * betacurr, epsilon)
        #Update the estimate
        Jacobi = (xPerturbed - xcurr)/epsilon
        #Here we add a step to enforce the conservation.
        metr = getMetricMatrix(xcurr)
        uncorrectedEstimate = Jacobi/(epsilon*h)
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


def parallel_transport_double_RK2_conservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK2Step(xtraj[k], alphatraj[k], epsilon)
        #Co-vector of w_k : g^{ab} w_b
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            xPerturbed, alphaPerturbed = RK2Step(xtraj[k], alphatraj[k] + pert * epsilon * betacurr, epsilon)
            Jacobi = Jacobi + Weights[i] * xPerturbed
        #Here we add a step to enforce the conservation.
        metr = getMetricMatrix(xcurr)
        uncorrectedEstimate = Jacobi/(epsilon*h)
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
def parallel_transport_RK4_conservation(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    h = 1./number_of_time_steps
    epsilon = h
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
    initialSquaredNorm = metric(x, w, w)
    for k in range(number_of_time_steps):
        xcurr, alphacurr = RK4tep(xtraj[k], alphatraj[k], epsilon)
        g = getMetricMatrix(xtraj[k])
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k], g)
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(dimension)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            xPerturbed, alphaPerturbed = RK4Step(xtraj[k], alphatraj[k] + pert * epsilon * betacurr, epsilon)
            Jacobi = Jacobi + Weights[i] * xPerturbed
        metr = getMetricMatrix(xcurr)
        uncorrectedEstimate = Jacobi/(epsilon*h)
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

def FitLinear(nb, errors, color):
    regr = linear_model.LinearRegression(fit_intercept=True)
    nbForFit = [[elt] for elt in nb[-4:]]
    errorsForFit = [[elt] for elt in errors[-4:]]
    regr.fit(nbForFit, errorsForFit)
    print("regression coefficients :", regr.coef_, regr.intercept_)
    assert(regr.intercept_ < 1e-1), "Does not seem to converge !"
    nbForFit = [[elt] for elt in np.linspace(0,0.075,100)]
    plt.plot(nbForFit, regr.predict(nbForFit), color=color)

w0 = np.array([[1.,0.2,0],[0.2,1,0],[0,0,0]])
x0 = np.eye(3)
print "x0", x0
v0 = np.array([[1,1,0],[1,0,0],[0,0,0]])/5.
orthov0 = np.array([[0,0,1.],[0,0,0],[1.,0,3.]])
# colors = ['b','g','r','c','k','y']
w0  = v0+orthov0
pFinal = trueGeodesic(x0, v0, 1.)
wFinal = trueParallelTransport(x0, v0, w0)

pFinal = trueGeodesic(x0, v0, 1.)
wFinal = trueParallelTransport(x0, v0, w0)
#Get the flat versions
x0Flat, v0Flat, wFlat = flatten(x0), flatten(v0), flatten(w0)
#get the initial momentum
initialMetric = getMetricMatrix(x0Flat)
alpha = co_vector_from_vector(x0Flat, v0Flat, initialMetric)

steps = range(5,20)
nb = [1./e for e in steps]
############RK2 Single conservation################
def processInputRK2Conservation(step):
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK2_conservation(x0Flat, alpha, wFlat, step)
    err = np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0)/100.
    print("Error with a single perturbed geodesic and conservation RK2 (%)", err)
    return err

errors = Parallel(n_jobs=num_cores)(delayed(processInputRK2Conservation)(s) for s in steps)
color = "royalblue"
plt.scatter(nb, errors, alpha=0.4, color=color, label="Single perturbed geodesic, Runge-Kutta 2, with conservation")
FitLinear(nb, errors, color)

############RK2 Single no conservation
def processInputRK2NoConservation(step):
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK2_noConservation(x0Flat, alpha, wFlat, step)
    err = np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0)/100.
    print("Error with a single perturbed geodesic and no conservation RK2(%)", err)
    return err

errors = Parallel(n_jobs=num_cores)(delayed(processInputRK2NoConservation)(s) for s in steps)
color = "red"
plt.scatter(nb, errors, alpha=0.4, color=color, label="Single perturbed geodesic, Runge-Kutta 2, without conservation")
FitLinear(nb, errors, color)

############RK4 Single conservation
def processInputRK4Conservation(step):
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK4_conservation(x0Flat, alpha, wFlat, step)
    err = np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0)/100.
    print("Error with a single perturbed geodesic and no conservation RK4 (%)", err)
    return err

errors = Parallel(n_jobs=num_cores)(delayed(processInputRK4Conservation)(s) for s in steps)
color = "orange"
plt.scatter(nb, errors, alpha=0.4, color=color, label="Single perturbed geodesic, Runge-Kutta 4, with conservation")
FitLinear(nb, errors, color)

#############RK2 Double conservation####################
def processInputDoubleRK2Conservation(step):
    xtraj, alphatraj, pwtraj = parallel_transport_double_RK2_conservation(x0Flat, alpha, wFlat, step)
    err = np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0)/100.
    print("Two perturbed geodesics, Runge-Kutta 2, with conservation (%)", err)
    return err

errors = Parallel(n_jobs=num_cores)(delayed(processInputDoubleRK2Conservation)(s) for s in steps)
color = "brown"
plt.scatter(nb, errors, alpha=0.4, color=color, label="Two perturbed geodesics, Runge-Kutta 2, with conservation")
FitLinear(nb, errors, color)

##########RK3 Double conservation################
def processInputDoubleRK4Conservation(step):
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK4_conservation(x0Flat, alpha, wFlat, step)
    err = np.linalg.norm(wFinal - reconstruct(pwtraj[-1]))/np.linalg.norm(w0)/100.
    print("Two perturbed geodesics, Runge-Kutta 4, with conservation (%)", err)
    return err

errors = Parallel(n_jobs=num_cores)(delayed(processInputDoubleRK4Conservation)(s) for s in steps)
color = "green"
plt.scatter(nb, errors, alpha=0.4, color=color, label="Two perturbed geodesics, Runge-Kutta 4, with conservation")
FitLinear(nb, errors, color)



plt.legend(loc='upper left', prop={'size':23})
plt.xlim(xmin=0)
# plt.ylim([0,1e-3])
plt.ylabel("Relative error (%)", fontsize=28)
plt.xlabel("Length of time steps", fontsize=28)
# plt.savefig("/Users/maxime.louis/Documents/Paper Parallel transport/figures/ErrorsSPD.pdf")
plt.show()
