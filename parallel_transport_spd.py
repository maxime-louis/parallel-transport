# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

n = 3

corresp = {}

#A basis of the symmetric matrices (could be written more generically...)
E0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
E1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
E2 = np.array([[0,0,0],[0,0,0],[0,0,1]])
E3 = np.array([[0,1,0],[1,0,0],[0,0,0]])
E4 = np.array([[0,0,1],[0,0,0],[1,0,0]])
E5 = np.array([[0,0,0],[0,0,1],[0,1,0]])
E = [E0,E1,E2,E3,E4,E5]


corresp = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]

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

def metric(sigmaFlat, vFlat, wFlat):
    sigma = reconstruct(sigmaFlat)
    v = reconstruct(vFlat)
    w = reconstruct(wFlat)
    assert(np.linalg.det(sigma)>1e-10), "Matrix not invertible"
    sqrtSigma = linalg.sqrtm(sigma)
    inverseSigma = linalg.inv(sigma)
    inverseSqrtSigma = linalg.inv(sqrtSigma)
    otherInverseSqrtSigma = linalg.sqrtm(inverseSigma)
    assert (np.linalg.norm(inverseSqrtSigma - otherInverseSqrtSigma)<1e-6), "Inverses differ a lot"
    fac1 = np.matmul(inverseSqrtSigma, np.transpose(v))
    fac2 = np.matmul(inverseSigma, w)
    fac3 = np.matmul(fac1, fac2)
    fac4 = np.matmul(fac3,inverseSqrtSigma)
    out = np.trace(fac4)
    return out

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

def co_vector_from_vector(xFlat, wFlat):
    dimSym = n*(n+1)/2
    #We evaluate the metric first:
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            g[i,j] = metric(xFlat, flatten(E[i]), flatten(E[j]))
    cowFlat = np.matmul(g,wFlat)
    return cowFlat

def vector_from_co_vector(xFlat, alphaFlat):
    dimSym = n*(n+1)/2
    #We evaluate the metric first:
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            g[i,j] = metric(x, E[i], E[j])
    inverseG = linalg.inv(g)
    alphaFlat = flatten(alpha)
    # print(alphaFlat)
    wFlat = np.matmul(inverseG, alphaFlat)
    return wFlat

def hamiltonian_equation(x, alpha):
  pass

#Takes vectors as input, expressed in the E basis.
def parallel_transport(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta

    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    #initialisation
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialNorm = np.sqrt(metric(x,w,w))
    initialCrossProductWithVelocity = metric(x,v,w)
    RK_Steps = [0.5, 1]
    time  = 0.
    print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    print "InitialNorm :", initialNorm
    print "initial cross product with velocity :", initialCrossProductWithVelocity
    for k in range(number_of_time_steps):
        time = time + delta
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]

        #Compute the position of the next point on the geodesic
        for i,step in enumerate(RK_Steps):
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha

        #Co-vector of w_k : g^{ab} w_b
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(2)
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


p0 = generateRandomSPD()
v0 = generateRandomSymmetric()
w = generateRandomSymmetric()

#Get the flat versions
p0Flat = flatten(p0)
v0Flat = flatten(v0)
wFlat = flatten(w)
alphaFlat = co_vector_from_vector(p0Flat, v0Flat)


xtraj, alphatraj, pwtraj = parallel_transport(v0Flat, p0Flat, wFlat, 100)





# print("p0 :", p0)
# print("v0 :", v0)
# print("w :", w)
# print("pExact :", pExact)
# print("wExact :", wExact)
