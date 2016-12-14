# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

n = 3

#For the sphere x = [theta, phi]
def squaredNorm(sigma, v):
    return metric(sigma, v, v)

def matrixIndicestoBasisIndex(i,j):
    if (i>=j):
        return i*n+j
    else:
        return matrixIndicestoBasisIndex(j,i)

def basisIndexToMatrixIndices(a):
    return a/n,a%n

for i in xrange(n*(n+1)/2):
    print(i,basisIndexToMatrixIndices(i))

for a in xrange(n):
    for b in xrange(n):
        print(a, b, matrixIndicestoBasisIndex(a,b))


def metric(sigma, v, w):
    sqrtSigma = linalg.sqrtm(sigma)
    inverseSigma = linalg.inv(sigma)
    inverseSqrtSigma = linalg.inv(sqrtSigma)
    otherInverseSqrtSigma = linalg.sqrtm(inverseSigma)
    assert (np.linalg.norm(inverseSqrtSigma - otherInverseSqrtSigma)<1e-8), "Inverses differ a lot"
    # print("difference of the two inverses : ", np.linalg.norm(inverseSqrtSigma - otherInverseSqrtSigma))
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
    print(ex)
    out = np.matmul(np.matmul(sqrtP0, ex), sqrtP0)
    return out

def trueParallelTransport(p0, v0, w):
    invP0 = linalg.inv(p0)
    ex1 = linalg.expm(0.5 * np.matmul(v0,invP0))
    ex2 = linalg.expm(0.5 * np.matmul(invP0, v0))
    out = np.matmul(np.matmul(ex1, w), ex2)
    return out

def generateRandomInvertible():
    #We take the exponential of a random matrix.
    B = np.random.rand(n,n)
    expB = linalg.expm(B)
    print(np.linalg.det(expB))
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

def co_vector_from_vector(x, w):
    dimSym = n*(n+1)/2 #This is the dimension of Sym(n)
    #We need to evaluate the metric at x first i.e. the g(E^i,E^j) for all i,j
    g = np.zeros((dimSym, dimSym))
    for i in xrange(dimSym):
        for j in xrange(dimSym):
            #This is our choice of ordering :
            a1, b1 = i/n, i%n + i/n-1
            print(a1,b1,i)
            a2, b2 = j/n, j%n + j/n-1
            print(a2,b2,j)
            #This is the ith element of the basis :
            E1 = np.zeros((n,n))
            E1[a1,b1] += 0.5
            #This is the jth element of the basis :
            E2 = np.zeros((n,n))
            E2[a2,b2] += 0.5
            g[i,j] = metric(x, E1, E2)
    #Now that we have the metric, we compute its inverse.
    inverseG = linalg.inv(g)
    #We use the inverse to get the co-vector.
    #First we flatten w :
    wFlattened = np.zeros(dimSym)
    for i in xrange(dimSym):
        a, b = i/n, i%n+i/n-1
        if (i!=j):
            wFlattened[i] = w[a,b] * 2
        else:
            wFlattened[i] = w[a,b]
    #Compute the co-vector in this basis
    coWFlattened = np.matmul(inverseG, wFlattened)
    #Build the corresponding matrix
    coW = np.zeros((n,n))
    for i in xrange(len(coWFlattened)):
        a, b = i/n, i%n+i/n
        print(i,a,b)
        coW[a,b] += coWFlattened[i]
    return coW



def vector_from_co_vector(x, alpha):
    inversemetric = [[1., 0.],[0., 1./np.sin(x[0])**2]]
    return np.matmul(inversemetric, alpha)

def hamiltonian_equation(x, alpha):
  if (abs(np.sin(x[0])) < 1e-20):
    print("error")
    raise ValueError("Cannot handle the poles of the sphere")
  Fx = np.array([alpha[0], alpha[1]/np.sin(x[0])**2]) #this is g_{ab} alpha^b
  Falpha = np.array([np.cos(x[0])/np.sin(x[0])**3. * alpha[1]**2., 0.])
  return Fx, Falpha

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
    initialNorm = np.sqrt(norm(x,w))
    initialCrossProductWithVelocity = metric(x,v,w)
    RK_Steps = [0.5, 1]
    time  = 0.
    print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)

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
            # print(i,Jacobi)
        # print(Jacobi / (epsilon * delta))
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        print(metric(xcurr, vector_from_co_vector(xcurr, alphacurr), pwtraj[k+1]))
        # pwtraj[k+1] = pwtraj[k+1] * initialNorm / np.sqrt(norm(xtraj[k+1], pwtraj[k+1]))
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj


p0 = generateRandomSPD()
v0 = generateRandomSymmetric()
# v0 = np.zeros((n,n))
w = generateRandomSymmetric()


alpha = co_vector_from_vector(p0, v0)
print(alpha)

#True geodesic and parallel transport :
pExact = trueGeodesic(p0, v0)
wExact = trueParallelTransport(p0, v0, w)



print("p0 :", p0)
print("v0 :", v0)
print("w :", w)
print("pExact :", pExact)
print("wExact :", wExact)
