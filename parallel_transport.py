# -*- coding: utf-8 -*-
import numpy as np
import render_sphere as rs
import matplotlib.pyplot as plt

#For the sphere x = [theta, phi]

def norm(x,w):
    # print(w)
    return w[0]**2 + np.sin(x[0])**2. * w[1]**2.

def metric(x,v,w):
    return w[0]*v[0] + np.sin(x[0])**2. * w[1]*v[1]

def co_vector_from_vector(x, w):
  #Here for the sphere :
  metric = [[1.,0.], [0., np.sin(x[0])**2.]]
  return np.matmul(metric, w)

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

def plot_norm(x, w):
    norms = [norm(x[i], w[i]) for i in range(len(xtraj))]
    # print(norms[:100])
    plt.plot(norms)
    plt.show()


x = np.array([0.5,-1.])
x3D = rs.localChartTo3D(x)
direction = np.array([1,0.5])
v = np.pi * direction /np.sqrt(np.dot(direction,co_vector_from_vector(x,direction)))
alpha = co_vector_from_vector(x, v)
w = [alpha[1], -alpha[0]]

w3D = rs.chartVelocityTo3D(x, w)
v3D = rs.chartVelocityTo3D(x,v)

#true endpoint and parallel vector
n = np.linalg.norm(v3D)
x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

proj = np.dot(v3D,w3D)/np.dot(v3D, v3D)
projOrtho = np.dot(np.cross(x3D,v3D), w3D)/np.dot(v3D,v3D)
truepw3D = proj * v3DFinal + projOrtho*np.cross(x3D, v3D)

nb_steps = []
errors = []
for i in xrange(1,2):
    nbSteps = i*3000
    xtraj, alphatraj, pwtraj = parallel_transport(x, alpha, w, nbSteps)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    print("Estimated : ", pwtraj3D, "Exact :", truepw3D)
    errors.append(np.linalg.norm(pwtraj3D - truepw3D))
    nb_steps.append(nbSteps)


# plt.plot(nb_steps, errors)
# plt.savefig("equivalent with norm correction.pdf")
# plt.show()


    # plot_norm(xtraj, pwtraj)
    rs.render_sphere(xtraj, alphatraj, pwtraj)
