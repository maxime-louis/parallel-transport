# -*- coding: utf-8 -*-
import numpy as np
import render_sphere as rs
import matplotlib.pyplot as plt

#For the sphere x = [theta, phi]

#colors : peru, royalblue, brown, green, yellow

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

def compute_geodesic(x, alpha, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    #To store the computed values of trajectory and transport
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    #initialisation
    xtraj[0] = x
    alphatraj[0] = alpha
    RK_Steps = [0.5,1.]
    time  = 0.
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        time = time + delta
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        xcurr = xcurr + delta * vector_from_co_vector(xcurr, alphacurr)
        #Co-vector of w_k : g^{ab} w_b
        betacurr = co_vector_from_vector(xtraj[k], alphacurr)
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
        xtraj[k+1] = xcurr
        alphatraj[k+1] = Jacobi / (epsilon * delta);
    return xtraj, alphatraj

x = np.array([1.,0.8])
x3D = rs.localChartTo3D(x)
direction = np.array([0.,1.])
v = np.pi * direction /np.sqrt(np.dot(direction,co_vector_from_vector(x,direction)))
alpha = co_vector_from_vector(x, v)
v3D = rs.chartVelocityTo3D(x,v)

#true endpoint and parallel vector
n = np.linalg.norm(v3D)
x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

errors = []
steps = [elt * 50 for elt in range(30,200)]
nb = [1./elt for elt in steps]
for step in steps:
    xtraj, alphatraj = compute_geodesic(x, alpha, 100000)
    errors.append(np.linalg.norm(rs.localChartTo3D(xtraj[-1]) - x3DFinal))
    print "Error when self-transporting :", errors[-1]

# plt.scatter(nb, errors, alpha=0.7, color="royalblue", label = "Self-Transporting")
# plt.legend()
# plt.ylim(ymin=0)
# plt.xlim(xmin=0)
# plt.show()
