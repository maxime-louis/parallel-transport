# -*- coding: utf-8 -*-
import numpy as np
import render_sphere as rs
import matplotlib.pyplot as plt
import math
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
    RK_Steps = [0.5,1.]
    time  = 0.
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
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

def parallel_transport_RK4(x, alpha, w, number_of_time_steps):
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
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        k1, l1 = hamiltonian_equation(xcurr, alphacurr)
        k2, l2 = hamiltonian_equation(xcurr + epsilon/2. * k1, alphacurr + epsilon/2. * l1)
        k3, l3 = hamiltonian_equation(xcurr + epsilon/2. * k2, alphacurr + epsilon/2. * l2)
        k4, l4 = hamiltonian_equation(xcurr + epsilon * k3, alphacurr + epsilon * l3)
        xcurr = xcurr + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
        alphacurr = alphacurr + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
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
            k1, l1 = hamiltonian_equation(xPerturbed, alphaPerturbed)
            k2, l2 = hamiltonian_equation(xPerturbed + epsilon/2. * k1, alphaPerturbed + epsilon/2. * l1)
            k3, l3 = hamiltonian_equation(xPerturbed + epsilon/2. * k2, alphaPerturbed + epsilon/2. * l2)
            k4, l4 = hamiltonian_equation(xPerturbed + epsilon * k3, alphaPerturbed + epsilon * l3)
            xPerturbed = xPerturbed + epsilon/6. * (k1 + 2*k2 + 2*k3 + k4)
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_order3Jacobi(x, alpha, w, number_of_time_steps):
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
    RK_Steps = [0.5,1.]
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        for step in RK_Steps:
            Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
            xcurr = xtraj[k] + step * delta * Fx
            alphacurr = alphatraj[k] + step * delta * Falpha

        #Co-vector of w_k : g^{ab} w_b
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        perturbations = [-2,-1,1,2]
        xP = []
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            for step in RK_Steps:
                Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
                xPerturbed = xtraj[k] + step * delta * Fx
                alphaPerturbed = alphaPk + step * delta * Falpha
            xP.append(xPerturbed)
        Jacobi = 1./12.*(xP[0]-8*xP[1]+8*xP[2]-xP[3])
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_order3Jacobi_RK4(x, alpha, w, number_of_time_steps, order=2):
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
    RK_Steps = np.linspace(1./order,1.,order)
    print(order, RK_Steps)
    time  = 0.
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        k1, l1 = hamiltonian_equation(xcurr, alphacurr)
        k2, l2 = hamiltonian_equation(xcurr + epsilon/2. * k1, alphacurr + epsilon/2. * l1)
        k3, l3 = hamiltonian_equation(xcurr + epsilon/2. * k2, alphacurr + epsilon/2. * l2)
        k4, l4 = hamiltonian_equation(xcurr + epsilon * k3, alphacurr + epsilon * l3)
        xcurr = xcurr + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
        alphacurr = alphacurr + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
        #Co-vector of w_k : g^{ab} w_b
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        perturbations = [-2,-1,1,2]
        xP = []
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            alphaPk = alphatraj[k] + pert * epsilon * betacurr
            alphaPerturbed = alphaPk
            xPerturbed = xtraj[k]
            k1, l1 = hamiltonian_equation(xPerturbed, alphaPerturbed)
            k2, l2 = hamiltonian_equation(xPerturbed + epsilon/2. * k1, alphaPerturbed + epsilon/2. * l1)
            k3, l3 = hamiltonian_equation(xPerturbed + epsilon/2. * k2, alphaPerturbed + epsilon/2. * l2)
            k4, l4 = hamiltonian_equation(xPerturbed + epsilon * k3, alphaPerturbed + epsilon * l3)
            xPerturbed = xPerturbed + epsilon/6. * (k1 + 2*k2 + 2*k3 + k4)
            xP.append(xPerturbed)
        Jacobi = 1./12.*(xP[0]-8*xP[1]+8*xP[2]-xP[3])
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_RK1(x, alpha, w, number_of_time_steps, order=2):
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
    time  = 0.
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        #Compute the position of the next point on the geodesic
        Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
        xcurr = xcurr + delta * Fx
        alphacurr = alphacurr + delta * Falpha
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
            Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
            xPerturbed = xPerturbed + delta * Fx
            alphaPerturbed = alphaPerturbed + delta  * Falpha
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_RK1Geodesic(x, alpha, w, number_of_time_steps):
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
    RK_Steps = [0.5,1.]
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
        xcurr = xcurr + Fx * delta
        alphacurr = alphacurr + Falpha * delta
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

def parallel_transport_RK1Jacobi(x, alpha, w, number_of_time_steps):
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
    RK_Steps = [0.5,1.]
    time  = 0.
    # print("Lauching computation with epsilon :", epsilon, "delta :", delta, "number_of_time_steps : ", number_of_time_steps)
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
            Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
            xPerturbed = xPerturbed + delta * Fx
            alphaPerturbed = alphaPerturbed + delta * Falpha
            #Update the estimate
            Jacobi = Jacobi + Weights[i] * xPerturbed
        pwtraj[k+1] = Jacobi / (epsilon * delta)
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj


def plot_norm(x, w):
    norms = [norm(x[i], w[i]) for i in range(len(xtraj))]
    # print(norms[:100])
    plt.plot(norms)
    plt.show()

x = [math.pi/2.,5.1]
x3D = rs.localChartTo3D(x)
v = [0., 1.]
v3D = rs.chartVelocityTo3D(x, v)
w = [1.,0.]
w3D = rs.chartVelocityTo3D(x, w)
alpha = co_vector_from_vector(x,v)

# x = np.array([1.,0.8])
# x3D = rs.localChartTo3D(x)
# direction = np.array([0.,1.])
# v = np.pi * direction /np.sqrt(np.dot(direction,co_vector_from_vector(x,direction)))/10.
# alpha = co_vector_from_vector(x, v)
# w = [alpha[1], -alpha[0]]
#
# w3D = rs.chartVelocityTo3D(x, w)
# v3D = rs.chartVelocityTo3D(x,v)

#true endpoint and parallel vector
n = np.linalg.norm(v3D)
x3DFinal = np.cos(n)*x3D + np.sin(n) * v3D/n
v3DFinal = -np.sin(n)* n * x3DFinal + np.cos(n) * v3D

proj = np.dot(v3D,w3D)/np.dot(v3D, v3D)
projOrtho = np.dot(np.cross(x3D,v3D), w3D)/np.dot(v3D,v3D)
truepw3D = proj * v3DFinal + projOrtho*np.cross(x3D, v3D)

steps = [int(i*10) for i in np.linspace(10,100,20)]
# steps = [elt * 5 for elt in range(30,200)]
nb = [1./elt for elt in steps]

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print("Error :",errors[-1], "Steps :", step)
#
# plt.scatter(nb, errors, alpha=0.7, color="royalblue", label = "Runge-Kutta 2")

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_RK4(x, alpha, w, step)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
    # print(np.linalg.norm(rs.localChartTo3D(xtraj[-1])- x3DFinal))
    print("Error:",errors[-1], "Steps :", step)

plt.scatter(nb, errors, alpha=0.7, color="brown", label = "Runge-Kutta 4")
# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_RK1(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print(errors[-1])
#
# plt.scatter(nb, errors, alpha=0.7, color="green", label = "Runge-Kutta 1")

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_RK1Geodesic(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print(errors[-1])
#
# plt.scatter(nb, errors, alpha=0.7, color="orange", label = "Runge-Kutta 1 for the main geodesic")

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_RK1Jacobi(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print(errors[-1])
#
# plt.scatter(nb, errors, alpha=0.7, color="black", label = "Runge-Kutta 1 for Jacobi")


# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print(errors[-1])
#
# plt.scatter(nb, errors, alpha=0.7, color="peru", label = "Five Point Method and Runge-Kutta 2")
#
# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi_RK4(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-truepw3D)/np.linalg.norm(w))
#     print(errors[-1])
# plt.scatter(nb, errors, alpha=0.7, color="yellow", label = "Five Point Method and Runge-Kutta 4")

# plt.xlabel("1/N")
# plt.legend(loc='upper left')
# plt.xlim([0,0.008])
# plt.ylim([0,0.0003])
# plt.savefig("/Users/maxime.louis/Documents/Paper Parallel transport/figures/ErrorsSPD.pdf")
# plt.show()
