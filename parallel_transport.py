# -*- coding: utf-8 -*-
import numpy as np
import render_sphere as rs
import matplotlib.pyplot as plt
import math
from scipy import linalg
from sklearn import datasets, linear_model
#For the sphere x = [theta, phi]

#colors : peru, royalblue, brown, green, yellow

def squaredNorm(x,w):
    # print(w)
    return w[0]**2 + np.sin(x[0])**2. * w[1]**2.

def to2DVector(x3D,v3D):
    x2D = to2D(x3D)
    theta = x2D[0]
    phi = x2D[1]
    M = np.array([[np.cos(theta)*np.cos(phi),-np.sin(phi)* np.sin(theta),  np.cos(phi)*np.sin(theta)],
                  [np.cos(theta)*np.sin(phi), np.sin(theta)* np.cos(phi), np.sin(phi)*np.sin(theta)],
                  [-1.*np.sin(theta), 0 , np.cos(theta)]])
    invM = linalg.inv(M)
    vSpherical = np.matmul(invM, v3D)
    thetaCoord = vSpherical[0]
    phiCoord = vSpherical[1]
    assert abs(vSpherical[2]) < 1e-5, "Watch out, it does not seem to be tangent to the sphere : %f" % abs(vSpherical[2])
    return np.array([thetaCoord, phiCoord])

def to2D(x):
    assert len(x)==3, "Not the right dimension of input !"
    assert(np.linalg.norm(x) -1 )<=1e-4, "Not of norm 1 %f" % (np.linalg.norm(x))
    phi = np.arctan(x[1]/x[0])
    if (x[0]<=0):
        phi += math.pi
    theta = np.arccos(x[2])#Lies between 0 and pi : ok.
    return np.array([theta,phi])

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

def trueParallelTransport(x,v,w,t):
    v3D = rs.chartVelocityTo3D(x, v)
    x3D = rs.localChartTo3D(x)
    w3D = rs.chartVelocityTo3D(x, w)
    n = np.linalg.norm(v3D)
    if n<1e-10:
        return w3D
    squaredNorm = np.dot(v3D, v3D)
    x3DFinal = computeGeodesic(x,v,t)
    v3DFinal = computeGeodesicVelocity(x,v,t)
    print(np.dot(w3D, w3D) - np.dot(w3D,v3D)/squaredNorm)
    sign = np.sign(np.dot(w3D, np.cross(x3D,v3D)))
    truepw3D = v3DFinal * np.dot(w3D,v3D) /(squaredNorm) + sign*np.sqrt(np.dot(w3D, w3D) - np.dot(w3D,v3D)**2/squaredNorm) * np.cross(x3DFinal, v3DFinal/n)
    return truepw3D

def computeGeodesic(x,v,t):
    x3D = rs.localChartTo3D(x)
    v3D = rs.chartVelocityTo3D(x,v)
    norm = np.linalg.norm(v3D)
    return np.cos(t*norm)*x3D + np.sin(t*norm) * v3D/norm

def computeGeodesicVelocity(x,v,t):
    x3D = rs.localChartTo3D(x)
    v3D = rs.chartVelocityTo3D(x,v)
    norm = np.linalg.norm(v3D)
    return -np.sin(t*norm)*x3D*norm + np.cos(t*norm) * v3D

def RK2Step(x, alpha, epsilon):
    RK_Steps = [0.5,1.]
    xOut = x
    alphaOut = alpha
    for step in RK_Steps:
        Fx, Falpha = hamiltonian_equation(xOut, alphaOut)
        xOut = x + step * epsilon * Fx
        alphaOut = alpha + step * epsilon * Falpha
    return xOut, alphaOut

def RK4Step(x, alpha, epsilon):
    k1, l1 = hamiltonian_equation(x, alpha)
    k2, l2 = hamiltonian_equation(x + epsilon/2. * k1, alpha + epsilon/2. * l1)
    k3, l3 = hamiltonian_equation(x + epsilon/2. * k2, alpha + epsilon/2. * l2)
    k4, l4 = hamiltonian_equation(x + epsilon * k3, alpha + epsilon * l3)
    xOut = x + epsilon * (k1 + 2*(k2+k3) + k4)/6. # Formula for RK 4
    alphaOut = alpha + epsilon * (l1 + 2*(l2+l3) + l4)/6. # Formula for RK 4
    return xOut, alphaOut

def parallel_transport_double_RK2(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = squaredNorm(x,w)
    initialCrossProductWithVelocity = metric(x,v,w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        xcurr, alphacurr = RK2Step(xcurr, alphacurr, epsilon)
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        perturbations = [1,-1]
        Weights = [0.5, -0.5]
        Jacobi = np.zeros(2)
        for i, pert in enumerate(perturbations):
            xPerturbed, _ = RK2Step(xtraj[k], alphatraj[k] + pert * epsilon * betacurr, epsilon)
            Jacobi = Jacobi + Weights[i] * xPerturbed
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop#np.sqrt((initialSquaredNorm/normProp)) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_single_RK2(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = squaredNorm(x,w)
    initialCrossProductWithVelocity = metric(x,v,w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        xcurr, alphacurr = RK2Step(xcurr, alphacurr, epsilon)
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        Jacobi = np.zeros(2)
        xPerturbed, _ = RK2Step(xtraj[k], alphatraj[k] + epsilon * betacurr, epsilon)
        Jacobi = (xPerturbed - xcurr)/epsilon
        prop = Jacobi / (delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop# np.sqrt((initialSquaredNorm/normProp)) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_single_RK4(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = squaredNorm(x,w)
    initialCrossProductWithVelocity = metric(x,v,w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        xcurr, alphacurr = RK4Step(xcurr, alphacurr, epsilon)
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        Jacobi = np.zeros(2)
        xPerturbed, _ = RK4Step(xtraj[k], alphatraj[k] + epsilon * betacurr, epsilon)
        Jacobi = (xPerturbed - xcurr)/epsilon
        prop = Jacobi / (delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop# np.sqrt((initialSquaredNorm/normProp)) * prop
        xtraj[k+1] = xcurr
        alphatraj[k+1] = alphacurr;
    return xtraj, alphatraj, pwtraj

def parallel_transport_double_RK4(x, alpha, w, number_of_time_steps):
    dimension = len(x) #Dimension of the manifold
    delta = 1./number_of_time_steps
    epsilon = delta
    xtraj = np.zeros((number_of_time_steps+1, dimension))
    pwtraj = np.zeros((number_of_time_steps+1, dimension))
    alphatraj = np.zeros((number_of_time_steps+1, dimension))
    xtraj[0] = x
    perturbations = [1,-1]
    Weights = [0.5, -0.5]
    alphatraj[0] = alpha
    pwtraj[0] = w
    initialSquaredNorm = squaredNorm(x,w)
    initialCrossProductWithVelocity = metric(x,v,w)
    for k in range(number_of_time_steps):
        xcurr = xtraj[k]
        alphacurr = alphatraj[k]
        xcurr, alphacurr = RK2Step(xcurr, alphacurr, epsilon)
        betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
        Jacobi = np.zeros(2)
        #For each perturbation, compute the perturbed geodesic
        for i, pert in enumerate(perturbations):
            xPerturbed,_ = RK4Step(xtraj[k], alphatraj[k] + pert * epsilon * betacurr, epsilon)
            Jacobi = Jacobi + Weights[i] * xPerturbed
        prop = Jacobi / (epsilon * delta)
        normProp = metric(xcurr, prop, prop)
        pwtraj[k+1] = prop #np.sqrt(initialSquaredNorm/normProp) * prop
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
    nbForFit = [[elt] for elt in np.linspace(0,0.02,100)]
    plt.plot(nbForFit, regr.predict(nbForFit), color=color)


x = [math.pi/4,0.]
v = np.array([2.9616, 1.4810])/2.
alpha = [ 1.4808 ,  0.37025]
vortho = np.array([-v[1], v[0]/np.sin(x[0])**2])
w=[alpha[1], -alpha[0]]
w = vortho + v
# x = [math.pi/4,0.]
# v = np.array([2.1, 1.4810])/2.
# alpha = co_vector_from_vector(x,v)
# print("alpha",alpha)
# vortho = np.array([-v[1], v[0]/np.sin(x[0])**2])
# w=[alpha[1], -alpha[0]]

v3D = rs.chartVelocityTo3D(x, v)
x3D = rs.localChartTo3D(x)
w3D = rs.chartVelocityTo3D(x, w)


#true end point, end velocity and parallel transport
x3DFinal = computeGeodesic(x,v,1.)
v3DFinal = computeGeodesicVelocity(x,v,1.)
pw3D = trueParallelTransport(x,v,w,1.)

print(x3D)
print("w3D", w3D)
print("v3DInitial", v3D, "v3DFinal", v3DFinal)
print("x3DInitial", x3D, "x3DFinal", x3DFinal)
print("initial w :", w3D, "pw :", pw3D)
steps = [i for i in range(10,200,3)]
# steps = [int(i) for i in np.linspace(10,100)]
nb = [1./elt for elt in steps]

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK2(x, alpha, w, step)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D)/100.)
    # print("Error Single geodesic RK2 :",errors[-1], "Steps :", step)

# np.save("errorFan",errors)
color="royalblue"
plt.scatter(nb, errors, alpha=0.7, color=color, label = "One perturbed geodesic, Runge-Kutta 2")
FitLinear(nb, errors, color)

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_single_RK4(x, alpha, w, step)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D)/100.)
    # print("Error RK2 with single geodesic :",errors[-1], "Steps :", step)

color="red"
plt.scatter(nb, errors, alpha=0.7, color=color, label = "One perturbed geodesic, Runge-Kutta 4")
FitLinear(nb, errors, color)

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_double_RK2(x, alpha, w, step)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D)/100.)
    # print("Error RK4:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)

color="brown"
plt.scatter(nb, errors, alpha=0.7, color=color, label = "Two perturbed geodesics, Runge-Kutta 2")
FitLinear(nb, errors, color)

errors = []
for step in steps:
    xtraj, alphatraj, pwtraj = parallel_transport_double_RK4(x, alpha, w, step)
    pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
    errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D)/100.)
    # print("Error RK1:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)
color="green"
plt.scatter(nb, errors, alpha=0.7, color=color, label = "Two perturbed geodesics, Runge-Kutta 4")
FitLinear(nb, errors, color)

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_RK1Geodesic(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D))
#     print("Error RK1 geodesic:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)
# color="orange"
# plt.scatter(nb, errors, alpha=0.7, color=color, label = "Runge-Kutta 1 for the main geodesic")
# FitLinear(nb, errors, color)

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_RK1Jacobi(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(pw3D))
#     print("Error RK1 Jacobi:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)
# color="black"
# plt.scatter(nb, errors, alpha=0.7, color=color, label = "Runge-Kutta 1 for Jacobi")
# FitLinear(nb, errors, color)

# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(w))
#     print("Error RK2 and five-point method:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)
# color="peru"
# plt.scatter(nb, errors, alpha=0.7, color=color, label = "Third order method for the differentiation of J")
# FitLinear(nb, errors, color)
# errors = []
# for step in steps:
#     xtraj, alphatraj, pwtraj = parallel_transport_order3Jacobi_RK4(x, alpha, w, step)
#     pwtraj3D = rs.chartVelocityTo3D(xtraj[-1], pwtraj[-1])
#     errors.append(np.linalg.norm(pwtraj3D-pw3D)/np.linalg.norm(w))
#     print("Error RK4 and five point:",errors[-1], "Steps :", step, "predicted :", pwtraj3D)
# color="green"
# plt.scatter(nb, errors, alpha=0.7, color=color, label = "Runge-Kutta 4 for perturbed geodesics and third order method for J")
# FitLinear(nb, errors, color)

plt.legend(loc='upper left', prop={'size':23})
plt.xlim(xmin=0)
plt.ylim([0,1e-3])
plt.ylabel("Relative error (%)", fontsize=28)
plt.xlabel("Length of time steps", fontsize=28)
# plt.savefig("/Users/maxime.louis/Documents/Paper Parallel transport/figures/ErrorsSPD.pdf")
plt.show()
















###############OLD VERSION : ################################








#
# def parallel_transport_RK1(x, alpha, w, number_of_time_steps, order=2):
#     dimension = len(x) #Dimension of the manifold
#     delta = 1./number_of_time_steps
#     epsilon = delta
#     xtraj = np.zeros((number_of_time_steps+1, dimension))
#     pwtraj = np.zeros((number_of_time_steps+1, dimension))
#     alphatraj = np.zeros((number_of_time_steps+1, dimension))
#     xtraj[0] = x
#     alphatraj[0] = alpha
#     pwtraj[0] = w
#     initialSquaredNorm = squaredNorm(x,w)
#     initialCrossProductWithVelocity = metric(x,v,w)
#     time  = 0.
#     for k in range(number_of_time_steps):
#         xcurr = xtraj[k]
#         alphacurr = alphatraj[k]
#         Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
#         xcurr = xcurr + delta * Fx
#         alphacurr = alphacurr + delta * Falpha
#         betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
#         perturbations = [1,-1]
#         Weights = [0.5, -0.5]
#         Jacobi = np.zeros(2)
#         for i, pert in enumerate(perturbations):
#             alphaPk = alphatraj[k] + pert * epsilon * betacurr
#             alphaPerturbed = alphaPk
#             xPerturbed = xtraj[k]
#             Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
#             xPerturbed = xPerturbed + delta * Fx
#             alphaPerturbed = alphaPerturbed + delta  * Falpha
#             Jacobi = Jacobi + Weights[i] * xPerturbed
#         prop = Jacobi / (epsilon * delta)
#         normProp = metric(xcurr, prop, prop)
#         pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
#         xtraj[k+1] = xcurr
#         alphatraj[k+1] = alphacurr;
#     return xtraj, alphatraj, pwtraj
#
# def parallel_transport_RK1Geodesic(x, alpha, w, number_of_time_steps):
#     dimension = len(x) #Dimension of the manifold
#     delta = 1./number_of_time_steps
#     epsilon = delta
#     xtraj = np.zeros((number_of_time_steps+1, dimension))
#     pwtraj = np.zeros((number_of_time_steps+1, dimension))
#     alphatraj = np.zeros((number_of_time_steps+1, dimension))
#     xtraj[0] = x
#     alphatraj[0] = alpha
#     pwtraj[0] = w
#     initialSquaredNorm = squaredNorm(x,w)
#     initialCrossProductWithVelocity = metric(x,v,w)
#     RK_Steps = [0.5,1.]
#     for k in range(number_of_time_steps):
#         xcurr = xtraj[k]
#         alphacurr = alphatraj[k]
#         Fx, Falpha = hamiltonian_equation(xcurr, alphacurr)
#         xcurr = xcurr + Fx * delta
#         alphacurr = alphacurr + Falpha * delta
#         betacurr = co_vector_from_vector(xtraj[k], pwtraj[k])
#         perturbations = [1,-1]
#         Weights = [0.5, -0.5]
#         Jacobi = np.zeros(2)
#         for i, pert in enumerate(perturbations):
#             alphaPk = alphatraj[k] + pert * epsilon * betacurr
#             alphaPerturbed = alphaPk
#             xPerturbed = xtraj[k]
#             for step in RK_Steps:
#                 Fx, Falpha = hamiltonian_equation(xPerturbed, alphaPerturbed)
#                 xPerturbed = xtraj[k] + step * delta * Fx
#                 alphaPerturbed = alphaPk + step * delta * Falpha
#             Jacobi = Jacobi + Weights[i] * xPerturbed
#         prop = Jacobi / (epsilon * delta)
#         normProp = metric(xcurr, prop, prop)
#         pwtraj[k+1] = np.sqrt(initialSquaredNorm/normProp) * prop
#         xtraj[k+1] = xcurr
#         alphatraj[k+1] = alphacurr;
#     return xtraj, alphatraj, pwtraj
