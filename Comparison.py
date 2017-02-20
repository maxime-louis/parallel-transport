# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


errorSchild = np.load("errorSchild.npy")
errorFan = np.load("errorFan.npy")
print(errorSchild, errorFan)

steps = [i for i in range(10,200,3)]

nb = [1./elt for elt in steps]

plt.scatter(nb, errorFan, alpha=0.7, color="royalblue", label = "Fanning Scheme")
plt.scatter(nb, errorSchild, alpha=0.7, color="green", label = "Schild's ladder")
plt.xlabel("1/N")
plt.legend(loc='upper left', prop={'size':12})
plt.show()
