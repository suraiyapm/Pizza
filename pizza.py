import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

print("X: \n ",X)
print("Y: \n ",Y)