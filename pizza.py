import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

print("X: \n ",X)
print("Y: \n ",Y)

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations". fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
plt.plot (X, Y, "bo")
plt.show()