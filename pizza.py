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
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
plt.plot (X, Y, "bo")
plt.show()

def predict(X, w, b=0):
    return X * w + b
# Our prediction function! This is our whole training phase.
# For now, leaving the bias parameter out of our prediction.

y_hat = predict(20, 2.1, 0)
print("num pizzas: ", y_hat)
# Forty-two! This is the number of pizzas estimated/predicted for 20 reservations.

# Training algorithm spat out a "weight", which is our slope. Use predict (equation of line) to predict num pizzas for 20 reservations.
# 2.1 was the slope of the line Prof drew, visually, himself. Estimation, not final, just for testing. Pretending we trained the model.

# Today's machine learning algorithm does not scale once we have too many variables (weights, which are slope).
# Becomes an NP problem (too big to solve) ->theoretical comp. sci. topic! <3

# Iterative approach: guess, try, improve.
# LOSS: Distance between points and points on the line, aiming to minimize the loss.
# When the weight is zero, starting at the X axis.

