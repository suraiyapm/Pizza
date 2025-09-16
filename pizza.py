import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# lol the name of this Python file is not a joke made by me, but by prof, very wholesome
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
# The output of predict is (y-hat) y with ^ on toppa it, idk how to type that

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

# Calculating the error: Guess weight, test with our labeled training data, error is the difference between y and y_hat.
# We don't want positives and negatives, so (rather than abs. value) we will SQUARE the data (makes it stronger + normalizes data).
# Our loss function will be the average of all of these (guess a weight W, plug in X values to test, etc)
# So add all numbers together, divide by inputs, gets us the MEAN ERROR. Ex, guessing "2". 49, 144, 16 then div by 3.
# Want to find a LOSS function to determine MEAN SQUARED ERROR for our guess.
# Average of SQUARED differences (Y and Y_hat) is our MEAN SQUARED ERROR.

def loss(X, Y, w, b=0):
    return np.average((predict(X,w,b)-Y)**2)
# Another name for Y_hat: predict(X,w,b). For all the Xs, take the average of Y
