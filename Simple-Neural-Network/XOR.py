import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from network import train
from network import predict
from activations import Tanh
from losses import mse_prime 
from losses import mse


X = np.reshape([[0, 0],
                [0, 1],
                [1, 0],  # Reshaping 4 x 2 matrix -> 4 x 2 x 1 matrix.
                [1, 1]], (4, 2, 1))  # Reshaping the array to column vector (n x 1).

Y = np.reshape([[0],
                [1],
                [1],    # Reshaping 4 x 1 matrix -> 4 x 1 x 1 matrix.
                [0]], (4, 1, 1))  # Reshaping the array to match the dimension of the above matrix.


network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

train(network, X, Y, mse, mse_prime, verbose=True)


def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))

choice = "y"
while True:
    if choice.lower() != "n":
        x, y = float(input("Enter x: ")), float(input("Enter y: "))
        print(round(predict(network,[[x], [y]])[0,0]))
    else:
        break
    choice = input("Press any key to continue[y/n]: ")


# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()