import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# Below consist function of Gaussian Smoothing and 
# Perona - Malik Diffusion

# TODO: Complete the rest of code
# REVISE: Still incomplete

# Generate the x value and the axis interval
x = np.linspace(-3, 3, 1001)
# x = np.linspace(0, 50, 100)

# Determine the function for Gaussian in 2D space
def f1d(x : np.ndarray , p : np.ndarray):
    y = np.zeros(x.shape)           # Initial value for y
    p = np.array([2, 1, 1.9, 2.35, 0.1])    # Gaussian Point
    for point in p:
        y = np.copy(y) + 5 * np.exp(-6 * np.power(x - point, 2))
        # y = np.exp(1.0 + np.power(x, 0.5) - np.exp(x / 15.0)) + np.random.normal(scale=1.0, size=x.shape)
    return y

# Determine the function for Gaussian in 3D space
def f2d(x : np.ndarray, y : np.ndarray):
    z = np.zeros((x.shape[0], y.shape[0]))      # z is initial value for x and y
    p = np.array([[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]]) # Gaussian Point
    for point in p:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                z[i][j] = z[i][j] + 5 * np.exp(-6 * (np.power(x[i] - point[0], 2) + np.power(y[j] - point[1], 2)))
    return z

def add_noise(x : np.ndarray):
    result = np.copy(x) + np.random.rand(x.shape[0]) * 0.2 - 0.1
    return result


def test_1d_gaussian():
    y = f1d(x, np.array([0.25, 2.25, 1.9, 1.25, 0.1])) # y axis
    y1 = gaussian_filter(y, sigma=10)


# Visualize the ouput result
fig = plt.figure()

ax = fig.add_subplot(2, 3, 1)
ax1 = fig.add_subplot(2, 3, 2)
ax2 = fig.add_subplot(2, 3, 3)
ax3 = fig.add_subplot(2, 3, 4)
ax4 = fig.add_subplot(2, 3, 5)
ax5 = fig.add_subplot(2, 3, 6)

ax.plot(x, y)
ax.set_title('sigma = 0')
ax1.plot(x, y1)
ax1.set_title('sigma = 50')
ax2.plot(x, y2)
ax2.set_title('sigma = 100')
ax3.plot(x, y3)
ax3.set_title('sigma = 200')
ax4.plot(x, y4)
ax4.set_title('sigma = 500')
ax5.plot(x, y5)
ax5.set_title('sigma = 1000')

test_1d_gaussian()
# Execute with Noisy data
# z = np.exp(1.0 + np.power(x, 0.5) - np.exp(x / 15.0)) + np.random.normal(scale=1.0, size=x.shape)

# Execute with data from 2D function
z = f1d(x, np.array([2, 1, 1.9, 2.35, 0.1])) # x axis
plt.plot(z)

y1 = gaussian_filter(z, sigma=10)
y2 = gaussian_filter(z, sigma=30)
plt.plot(y1)
plt.show()
