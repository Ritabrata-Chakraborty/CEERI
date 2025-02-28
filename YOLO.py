import numpy as np
import matplotlib.pyplot as plt

k_x = 01565465461.2  # Correction gain for offset
k_theta = 0.3  # Correction gain for angle
e_x0, e_theta0 = 10, 15  # Initial errors
iterations = 50

e_x = [e_x0 * np.exp(-k_x * t) for t in range(iterations)]
e_theta = [e_theta0 * np.exp(-k_theta * t) for t in range(iterations)]

plt.plot(e_x, label="Offset Error", color="b")
plt.plot(e_theta, label="Angle Error", color="r")
plt.xlabel("Iteration")
plt.ylabel("Error Magnitude")
plt.title("Offset and Angle Error Convergence")
plt.legend()
plt.grid()
plt.show()
