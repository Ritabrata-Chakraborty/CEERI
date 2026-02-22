#!/usr/bin/env python3
"""Visualize offset and angle error convergence for control tuning."""

import numpy as np
import matplotlib.pyplot as plt

K_X = 0.1565465461  # Correction gain for offset
K_THETA = 0.3  # Correction gain for angle
E_X0, E_THETA0 = 10.0, 15.0  # Initial errors
ITERATIONS = 50


def main():
    e_x = [E_X0 * np.exp(-K_X * t) for t in range(ITERATIONS)]
    e_theta = [E_THETA0 * np.exp(-K_THETA * t) for t in range(ITERATIONS)]

    plt.plot(e_x, label="Offset Error", color="b")
    plt.plot(e_theta, label="Angle Error", color="r")
    plt.xlabel("Iteration")
    plt.ylabel("Error Magnitude")
    plt.title("Offset and Angle Error Convergence")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
