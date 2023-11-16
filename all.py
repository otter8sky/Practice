from operations import *
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x_start = 0
y_start = 1
x_end = 2
h = 0.2
l = 11

x = np.linspace(x_start, x_end, l)

print('x = ', x)
y_implicit_Euler = np.array([])
y_implicit_Euler = np.append(y_implicit_Euler, y_start)  # добавление начальной точки

y_explicit_Euler = np.array([])
y_explicit_Euler = np.append(y_explicit_Euler, y_start)

y_Runge_Kutta = np.array([])
y_Runge_Kutta = np.append(y_Runge_Kutta, y_start)

y_corrector = np.array([])
y_corrector = np.append(y_corrector, y_start)


class Body:
    def __init__(self, velicity_x, velicity_y, coord_x, coord_y, mass):
        self.velicity_x = velicity_x
        self.velicity_y = velicity_y
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.mass = mass

def f(x, y):
    ''' y = np.exp(-15 * x) '''
    return -15 * y


def explicit_Euler(y_explicit_Euler, h):
    y = y_start
    for i in range(len(x) - 1):
        y = y + f(x, y) * h
        y_explicit_Euler = np.append(y_explicit_Euler, y)
    print('y_explicit_Euler = ', y_explicit_Euler)
    print("---------------------------------------------------------------")
    return y_explicit_Euler


def implicit_Euler(y_implicit_Euler, h):
    y = y_start
    for i in range(len(x) - 1):
        y = y / (1 + 15 * h)
        y_implicit_Euler = np.append(y_implicit_Euler, y)
    print('y_implicit_Euler = ', y_implicit_Euler)
    print("---------------------------------------------------------------")
    return y_implicit_Euler


def Runge_Kutta(x, y_Runge_Kutta, h):
    y = y_start
    for i in range(len(x)-1):
        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_Runge_Kutta = np.append(y_Runge_Kutta, y)
    print('y_Runge_Kutta = ', y_Runge_Kutta)
    print("---------------------------------------------------------------")
    return y_Runge_Kutta


def corrector(y_corrector, h):
    y = y_start
    for i in range(len(x) - 1):
        y_ii = y + f(x, y) * h
        y = y + h * (f(x, y) + f(x + h, y_ii)) / 2
        y_corrector = np.append(y_corrector, y)
    print('y_corrector = ', y_corrector)
    print("---------------------------------------------------------------")
    return y_corrector


plt.plot(x, explicit_Euler(y_explicit_Euler, h), color='red', label='Explicit_Euler')
plt.plot(x, implicit_Euler(y_implicit_Euler, h), color='blue', label='Implicit_Euler')
plt.plot(x, Runge_Kutta(x, y_Runge_Kutta, h), color='green', label='Runge_Kutta')
plt.plot(x, corrector(y_corrector, h), color='orange', label='Corrector')

plt.legend()
ax.set_xlabel("x", fontsize=10)
ax.set_ylabel("y", fontsize=10)
ax.grid(True, linestyle='-')
ax.tick_params(labelsize='medium', width=3)
ax.set_xlim(0, 2)
ax.set_ylim(-3, 3)
plt.show()