from operations import *
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x_start = 2
y_start = 3
x_end = 4
h = 0.2
l = 10

x = np.linspace(x_start, x_end, l)

y_implicit_Euler = np.array([])
y_implicit_Euler = np.append(y_implicit_Euler, y_start)  # добавление начальной точки

y_explicit_Euler = np.array([])
y_explicit_Euler = np.append(y_explicit_Euler, y_start)

y_Runge_Kutta = np.array([])
y_Runge_Kutta = np.append(y_Runge_Kutta, y_start)

y_corrector = np.array([])
y_corrector = np.append(y_corrector, y_start)

def f(x, y):
    return y

def explicit_Euler(f, y_explicit_Euler, h):
    y = y_start
    for i in range(len(x) - 1):
        y = y + f * h
        y_explicit_Euler = np.append(y_explicit_Euler, y)
    return y_explicit_Euler


def implicit_Euler(y_implicit_Euler, h):
    y = y_start
    for i in range(len(x) - 1):
        y = y / (1 - h)
        y_implicit_Euler = np.append(y_implicit_Euler, y)
    return y_implicit_Euler


def Runge_Kutta(f, x, y_Runge_Kutta, h):
    y = y_start
    for i in range(len(x) - 1):
        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_Runge_Kutta = np.append(y_Runge_Kutta, y)
    return y_Runge_Kutta


def corrector(f1, f2, y_corrector, h):
    y = y_start
    for i in range(len(x) - 1):
        y_ii = y + f(x, y) * h
        y = y + h * (f(x, y) + f(x + h, y_ii)) / 2
        y_corrector = np.append(y_corrector, y)
    return y_corrector


plt.plot(x, implicit_Euler(y_implicit_Euler, h), color='red', label='implicit_Euler')
plt.plot(x, Runge_Kutta(f, x, y_Runge_Kutta, h), color='black', label='runge_kutt')
plt.plot(x, explicit_Euler(f(x, y_explicit_Euler), y_explicit_Euler, h), color='blue', label='explicit_Euler')
plt.plot(x, corrector(f(x, y_corrector), f(x, y_corrector), y_corrector, h), color='green', label='corrector')

plt.legend()
ax.grid(True, linestyle='-')
ax.tick_params(labelsize='medium', width=3)
plt.show()