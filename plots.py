import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
from main import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


Methods = [By_Ex_Euler,
           By_PC,
           By_RK_4]
Names = ['Explicit Euler', 'Predictor-Corrector', 'RK-4', 'Leap-frog']

# ввести метод и количество тел СС
method = By_PC
n = 10
time_end = 5
data_time_step = 0.001
h = 0.0001

energy = comp(method, Bodies, t, time_end, h)
fig, ax = plt.subplots()

for i in range(n):
    f = open(f"{i+1}.txt", "r+")
    x_i = ast.literal_eval(f.readline())
    y_i = ast.literal_eval(f.readline())
    z_i = ast.literal_eval(f.readline())
    color = f.readline().strip()
    name = f.readline().strip()
    plt.plot(x_i, y_i, color=color, label=name)
    f.close()

Name = Names[Methods.index(method)]
plt.title(f"{Name}", fontsize=20, color="green")
plt.xlabel('X, а.е.')
plt.ylabel('Y, а.е.')
plt.legend(loc='best')
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.gca().set_aspect("equal")
plt.show()

f = open("energy.txt", "r+")
energy = ast.literal_eval(f.readline())
time_en = ast.literal_eval(f.readline())
time = ast.literal_eval(f.readline())
f.close()

plt.legend(loc='best')
plt.xlabel('t, years')
plt.ylabel('energy')
plt.title("Energy", fontsize=20, color="green")
plt.plot(time_en, energy, label="energy")
plt.show()
