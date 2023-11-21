from mpl_toolkits.mplot3d import Axes3D
import ast
from main import *
import matplotlib.pyplot as plt

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
plt.title(f"{Name} for {time_end} year(s)", fontsize=20, color="purple")
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
plt.title(f"{Name} Energy", fontsize=20, color="purple")
plt.plot(time_en, energy, label="energy")
plt.show()
