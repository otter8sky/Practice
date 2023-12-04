from mpl_toolkits.mplot3d import Axes3D
import ast
import matplotlib.pyplot as plt
from comp import read_data_file
from operations import *

Bodies, names, colors = read_data_file('DATA_strange.txt')

bodies_x = [[] for i in range(len(Bodies))]
bodies_y = [[] for i in range(len(Bodies))]
bodies_z = [[] for i in range(len(Bodies))]
bodies_vel = [[] for i in range(len(Bodies))]
bodies_mass = []
for i in range(len(Bodies)):
    bodies_mass.append(Bodies[i].mass)

method = "idonotknowyet"

for i in range(len(Bodies)):
    f = open(f"{names[i]}.txt", "r+")
    method = f.readline().strip()
    bodies_x[i].append(ast.literal_eval(f.readline()))
    bodies_y[i].append(ast.literal_eval(f.readline()))
    bodies_z[i].append(ast.literal_eval(f.readline()))
    bodies_vel[i] = [ast.literal_eval(f.readline())]
    f.close()

f = open("energy.txt", "r+")
method = f.readline().strip()
energy = ast.literal_eval(f.readline())
time_en = ast.literal_eval(f.readline())
f.close()

vect_total_momentum_x = []
vect_total_momentum_y = []
vect_total_momentum_z = []
mag_vect_total_momentum = []

for i in range(len(bodies_x[0])):
    for j in range(len(Bodies)):
        vect_total_momentum_x.append(get_vect_total_momentum_out(bodies_vel[i], bodies_mass)[0])
        vect_total_momentum_y.append(get_vect_total_momentum_out(bodies_vel[i], bodies_mass)[1])
        vect_total_momentum_z.append(get_vect_total_momentum_out(bodies_vel[i], bodies_mass)[0])
        mag_vect_total_momentum.append(get_mag(get_vect_total_momentum_out(bodies_vel[i], bodies_mass)))

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].plot(vect_total_momentum_x, vect_total_momentum_y, color="green")
axs[0].ylabel("Y, a.e.")
axs[0].xlabel("X, a.e.")
axs[0].set_title("Trajectory of vector of total momentum", fontsize=15, color="green")
axs[1].plot(time_en, mag_vect_total_momentum, color="blue")
axs[1].ylabel("L", r'$\frac{M_sun * a.e.}{years}$')
axs[1].xlabel("t, years")
axs[1].set_title("Magnitude of vector of total momentum", fontsize=15, color="blue")
axs.set_title(f"{method} for {time_en[-1]} years", fontsize=15)
plt.show()
