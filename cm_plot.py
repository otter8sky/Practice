from mpl_toolkits.mplot3d import Axes3D
import ast
from main import *
import matplotlib.pyplot as plt

Names = ['Explicit Euler', 'Predictor-Corrector', 'RK-4', 'Verlet', "Leap-Frog"]

method = "Leap-Frog"

fig, ax = plt.subplots()

f = open("center_mass.txt", "r+")
coord_cm_x = ast.literal_eval(f.readline())
coord_cm_y = ast.literal_eval(f.readline())
coord_cm_z = ast.literal_eval(f.readline())
time = ast.literal_eval(f.readline())

f.close()
plt.xlabel('a.e.')
plt.ylabel('a.e.')
plt.title(f"{method} Trajectory of center of mass", fontsize=18, color="green")
plt.plot(coord_cm_x, coord_cm_y)
f.close()
plt.show()
