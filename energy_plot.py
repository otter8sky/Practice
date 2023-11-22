from mpl_toolkits.mplot3d import Axes3D
import ast
import matplotlib.pyplot as plt

Names = ['Explicit Euler', 'Predictor-Corrector', 'RK-4', 'Verlet', "Leap-Frog"]

method = "Leap-Frog"

fig, ax = plt.subplots()

f = open("energy.txt", "r+")
energy = ast.literal_eval(f.readline())
time_en = ast.literal_eval(f.readline())
time = ast.literal_eval(f.readline())
f.close()
plt.xlabel('t, years')
plt.ylabel('energy')
plt.title(f"{method} Energy", fontsize=20, color="purple")
plt.plot(time_en, energy)
plt.show()
