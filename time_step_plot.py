from mpl_toolkits.mplot3d import Axes3D
import ast
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

f = open("time_step.txt", "r+")
method = f.readline().strip()
time_en = ast.literal_eval(f.readline())
time_step = ast.literal_eval(f.readline())
f.close()
plt.xlabel('time, years')
plt.ylabel('time step, years')
plt.title(f"{method} Time Step", fontsize=20, color="purple")
plt.plot(time_en, time_step)
plt.show()
