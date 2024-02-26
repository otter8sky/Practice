from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from comp import read_data_file

bodies = read_data_file('DATA.txt')

fig, ax = plt.subplots()

for i in range(len(bodies)):
    df_i = pd.read_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
    x_i = df_i['X, a.u.'].tolist()
    y_i = df_i['Y, a.u.'].tolist()
    z_i = df_i['Z, a.u.'].tolist()

    velocities_x = df_i['Vx, a.u./year'].tolist()
    velocities_y = df_i['Vy, a.u./year'].tolist()
    velocities_z = df_i['Vz, a.u./year'].tolist()

    plt.plot(x_i, y_i, color=color, label=name)
    f.close()

plt.title(f"{Method} for 100 year(s)", fontsize=20, color="purple")
plt.xlabel('X, а.е.')
plt.ylabel('Y, а.е.')
plt.legend(loc='best')
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.gca().set_aspect("equal")
plt.show()
