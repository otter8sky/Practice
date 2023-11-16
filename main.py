from methods import *
from r_data_file import *
import numpy as np

def comp(method, bodies, t, time_end, time_step):
    bodies_x = [[] for i in range(n)]
    bodies_y = [[] for i in range(n)]
    bodies_z = [[] for i in range(n)]
    energy = []
    time = []
    time_en = []
    cnt = 0
    while t <= time_end:
        result = method(bodies, time_step)
        energy.append(get_Energy(result))
        if cnt == 100:
            for i in range(n):
                bodies_x[i].append(result[i].coord[0])
                bodies_y[i].append(result[i].coord[1])
                bodies_z[i].append(result[i].coord[2])
            time.append(t)
            cnt = 0
        time_en.append(t)
        cnt += 1
        t += time_step

    for i in range(n):
        f = open(f"{i + 1}.txt", "w+")
        f.writelines(
            [str(bodies_x[i]), "\n", str(bodies_y[i]), "\n", str(bodies_z[i]), "\n", colors[i], "\n", names[i], "\n"])
        f.close()

    f = open("energy.txt", "w+")
    f.writelines([str(energy), "\n", str(time_en), "\n", str(time), "\n"])
    f.close()
