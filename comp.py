from operations import *
from methods import By_Leap_Frog

def read_data_file(file_name, n):
    file = open(f"{file_name}", 'r')
    names = []
    Bodies = []
    colors = []
    mas = []
    s = 0
    cnt = 0
    cnt_body = 1
    kms_au_kg = False
    auyear_au_masssun = False
    for item in file:
        item = item.strip()
        if cnt == 0:
            if item == "au_year/au/mass_sun":
                auyear_au_masssun = True
            elif item == "kms/au/kg":
                kms_au_kg = True
        if cnt > 1:
            if cnt_body > n:
                break
            if s < 5:
                if s == 1:
                    if kms_au_kg:
                        item = change_vel(list(map(float, item.split())))[:]
                    elif auyear_au_masssun:
                        item = list(map(float, item.split()))
                elif s == 2:
                    item = list(map(float, item.split()))
                elif s == 3:
                    if kms_au_kg:
                        item = change_mass(float(item))
                    elif auyear_au_masssun:
                        item = float(item)
                mas.append(item)
                s += 1
            else:
                names.append(mas[0])
                colors.append(mas[4])
                Bodies.append(Body(mas[1], mas[2], mas[3], Acs))
                mas = []
                s = 0
                cnt_body += 1
        else:
            cnt += 1
    file.close()
    Bodies = get_acs_for_all(Bodies)
    return Bodies, names, colors

def comp(method, bodies, t, time_end, time_step, colors, names, delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min):
    bodies_x = [[] for i in range(len(bodies))]
    bodies_y = [[] for i in range(len(bodies))]
    bodies_z = [[] for i in range(len(bodies))]
    coords_cm_x = []
    coords_cm_y = []
    coords_cm_z = []

    energy = []
    time = []
    time_en = []
    cnt = 0
    while t <= time_end:
        if t == 0 and method == By_Leap_Frog:
            for i in range(len(bodies)):
                bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))
        else:
            time_step = get_time_step(bodies, time_step, delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min)
            result = method(bodies, time_step)
            energy.append(get_Energy(result))
            if cnt == 100 and t != 0:
                for i in range(len(bodies)):
                    bodies_x[i].append(result[i].coord[0])
                    bodies_y[i].append(result[i].coord[1])
                    bodies_z[i].append(result[i].coord[2])
                    coords_cm_x.append(get_coord_cm(bodies)[0])
                    coords_cm_y.append(get_coord_cm(bodies)[1])
                    coords_cm_z.append(get_coord_cm(bodies)[2])
                time.append(t)
                cnt = 0
            elif t == 0:
                for i in range(len(bodies)):
                    bodies_x[i].append(bodies[i].coord[0])
                    bodies_y[i].append(bodies[i].coord[1])
                    bodies_z[i].append(bodies[i].coord[2])
                    coords_cm_x.append(get_coord_cm(bodies)[0])
                    coords_cm_y.append(get_coord_cm(bodies)[1])
                    coords_cm_z.append(get_coord_cm(bodies)[2])
                time.append(t)
        time_en.append(t)
        cnt += 1
        t += time_step
    if method == By_Leap_Frog:
        for i in range(len(bodies)):
            bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, -time_step / 2))
        energy.append(get_Energy(bodies))

    for i in range(len(bodies)):
        f = open(f"{i + 1}.txt", "w+")
        f.writelines(
            [str(bodies_x[i]), "\n", str(bodies_y[i]), "\n", str(bodies_z[i]), "\n", colors[i], "\n", names[i], "\n"])
        f.close()
    f = open("energy.txt", "w+")
    f.writelines([str(energy), "\n", str(time_en), "\n", str(time), "\n"])
    f.close()
    f = open("center_mass.txt", "w+")
    f.writelines([str(coords_cm_x), "\n", str(coords_cm_y), "\n", str(coords_cm_z), "\n", str(time), "\n"])
    f.close()

