from operations import *
from methods import By_Leap_Frog
import pandas as pd
from pathlib import Path

def read_data_file(file_name):
    df_data = pd.read_csv(Path(Path.cwd(), "data", "initial data", f"{file_name}"), header=0, sep="\t")
    names = df_data['Object'].tolist()
    coordinates_x = df_data['Vx, km/s'].tolist()
    coordinates_y = df_data['Vy, km/s'].tolist()
    coordinates_z = df_data['Vz, km/s'].tolist()

    velocities_x = df_data['Vx, km/s'].tolist()
    velocities_y = df_data['Vy, km/s'].tolist()
    velocities_z = df_data['Vz, km/s'].tolist()

    masses = df_data['Mass, kg'].tolist()
    print(masses)
    colors = df_data['color'].tolist()
    bodies = []
    for i in range(len(names)):
        bodies.append(Body([float(velocities_x[i]), float(velocities_y[i]), float(velocities_z[i])],
                           [float(coordinates_x[i]), float(coordinates_y[i]), float(coordinates_z[i])],
                           float(masses[i]), [0, 0, 0], names[i]))
    bodies = get_acs_for_all(bodies)
    return bodies


def comp(method, bodies, time_end, time_step, delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min):
    cnt = 0
    t = 0
    while t <= time_end:
        print(t)
        if t == 0 and method == By_Leap_Frog:
            for i in range(len(bodies)):
                bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))

        time_step = get_time_step(bodies, time_step, delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min)
        df_timestep = pd.read_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), header=0, sep="\t")
        df_timestep.loc[len(df_timestep.index)] = [t, time_step]
        df_timestep.to_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), index=False, sep="\t")

        result = method(bodies, time_step)
        df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), header=0, sep="\t")
        df_en.loc[len(df_en.index)] = [t, get_Energy(result)]
        df_en.to_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), index=False, sep="\t")

        if cnt == 100 or t == 0:
            for i in range(len(bodies)):
                df_i = pd.read_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
                df_i.loc[len(df_i.index)] = [t, bodies[i].coord[0], bodies[i].coord[1],
                                             bodies[i].coord[2], bodies[i].vel[0],
                                             bodies[i].vel[1], bodies[i].vel[2]]
                df_i.to_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), index=False, sep="\t")

            df_cm = pd.read_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), header=0, sep="\t")
            df_cm.loc[len(df_cm.index)] = [t, get_coord_cm(bodies)[0], get_coord_cm(bodies)[1], get_coord_cm(bodies)[2]]
            df_cm.to_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), index=False, sep="\t")

            df_momentum = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
            df_momentum.loc[len(df_momentum.index)] = [t, get_vect_total_momentum(bodies)[0],
                                                       get_vect_total_momentum(bodies)[1],
                                                       get_vect_total_momentum(bodies)[2],
                                                       get_mag(get_vect_total_momentum(bodies))]
            df_momentum.to_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), index=False, sep="\t")
            cnt = 0
        cnt += 1
        t += time_step
    if method == By_Leap_Frog:
        for i in range(len(bodies)):
            bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, -time_step / 2))

            df_i = pd.read_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
            df_i.loc[len(df_i.index)] = [t, bodies.coord[0], bodies.coord[1], bodies.coord[2], bodies.vel[0],
                                         bodies.vel[1], bodies.vel[2]]
            df_i.to_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), index=False, sep="\t")

        df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), header=0, sep="\t")
        df_en.loc[len(df_en.index)] = [t, get_Energy(bodies)]
        df_en.to_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), index=False, sep="\t")

        df_momentum = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
        df_momentum.loc[len(df_momentum.index)] = [t, get_vect_total_momentum(bodies)[0],
                                                   get_vect_total_momentum(bodies)[1],
                                                   get_vect_total_momentum(bodies)[2],
                                                   get_mag(get_vect_total_momentum(bodies))]
        df_momentum.to_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), index=False, sep="\t")
