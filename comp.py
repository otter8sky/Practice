from operations import *
from methods import methods
from methods import methods_names
from methods import By_Leap_Frog
import pandas as pd
from pathlib import Path
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def read_comp_file(file_name):
    df_data = pd.read_csv(Path(Path.cwd(), "data", "initial data", f"{file_name}"), header=0, sep="\t")
    method = df_data['method'].tolist()[0]
    time_end = float(df_data['time_end, years'].tolist()[0])
    initial_timestep = float(df_data['initial_timestep, years'].tolist()[0])
    delta_vel = float(df_data['delta_vel, a.u./years'].tolist()[0])
    delta_coord = float(df_data['delta_coord, a.u.'].tolist()[0])
    delta_timestep = float(df_data['delta_timestep, years'].tolist()[0])
    timestep_max = float(df_data['timestep_max, years'].tolist()[0])
    timestep_min = float(df_data['timestep_min, years'].tolist()[0])
    dt_output = float(df_data['dt_output, years'].tolist()[0])
    problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep, delta_vel,
                      delta_coord, delta_timestep, timestep_max, timestep_min, dt_output)
    return problem
def read_data_file():
    bodies = []
    for f in os.listdir(Path(Path.cwd(), "data", "initial data", "coordinates")):
        f = open(Path(Path.cwd(), "data", "initial data", "coordinates", f"{f}"), "r")
        file = f.read()

        match = re.search(r'Object *: *\w+', file)
        if match:
            name_i = re.split(r' *: *', match[0])[1]
            print(name_i)
        else:
            print("Pattern not found")

        file = re.split(r"\n", file)

        data = re.split(r" +", file[-2])

        x_i = float(data[6])
        y_i = float(data[7])
        z_i = float(data[8])
        vx_i = float(data[9]) * 365.2422
        vy_i = float(data[10]) * 365.2422
        vz_i = float(data[11]) * 365.2422

        df_data = pd.read_csv(Path(Path.cwd(), "data", "initial data", "fixed data", "DATA.txt"), header=0, sep="\t")
        data_i = df_data[f"{name_i}"]
        mass_i = data_i[0]
        color_i = data_i[1]

        bodies.append(Body([vx_i, vy_i, vz_i],
                           [vx_i, vy_i, vz_i],
                           [x_i, y_i, z_i],
                           float(mass_i), [0, 0, 0], name_i, color_i))
        print("OK")
    print(bodies[-1])
    bodies = get_acs_for_all(bodies)
    return bodies
def comp(problem, bodies):
    cnt = 0
    t = 0
    time_step = problem.initial_timestep
    timestep = [time_step]
    bodies_coord = [[[bodies[i].coord[0]],
                     [bodies[i].coord[1]],
                     [bodies[i].coord[2]]] for i in range(len(bodies))]
    bodies_vel = [[[bodies[i].vel[0]],
                   [bodies[i].vel[1]],
                   [bodies[i].vel[2]]] for i in range(len(bodies))]
    bodies_axis = [[major_axis(bodies, i)] for i in range(len(bodies))]
    bodies_ecc = [[eccentricity(bodies, i)] for i in range(len(bodies))]
    bodies_inc = [[inclination(bodies, i)] for i in range(len(bodies))]
    bodies_long = [[longitude_of_asc_node(bodies, i)] for i in range(len(bodies))]
    bodies_per = [[arg_of_periapsis(bodies, i)] for i in range(len(bodies))]

    energy = [get_Energy(bodies)]
    time_full = [t]
    time = [t]
    cm_coord = [[get_coord_cm(bodies)[0]],
                [get_coord_cm(bodies)[1]],
                [get_coord_cm(bodies)[2]]]
    mom_coord = [[get_total_momentum(bodies)[0]],
                 [get_total_momentum(bodies)[1]],
                 [get_total_momentum(bodies)[2]]]
    mag_mom = [get_mag(get_total_momentum(bodies))]
    ang_mom_coord = [[get_total_angular_momentum(bodies)[0]],
                     [get_total_angular_momentum(bodies)[1]],
                     [get_total_angular_momentum(bodies)[2]]]
    mag_ang_mom = [get_mag(get_total_angular_momentum(bodies))]

    next_output = problem.dt_output

    while t <= problem.time_end:
        print(round(t * 100/problem.time_end, 2), "%")
        time_full.append(t)
        if t == 0 and problem.method == By_Leap_Frog:
            for i in range(len(bodies)):
                # FIXME: change velocities into correct ones in Leap-Frog
                bodies[i].half_vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))
        time_step = get_time_step(bodies, time_step, problem)
        timestep.append(time_step)
        result = problem.method(bodies, time_step)

        if abs(t - next_output) <= problem.dt_output:
            cnt += 1
            for i in range(len(bodies)):
                bodies_coord[i] = fill_coord_list_3d(bodies_coord[i], bodies[i])[:]
                bodies_vel[i] = fill_vel_list_3d(bodies_vel[i], bodies[i])[:]

                bodies_axis[i].append(major_axis(bodies, i))
                bodies_ecc[i].append(eccentricity(bodies, i))
                bodies_inc[i].append(inclination(bodies, i))
                bodies_long[i].append(longitude_of_asc_node(bodies, i))
                bodies_per[i].append(arg_of_periapsis(bodies, i))

            cm_coord = fill_list_3d_func(cm_coord, get_coord_cm, bodies)[:]
            mom_coord = fill_list_3d_func(mom_coord, get_total_momentum, bodies)[:]
            mag_mom.append(get_mag(get_total_momentum(bodies)))
            ang_mom_coord = fill_list_3d_func(ang_mom_coord, get_total_angular_momentum, bodies)[:]
            mag_ang_mom.append(get_mag(get_total_angular_momentum(bodies)))
            time.append(t)
            energy.append(get_Energy(result))

            next_output += problem.dt_output
        t += time_step

    for i in range(len(bodies)):
        bodies_coord[i] = fill_coord_list_3d(bodies_coord[i], bodies[i])[:]
        bodies_vel[i] = fill_vel_list_3d(bodies_vel[i], bodies[i])[:]

        bodies_axis[i].append(major_axis(bodies, i))
        bodies_ecc[i].append(eccentricity(bodies, i))
        bodies_inc[i].append(inclination(bodies, i))
        bodies_long[i].append(longitude_of_asc_node(bodies, i))
        bodies_per[i].append(arg_of_periapsis(bodies, i))

    cm_coord = fill_list_3d_func(cm_coord, get_coord_cm, bodies)[:]
    mom_coord = fill_list_3d_func(mom_coord, get_total_momentum, bodies)[:]
    mag_mom.append(get_mag(get_total_momentum(bodies)))
    ang_mom_coord = fill_list_3d_func(ang_mom_coord, get_total_angular_momentum, bodies)[:]
    mag_ang_mom.append(get_mag(get_total_angular_momentum(bodies)))
    time.append(t)
    energy.append(get_Energy(bodies))

    time_full = pd.Series(time_full).to_frame(name="time, years")
    time = pd.Series(time).to_frame(name="time, years")
    timestep = pd.Series(timestep).to_frame(name="time step, years")
    df_ts = pd.concat([time, timestep, time_full], axis=1)
    df_ts.to_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), sep="\t")

    energy = pd.Series(energy).to_frame(name="energy")
    df_en = pd.concat([time, energy], axis=1)
    df_en.to_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), sep="\t")

    cm_x = pd.Series(cm_coord[0]).to_frame(name="cm_x, a.u.")
    cm_y = pd.Series(cm_coord[1]).to_frame(name="cm_y, a.u.")
    cm_z = pd.Series(cm_coord[2]).to_frame(name="cm_z, a.u.")
    df_cm = pd.concat([time, cm_x, cm_y, cm_z], axis=1)
    df_cm.to_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), sep="\t")

    mom_x = pd.Series(mom_coord[0]).to_frame(name="momentum_x")
    mom_y = pd.Series(mom_coord[1]).to_frame(name="momentum_y")
    mom_z = pd.Series(mom_coord[2]).to_frame(name="momentum_z")
    mag_mom = pd.Series(mag_mom).to_frame(name="momentum_mag")
    df_mom = pd.concat([time, mom_x, mom_y, mom_z, mag_mom], axis=1)
    df_mom.to_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), sep="\t")

    ang_mom_x = pd.Series(ang_mom_coord[0]).to_frame(name="angular_momentum_x")
    ang_mom_y = pd.Series(ang_mom_coord[1]).to_frame(name="angular_momentum_y")
    ang_mom_z = pd.Series(ang_mom_coord[2]).to_frame(name="angular_momentum_z")
    ang_mag_mom = pd.Series(mag_ang_mom).to_frame(name="angular_momentum_mag")
    df_ang_mom = pd.concat([time, ang_mom_x, ang_mom_y, ang_mom_z, ang_mag_mom], axis=1)
    df_ang_mom.to_csv(Path(Path.cwd(), "data", "data out", "angular momentum.txt"), sep="\t")

    for i in range(len(bodies)):
        x_i = pd.Series(bodies_coord[i][0]).to_frame(name="X, a.u.")
        y_i = pd.Series(bodies_coord[i][1]).to_frame(name="Y, a.u.")
        z_i = pd.Series(bodies_coord[i][2]).to_frame(name="Z, a.u.")
        vel_x_i = pd.Series(bodies_vel[i][0]).to_frame(name="V_x, years")
        vel_y_i = pd.Series(bodies_vel[i][1]).to_frame(name="V_y, a.u./year")
        vel_z_i = pd.Series(bodies_vel[i][2]).to_frame(name="V_z, a.u./year")

        df_i = pd.concat([time, x_i, y_i, z_i, vel_x_i, vel_y_i, vel_z_i], axis=1)
        df_i.to_csv(Path(Path.cwd(), "data", "data out", "objects", f"{bodies[i].name}.txt"), sep="\t")

        a_i = pd.Series(bodies_axis[i]).to_frame(name="a, a.u.")
        e_i = pd.Series(bodies_ecc[i]).to_frame(name="e")
        i_i = pd.Series(bodies_inc[i]).to_frame(name="i, degrees")
        long_of_asc_node_i = pd.Series(bodies_long[i]).to_frame(name="long_of_asc_node, degrees")
        arg_of_periapsis_i = pd.Series(bodies_per[i]).to_frame(name="arg_of_periapsis, degrees")

        df_i = pd.concat([time, a_i, e_i, i_i, long_of_asc_node_i, arg_of_periapsis_i], axis=1)
        df_i.to_csv(Path(Path.cwd(), "data", "data out", "elements", f"elements of {bodies[i].name}.txt"), sep="\t")

def plot_bodies(bodies, problem):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(bodies)):
        df_i = pd.read_csv(Path(Path.cwd(), "data", "data out", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
        x_i = df_i['X, a.u.'].tolist()
        y_i = df_i['Y, a.u.'].tolist()
        z_i = df_i['Z, a.u.'].tolist()
        ax.plot(x_i, y_i, z_i, color=bodies[i].color, label=bodies[i].name)
        # plt.plot(x_i, y_i, color=bodies[i].color, label=bodies[i].name)
    plt.title(f"{problem.method.__name__} for {problem.time_end} year(s)", fontsize=20, color="purple")
    plt.xlabel('X, а.е.')
    plt.ylabel('Y, а.е.')
    plt.legend(loc='best')
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
def plot_elements(bodies, problem):
    figure, axis = plt.subplots(1, 3)

    elements = [[] for i in range(5)]
    list_for_elements = [["major axis", "au"],
                         ["eccentricity", ""],
                         ["inclination", "degrees"],
                         ["longitude of ascending node", "degrees"],
                         ["argument of periapsis", "degrees"]]
    for j in range(len(bodies)):
        df_j = pd.read_csv(Path(Path.cwd(), "data", "data out", "elements", f"elements of {bodies[j].name}.txt"),
                           header=0, sep="\t")
        time = df_j['time, years'].tolist()
        elements[0].append(df_j["a, a.u."].tolist())
        elements[1].append(df_j["e"].tolist())
        elements[2].append(df_j["i, degrees"].tolist())
        elements[3].append(df_j["long_of_asc_node, degrees"].tolist())
        elements[4].append(df_j["arg_of_periapsis, degrees"].tolist())

    for i in range(len(bodies)):
        axis[0].plot(time, elements[0][i], color=bodies[i].color, label=bodies[i].name)
        axis[1].plot(time, elements[1][i], color=bodies[i].color, label=bodies[i].name)
        axis[2].plot(time, elements[2][i], color=bodies[i].color, label=bodies[i].name)

    axis[0].set_title("major axis")
    axis[1].set_title("eccentricity")
    axis[2].set_title("inclination")
    plt.legend(loc='best')

    figure, axis = plt.subplots(1, 2)

    for i in range(len(bodies)):
        axis[0].plot(time, elements[3][i], color=bodies[i].color, label=bodies[i].name)
        axis[1].plot(time, elements[4][i], color=bodies[i].color, label=bodies[i].name)

    axis[0].set_title("longitude of ascending node")
    axis[1].set_title("argument of periapsis")
    plt.legend(loc='best')

    plt.show()
def plot_energy(problem):
    df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), header=0, sep="\t")
    time = df_en['time, years'].tolist()
    energy = df_en['energy'].tolist()
    plt.plot(time, energy, color='red')

    plt.title(f"{problem.method.__name__} energy", fontsize=20, color="red")
    plt.xlabel('time, years')
    plt.ylabel('energy')
    # plt.gca().set_aspect("equal")
    plt.show()
def plot_momentum(problem):
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    momentum_x = df_mom['momentum_x'].tolist()
    momentum_y = df_mom['momentum_y'].tolist()
    momentum_z = df_mom['momentum_z'].tolist()
    momentum_mag = df_mom['momentum_mag'].tolist()

    plt.plot(momentum_x, momentum_y, color='blue')
    plt.title(f"{problem.method.__name__} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')
    plt.show()

    plt.plot(time, momentum_mag, color='purple')
    plt.title(f"{problem.method.__name__} magnitude of vector of total momentum", fontsize=20, color="purple")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()
def plot_angular_momentum(problem):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "angular momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    ang_mom_x = df_mom['angular_momentum_x'].tolist()
    ang_mom_y = df_mom['angular_momentum_y'].tolist()
    ang_mom_z = df_mom['angular_momentum_z'].tolist()
    ang_mom_mag = df_mom['angular_momentum_mag'].tolist()

    ax.plot(ang_mom_x, ang_mom_y, ang_mom_z, color='blue')
    plt.title(f"{problem.method.__name__} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.plot(time, ang_mom_mag, color='cyan')
    plt.title(f"{problem.method.__name__} magnitude of vector of angular momentum", fontsize=20, color="black")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()
def plot_cm(problem):
    fig, ax = plt.subplots()
    df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), header=0, sep="\t")
    time = df_en['time, years'].tolist()
    coord_cm_x = df_en['cm_x, a.u.'].tolist()
    coord_cm_y = df_en['cm_y, a.u.'].tolist()
    coord_cm_z = df_en['cm_z, a.u.'].tolist()
    plt.plot(coord_cm_x, coord_cm_y, color='red')

    plt.title(f"{problem.method.__name__} trajectory of center of mass ({problem.time_end} years)",
              fontsize=20, color="green")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')
    plt.show()
def plot_time_step(problem):
    df_ts = pd.read_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), header=0, sep="\t")
    time = df_ts['time, years'].tolist()
    time_step = df_ts['time step, years'].tolist()
    plt.plot(time, time_step, color='red')
    plt.title(f"Magnitude of time step ({problem.method.__name__})", fontsize=20, color="green")
    plt.xlabel('time, years')
    plt.ylabel('time step, years')
    plt.show()
def plot_all(problem, bodies):
    plt.figure()
    for i in range(len(bodies)):
        df_i = pd.read_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
        x_i = df_i['X, a.u.'].tolist()
        y_i = df_i['Y, a.u.'].tolist()
        z_i = df_i['Z, a.u.'].tolist()
        plt.plot(x_i, y_i, color=bodies[i].color, label=bodies[i].name)

    plt.title(f"{problem.method.__name__} for {problem.time_end} year(s)", fontsize=20, color="purple")
    plt.xlabel('X, а.е.')
    plt.ylabel('Y, а.е.')
    plt.legend(loc='best')
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    plt.figure()
    df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), header=0, sep="\t")
    time = df_en['time, years'].tolist()
    energy = df_en['energy'].tolist()
    plt.plot(time, energy, color='red')

    plt.title(f"{problem.method.__name__} energy", fontsize=20, color="red")
    plt.xlabel('time, years')
    plt.ylabel('energy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    momentum_x = df_mom['momentum_y'].tolist()
    momentum_y = df_mom['momentum_y'].tolist()
    momentum_z = df_mom['momentum_z'].tolist()
    momentum_mag = df_mom['momentum_mag'].tolist()
    plt.plot(momentum_x, momentum_y, momentum_z, color='blue')
    plt.title(f"{problem.method.__name__} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')

    plt.figure()
    plt.plot(time, momentum_mag, color='purple')
    plt.title(f"{problem.method.__name__} magnitude of vector of total momentum", fontsize=20, color="purple")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_cm = pd.read_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), header=0, sep="\t")
    time = df_cm['time, years'].tolist()
    coord_cm_x = df_cm['cm_x, a.u.'].tolist()
    coord_cm_y = df_cm['cm_y, a.u.'].tolist()
    coord_cm_z = df_cm['cm_z, a.u.'].tolist()
    plt.plot(coord_cm_x, coord_cm_y, coord_cm_z, color='red')
    plt.title(f"{problem.method.__name__} trajectory of center of mass ({problem.time_end} years)",
              fontsize=20, color="green")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')

    df_ts = pd.read_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), header=0, sep="\t")
    time = df_ts['time, years'].tolist()
    time_step = df_ts['time step, years'].tolist()
    plt.figure()
    plt.plot(time, time_step, color='red')
    plt.title(f"Magnitude of time step ({problem.method.__name__})", fontsize=20, color="green")
    plt.xlabel('time, years')
    plt.ylabel('time step, years')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "angular momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    ang_mom_x = df_mom['angular_momentum_x'].tolist()
    ang_mom_y = df_mom['angular_momentum_y'].tolist()
    ang_mom_z = df_mom['angular_momentum_z'].tolist()
    ang_mom_mag = df_mom['angular_momentum_mag'].tolist()

    ax.plot(ang_mom_x, ang_mom_y, ang_mom_z, color='blue')
    plt.title(f"{problem.method.__name__} vector of angular momentum", fontsize=20, color="blue")
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.figure()
    plt.plot(time, ang_mom_mag, color='cyan')
    plt.title(f"{problem.method.__name__} magnitude of vector of angular momentum", fontsize=20, color="black")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()
