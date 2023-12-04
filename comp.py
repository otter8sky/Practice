from operations import *
from methods import methods
from methods import methods_names
from methods import By_Leap_Frog
import pandas as pd
from pathlib import Path
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
    problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep, delta_vel,
                      delta_coord, delta_timestep, timestep_max, timestep_min)
    return problem
def read_data_file(file_name):
    df_data = pd.read_csv(Path(Path.cwd(), "data", "initial data", f"{file_name}"), header=0, sep="\t")
    names = df_data['Object'].tolist()
    coordinates_x = df_data['X, a.u.'].tolist()
    coordinates_y = df_data['Y, a.u.'].tolist()
    coordinates_z = df_data['Z, a.u.'].tolist()

    velocities_x = df_data["Vx, a.u./year"].tolist()
    velocities_y = df_data['Vy, a.u./year'].tolist()
    velocities_z = df_data['Vz, a.u./year'].tolist()

    masses = df_data['Mass, M_sun'].tolist()
    colors = df_data['color'].tolist()
    bodies = []
    for i in range(len(names)):
        bodies.append(Body([float(velocities_x[i]), float(velocities_y[i]), float(velocities_z[i])],
                           [float(coordinates_x[i]), float(coordinates_y[i]), float(coordinates_z[i])],
                           float(masses[i]), [0, 0, 0], names[i], colors[i]))
    bodies = get_acs_for_all(bodies)
    return bodies
def comp(problem, bodies):
    cnt = 0
    t = 0
    time_step = problem.initial_timestep
    timestep = [time_step]
    bodies_coord = [[[bodies[i].coord[0]], [bodies[i].coord[1]], [bodies[i].coord[2]]] for i in range(len(bodies))]
    bodies_vel = [[[bodies[i].vel[0]], [bodies[i].vel[1]], [bodies[i].vel[2]]] for i in range(len(bodies))]
    energy = [get_Energy(bodies)]
    time_full = [t]
    time = [t]
    cm_coord = [[], [], []]
    mom_coord = [[], [], []]
    mag_mom = []
    while t <= problem.time_end:
        print(t * 100/problem.time_end, "%")
        time_full.append(t)
        if t == 0 and problem.method == By_Leap_Frog:
            for i in range(len(bodies)):
                bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))
                # bodies_vel[i][0] = bodies[i].vel
        time_step = get_time_step(bodies, time_step, problem)
        timestep.append(time_step)
        result = problem.method(bodies, time_step)
        if cnt == 1:
            for i in range(len(bodies)):
                bodies_coord[i][0].append(bodies[i].coord[0])
                bodies_coord[i][1].append(bodies[i].coord[1])
                bodies_coord[i][2].append(bodies[i].coord[2])
                bodies_vel[i][0].append(bodies[i].vel[0])
                bodies_vel[i][1].append(bodies[i].vel[1])
                bodies_vel[i][2].append(bodies[i].vel[2])

            cm_coord[0].append(get_coord_cm(bodies)[0])
            cm_coord[1].append(get_coord_cm(bodies)[1])
            cm_coord[2].append(get_coord_cm(bodies)[2])
            mom_coord[0].append(get_vect_total_momentum(bodies)[0])
            mom_coord[1].append(get_vect_total_momentum(bodies)[1])
            mom_coord[2].append(get_vect_total_momentum(bodies)[2])
            mag_mom.append(get_mag(get_vect_total_momentum(bodies)))
            time.append(t)
            energy.append(get_Energy(result))
            cnt = 0
        cnt += 1
        t += time_step
    if problem.method == By_Leap_Frog:
        for i in range(len(bodies)):
            bodies[i].vel = add(bodies[i].vel, mult(bodies[i].acs, -time_step / 2))
            # bodies_vel.append(bodies[i].vel)

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

    mom_x = pd.Series(mom_coord[0]).to_frame(name="momentum_x, a.u.")
    mom_y = pd.Series(mom_coord[1]).to_frame(name="momentum_y, a.u.")
    mom_z = pd.Series(mom_coord[2]).to_frame(name="momentum_z, a.u.")
    mag_mom = pd.Series(mag_mom).to_frame(name="momentum_mag")
    df_cm = pd.concat([time, mom_x, mom_y, mom_z, mag_mom], axis=1)
    df_cm.to_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), sep="\t")

    for i in range(len(bodies)):
        x_i = pd.Series(bodies_coord[i][0]).to_frame(name="X, a.u.")
        y_i = pd.Series(bodies_coord[i][1]).to_frame(name="Y, a.u.")
        z_i = pd.Series(bodies_coord[i][2]).to_frame(name="Z, a.u.")
        vel_x_i = pd.Series(bodies_vel[i][0]).to_frame(name="V_x, years")
        vel_y_i = pd.Series(bodies_vel[i][1]).to_frame(name="V_y, a.u./year")
        vel_z_i = pd.Series(bodies_vel[i][2]).to_frame(name="V_z, a.u./year")
        df_i = pd.concat([time, x_i, y_i, z_i, vel_x_i, vel_y_i, vel_z_i], axis=1)
        df_i.to_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), sep="\t")

def plot_bodies(bodies, problem):
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
    momentum_x = df_mom['momentum_y, a.u.'].tolist()
    momentum_y = df_mom['momentum_y, a.u.'].tolist()
    momentum_z = df_mom['momentum_z, a.u.'].tolist()
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

    plt.figure()
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    momentum_x = df_mom['momentum_y, a.u.'].tolist()
    momentum_y = df_mom['momentum_y, a.u.'].tolist()
    momentum_z = df_mom['momentum_z, a.u.'].tolist()
    momentum_mag = df_mom['momentum_mag'].tolist()
    plt.plot(momentum_x, momentum_y, color='blue')
    plt.title(f"{problem.method.__name__} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')

    plt.figure()
    plt.plot(time, momentum_mag, color='purple')
    plt.title(f"{problem.method.__name__} magnitude of vector of total momentum", fontsize=20, color="purple")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')

    plt.figure()
    df_cm = pd.read_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), header=0, sep="\t")
    time = df_cm['time, years'].tolist()
    coord_cm_x = df_cm['cm_x, a.u.'].tolist()
    coord_cm_y = df_cm['cm_y, a.u.'].tolist()
    coord_cm_z = df_cm['cm_z, a.u.'].tolist()
    plt.plot(coord_cm_x, coord_cm_y, color='red')
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
    plt.show()
