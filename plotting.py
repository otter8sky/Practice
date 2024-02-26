from operations import *
from methods import methods
from methods import methods_names
from methods import By_Leap_Frog
from methods import By_RK_N
import pandas as pd
from pathlib import Path
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_stability_of_method(bodies, problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__

    delta_x = []
    delta_y = []
    delta_z = []
    delta_vx = []
    delta_vy = []
    delta_vz = []

    delta_a = []
    delta_e = []
    delta_i = []
    delta_l = []
    delta_p = []

    for i in range(len(bodies)):
        df_i = pd.read_csv(Path(Path.cwd(), "data", "stability", "stability_data", f"{method_name}",
                                f"{bodies[i].name}.txt"), header=0, sep="\t")

        delta_x.append([float(i) for i in df_i["delta_x, a.u."].tolist()])
        delta_y.append([float(i) for i in df_i["delta_y, a.u."].tolist()])
        delta_z.append([float(i) for i in df_i["delta_z, a.u."].tolist()])

        delta_vx.append([float(i) for i in df_i["delta_vx, a.u./year"].tolist()])
        delta_vy.append([float(i) for i in df_i["delta_vy, a.u./year"].tolist()])
        delta_vz.append([float(i) for i in df_i["delta_vz, a.u./year"].tolist()])

        delta_a.append([float(i) for i in df_i["delta_a, a.u."].tolist()])
        delta_e.append([float(i) for i in df_i["delta_e"].tolist()])
        delta_i.append([float(i) for i in df_i["delta_i, degrees"].tolist()])
        delta_l.append([float(i) for i in df_i["delta_l, degrees"].tolist()])
        delta_p.append([float(i) for i in df_i["delta_p, degrees"].tolist()])

    df = pd.read_csv(Path(Path.cwd(), "data", "stability", "stability_data", f"{method_name}",
                          "general.txt"), header=0, sep="\t")
    list_of_time_step = [float(i) for i in df["time step"].tolist()]
    delta_an_x = [float(i) for i in df["delta_an_x"].tolist()]
    delta_an_y = [float(i) for i in df["delta_an_y"].tolist()]
    delta_an_z = [float(i) for i in df["delta_an_z"].tolist()]
    delta_en = [float(i) for i in df["delta_en"].tolist()]

    for i in range(len(bodies)):
        plt.figure()
        plt.plot(list_of_time_step, delta_x[i], marker="o", color="red", label="delta_x")
        plt.plot(list_of_time_step, delta_y[i], marker="o", color="blue", label="delta_y")
        plt.plot(list_of_time_step, delta_z[i], marker="o", color="green", label="delta_z")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(f"{method_name}, {bodies[i].name}", fontsize=20, color="black")
        plt.xlabel('time step, years')
        plt.ylabel('delta, a.u.')
        plt.legend(loc='best')
        plt.savefig(Path(Path.cwd(), "data", "stability", "stability_plots",
                         f"{method_name}", f"{bodies[i].name}", f"delta_coord_{bodies[i].name}.jpg"))

        plt.figure()
        plt.plot(list_of_time_step, delta_vx[i], marker="o", color="yellow", label="delta_vx")
        plt.plot(list_of_time_step, delta_vy[i], marker="o", color="orange", label="delta_vy")
        plt.plot(list_of_time_step, delta_vz[i], marker="o", color="purple", label="delta_vz")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(f"{method_name}, {bodies[i].name}", fontsize=20, color="black")
        plt.xlabel('time step, years')
        plt.ylabel('delta, a.u./year')
        plt.legend(loc='best')
        plt.savefig(Path(Path.cwd(), "data", "stability", "stability_plots",
                         f"{method_name}", f"{bodies[i].name}", f"delta_vel_{bodies[i].name}.jpg"))

        plt.figure()
        plt.plot(list_of_time_step, delta_a[i], marker="o", color="green", label="delta_a")
        plt.plot(list_of_time_step, delta_e[i], marker="o", color="yellow", label="delta_e")
        plt.plot(list_of_time_step, delta_i[i], marker="o", color="orange", label="delta_i")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(f"{method_name}, {bodies[i].name}", fontsize=20, color="black")
        plt.xlabel('time step, years')
        plt.ylabel('delta, a.u./nan/degrees')
        plt.legend(loc='best')
        plt.savefig(Path(Path.cwd(), "data", "stability", "stability_plots",
                         f"{method_name}", f"{bodies[i].name}", f"delta_aei_{bodies[i].name}.jpg"))

        plt.figure()
        plt.plot(list_of_time_step, delta_l[i], marker="o", color="red", label="delta_l")
        plt.plot(list_of_time_step, delta_p[i], marker="o", color="purple", label="delta_p")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(f"{method_name}, {bodies[i].name}", fontsize=20, color="black")
        plt.xlabel('time step, years')
        plt.ylabel('delta, degrees')
        plt.legend(loc='best')
        plt.savefig(Path(Path.cwd(), "data", "stability", "stability_plots",
                         f"{method_name}", f"{bodies[i].name}", f"delta_lp_{bodies[i].name}.jpg"))

    plt.figure()
    plt.plot(list_of_time_step, delta_en, marker="o", color="red", label="delta_en")
    plt.plot(list_of_time_step, delta_an_x, marker="o", color="green", label="delta_an_x")
    plt.plot(list_of_time_step, delta_an_y, marker="o", color="blue", label="delta_an_y")
    plt.plot(list_of_time_step, delta_an_z, marker="o", color="yellow", label="delta_an_z")
    plt.yscale("log")
    plt.xscale("log")
    plt.title(f"{method_name}, general things", fontsize=20, color="black")
    plt.xlabel('time step, years')
    plt.ylabel('delta')
    plt.legend(loc='best')
    plt.savefig(Path(Path.cwd(), "data", "stability", "stability_plots",
                     f"{method_name}", "general.jpg"))


def plot_bodies(bodies, problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    time_step = problem.initial_timestep
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(bodies)):
        df_i = pd.read_csv(Path(Path.cwd(), "data", "data out", "objects", f"{bodies[i].name}_{time_step}.txt"), header=0, sep="\t")
        x_i = df_i['X, a.u.'].tolist()
        y_i = df_i['Y, a.u.'].tolist()
        z_i = df_i['Z, a.u.'].tolist()
        ax.plot(x_i, y_i, z_i, color=bodies[i].color, label=bodies[i].name)
    plt.title(f"{method_name} for {problem.time_end} year(s)", fontsize=20, color="purple")
    plt.xlabel('X, а.е.')
    plt.ylabel('Y, а.е.')
    plt.legend(loc='best')
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_elements(bodies, problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
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
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "energy.txt"), header=0, sep="\t")
    time = df_en['time, years'].tolist()
    energy = df_en['energy'].tolist()
    plt.plot(time, energy, color='red')

    plt.title(f"{method_name} energy", fontsize=20, color="red")
    plt.xlabel('time, years')
    plt.ylabel('energy')
    # plt.gca().set_aspect("equal")
    plt.show()


def plot_momentum(problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    momentum_x = df_mom['momentum_x'].tolist()
    momentum_y = df_mom['momentum_y'].tolist()
    momentum_z = df_mom['momentum_z'].tolist()
    momentum_mag = df_mom['momentum_mag'].tolist()

    plt.plot(momentum_x, momentum_y, color='blue')
    plt.title(f"{method_name} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')
    plt.show()

    plt.plot(time, momentum_mag, color='purple')
    plt.title(f"{method_name} magnitude of vector of total momentum", fontsize=20, color="purple")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()


def plot_angular_momentum(problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_mom = pd.read_csv(Path(Path.cwd(), "data", "data out", "angular momentum.txt"), header=0, sep="\t")
    time = df_mom['time, years'].tolist()
    ang_mom_x = df_mom['angular_momentum_x'].tolist()
    ang_mom_y = df_mom['angular_momentum_y'].tolist()
    ang_mom_z = df_mom['angular_momentum_z'].tolist()
    ang_mom_mag = df_mom['angular_momentum_mag'].tolist()

    ax.plot(ang_mom_x, ang_mom_y, ang_mom_z, color='blue')
    plt.title(f"{method_name} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.plot(time, ang_mom_mag, color='cyan')
    plt.title(f"{method_name} magnitude of vector of angular momentum", fontsize=20, color="black")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()


def plot_cm(problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    fig, ax = plt.subplots()
    df_en = pd.read_csv(Path(Path.cwd(), "data", "data out", "center_mass.txt"), header=0, sep="\t")
    time = df_en['time, years'].tolist()
    coord_cm_x = df_en['cm_x, a.u.'].tolist()
    coord_cm_y = df_en['cm_y, a.u.'].tolist()
    coord_cm_z = df_en['cm_z, a.u.'].tolist()
    plt.plot(coord_cm_x, coord_cm_y, color='red')

    plt.title(f"{method_name} trajectory of center of mass ({problem.time_end} years)",
              fontsize=20, color="green")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')
    plt.show()


def plot_time_step(problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    df_ts = pd.read_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), header=0, sep="\t")
    time = df_ts['time, years'].tolist()
    time_step = df_ts['time step, years'].tolist()
    plt.plot(time, time_step, color='red')
    plt.title(f"Magnitude of time step ({method_name})", fontsize=20, color="green")
    plt.xlabel('time, years')
    plt.ylabel('time step, years')
    plt.show()


def plot_all(problem, bodies):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    plt.figure()
    for i in range(len(bodies)):
        df_i = pd.read_csv(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), header=0, sep="\t")
        x_i = df_i['X, a.u.'].tolist()
        y_i = df_i['Y, a.u.'].tolist()
        z_i = df_i['Z, a.u.'].tolist()
        plt.plot(x_i, y_i, color=bodies[i].color, label=bodies[i].name)

    plt.title(f"{method_name} for {problem.time_end} year(s)", fontsize=20, color="purple")
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

    plt.title(f"{method_name} energy", fontsize=20, color="red")
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
    plt.title(f"{method_name} vector of total momentum", fontsize=20, color="blue")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')

    plt.figure()
    plt.plot(time, momentum_mag, color='purple')
    plt.title(f"{method_name} magnitude of vector of total momentum", fontsize=20, color="purple")
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
    plt.title(f"{method_name} trajectory of center of mass ({problem.time_end} years)",
              fontsize=20, color="green")
    plt.xlabel('X, a.u.')
    plt.ylabel('Y, a.u.')

    df_ts = pd.read_csv(Path(Path.cwd(), "data", "data out", "time_step.txt"), header=0, sep="\t")
    time = df_ts['time, years'].tolist()
    time_step = df_ts['time step, years'].tolist()
    plt.figure()
    plt.plot(time, time_step, color='red')
    plt.title(f"Magnitude of time step ({method_name})", fontsize=20, color="green")
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
    plt.title(f"{method_name} vector of angular momentum", fontsize=20, color="blue")
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.figure()
    plt.plot(time, ang_mom_mag, color='cyan')
    plt.title(f"{method_name} magnitude of vector of angular momentum", fontsize=20, color="black")
    plt.xlabel('time, years')
    plt.ylabel('magnitude')
    plt.show()
