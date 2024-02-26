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


def descartes_from_kepler(name, bodies, r, v, i, omega, big_omega, q, mass, color):
    a_inv = (2 / r - v ** 2 / G)
    a = 1 / a_inv
    e = 1 - q * a_inv
    p = 1 / a_inv * (1 - e ** 2)
    f = 2 * np.pi - np.arccos((p - r) / (e * r))

    r_dot = np.sqrt(G * M_sun / p) * e * np.sin(f)
    r_f_dot = np.sqrt(G * M_sun / p) * (1 + e * np.cos(f))

    alpha = np.cos(big_omega) * np.cos(f + omega) - np.sin(big_omega) * np.sin(f + omega) * np.cos(i)
    betta = np.sin(big_omega) * np.cos(f + omega) - np.cos(big_omega) * np.sin(f + omega) * np.cos(i)
    gamma = np.sin(f + omega) * np.sin(i)

    alpha_stroke = - np.cos(big_omega) * np.sin(f + omega) - np.sin(big_omega) * np.cos(f + omega) * np.cos(i)
    betta_stroke = - np.sin(big_omega) * np.sin(f + omega) - np.cos(big_omega) * np.cos(f + omega) * np.cos(i)
    gamma_stroke = np.cos(f + omega) * np.sin(i)

    x = r * alpha
    y = r * betta
    z = r * gamma

    v_x = r_dot * alpha + r_f_dot * alpha_stroke
    v_y = r_dot * betta + r_f_dot * betta_stroke
    v_z = r_dot * gamma + r_f_dot * gamma_stroke

    bodies.append(Body([v_x, v_y, v_z],
                       [v_x, v_y, v_z],
                       [x, y, z],
                       float(mass), [0, 0, 0], name, color))

    return bodies


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
    bodies = get_acs_for_all(bodies)
    return bodies


def comp(problem, bodies):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
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

    while t < problem.time_end:
        print("\r" + str(round(t * 100 / problem.time_end, 2)) + "%", end="")
        # print(round(t * 100 / problem.time_end, 2), "%")
        time_full.append(t)
        if t == 0 and problem.method[0] == By_Leap_Frog:
            for i in range(len(bodies)):
                bodies[i].half_vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))

        if problem.method[0] == By_RK_N:
            result = problem.method[0](bodies, time_step, problem.method[1])
        else:
            result = problem.method[0](bodies, time_step)

        if abs(t - next_output) < time_step / 2 and abs(t - problem.time_end) > time_step / 2:
            for i in range(len(bodies)):
                bodies_coord[i] = fill_coord_list_3d(bodies_coord[i], bodies[i])[:]
                bodies_vel[i] = fill_vel_list_3d(bodies_vel[i], bodies[i])[:]

                bodies_axis[i].append(major_axis(bodies, i))
                bodies_ecc[i].append(eccentricity(bodies, i))
                bodies_inc[i].append(inclination(bodies, i))
                bodies_long[i].append(longitude_of_asc_node(bodies, i))
                bodies_per[i].append(arg_of_periapsis(bodies, i))
            timestep.append(time_step)
            cm_coord = fill_list_3d_func(cm_coord, get_coord_cm, bodies)[:]
            mom_coord = fill_list_3d_func(mom_coord, get_total_momentum, bodies)[:]
            mag_mom.append(get_mag(get_total_momentum(bodies)))
            ang_mom_coord = fill_list_3d_func(ang_mom_coord, get_total_angular_momentum, bodies)[:]
            mag_ang_mom.append(get_mag(get_total_angular_momentum(bodies)))
            time.append(t)
            energy.append(get_Energy(result))

            next_output += problem.dt_output
        if not abs(problem.time_end - t) <= 1e-6:
            t += time_step
        else:
            break

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

    time = pd.Series(time).to_frame(name="time, years")
    timestep = pd.Series(timestep).to_frame(name="time step, years")

    energy = pd.Series(energy).to_frame(name="energy")

    cm_x = pd.Series(cm_coord[0]).to_frame(name="cm_x, a.u.")
    cm_y = pd.Series(cm_coord[1]).to_frame(name="cm_y, a.u.")
    cm_z = pd.Series(cm_coord[2]).to_frame(name="cm_z, a.u.")

    mom_x = pd.Series(mom_coord[0]).to_frame(name="momentum_x")
    mom_y = pd.Series(mom_coord[1]).to_frame(name="momentum_y")
    mom_z = pd.Series(mom_coord[2]).to_frame(name="momentum_z")
    mag_mom = pd.Series(mag_mom).to_frame(name="momentum_mag")

    ang_mom_x = pd.Series(ang_mom_coord[0]).to_frame(name="angular_momentum_x")
    ang_mom_y = pd.Series(ang_mom_coord[1]).to_frame(name="angular_momentum_y")
    ang_mom_z = pd.Series(ang_mom_coord[2]).to_frame(name="angular_momentum_z")
    ang_mag_mom = pd.Series(mag_ang_mom).to_frame(name="angular_momentum_mag")
    df = pd.concat([time, timestep, energy, cm_x, cm_y, cm_z, mom_x, mom_y, mom_z, mag_mom,
                    ang_mom_x, ang_mom_y, ang_mom_z, ang_mag_mom], axis=1)
    df.to_csv(Path(Path.cwd(), "data", "data out", "general data", f"{method_name}_general_{time_step}.txt"), sep="\t")

    for i in range(len(bodies)):
        x_i = pd.Series(bodies_coord[i][0]).to_frame(name="X, a.u.")
        y_i = pd.Series(bodies_coord[i][1]).to_frame(name="Y, a.u.")
        z_i = pd.Series(bodies_coord[i][2]).to_frame(name="Z, a.u.")
        vel_x_i = pd.Series(bodies_vel[i][0]).to_frame(name="V_x, a.u./year")
        vel_y_i = pd.Series(bodies_vel[i][1]).to_frame(name="V_y, a.u./year")
        vel_z_i = pd.Series(bodies_vel[i][2]).to_frame(name="V_z, a.u./year")

        a_i = pd.Series(bodies_axis[i]).to_frame(name="a, a.u.")
        e_i = pd.Series(bodies_ecc[i]).to_frame(name="e")
        i_i = pd.Series(bodies_inc[i]).to_frame(name="i, degrees")
        long_of_asc_node_i = pd.Series(bodies_long[i]).to_frame(name="long_of_asc_node, degrees")
        arg_of_periapsis_i = pd.Series(bodies_per[i]).to_frame(name="arg_of_periapsis, degrees")
        # f"{method_name}",
        df_i = pd.concat([time, x_i, y_i, z_i, vel_x_i, vel_y_i, vel_z_i, a_i, e_i, i_i,
                          long_of_asc_node_i, arg_of_periapsis_i], axis=1)
        df_i.to_csv(Path(Path.cwd(), "data", "data out", "objects",
                         f"{bodies[i].name}_{problem.initial_timestep}.txt"), sep="\t")


def get_ideal_data_stability_of_method(bodies, problem):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    t = 0
    time_step = 1e-6

    x_ideal = []
    y_ideal = []
    z_ideal = []

    vx_ideal = []
    vy_ideal = []
    vz_ideal = []

    a_ideal = []
    e_ideal = []
    i_ideal = []
    l_ideal = []
    p_ideal = []

    while t <= problem.time_end:
        print("computing ideal:", round(t * 100 / problem.time_end, 2), "%")
        if t == 0 and problem.method[0] == By_Leap_Frog:
            for i in range(len(bodies)):
                bodies[i].half_vel = add(bodies[i].vel, mult(bodies[i].acs, time_step / 2))
        if problem.method[0] == By_RK_N:
            problem.method[0](bodies, time_step, problem.method[1])
        else:
            problem.method[0](bodies, time_step)
        if not abs(problem.time_end - t) <= 1e-6:
            t += time_step
        else:
            break

    for i in range(len(bodies)):
        x_ideal.append(bodies[i].coord[0])
        y_ideal.append(bodies[i].coord[1])
        z_ideal.append(bodies[i].coord[2])

        vx_ideal.append(bodies[i].vel[0])
        vy_ideal.append(bodies[i].vel[1])
        vz_ideal.append(bodies[i].vel[2])

        a_ideal.append(major_axis(bodies, i))
        e_ideal.append(eccentricity(bodies, i))
        i_ideal.append(inclination(bodies, i))

        l_ideal.append(longitude_of_asc_node(bodies, i))
        p_ideal.append(arg_of_periapsis(bodies, i))

    en_ideal = get_Energy(bodies)
    an_ideal_x = get_total_angular_momentum(bodies)[0]
    an_ideal_y = get_total_angular_momentum(bodies)[1]
    an_ideal_z = get_total_angular_momentum(bodies)[2]

    x_ideal = pd.Series(x_ideal).to_frame(name="x_ideal, a.u.")
    y_ideal = pd.Series(y_ideal).to_frame(name="y_ideal, a.u.")
    z_ideal = pd.Series(z_ideal).to_frame(name="z_ideal, a.u.")

    vx_ideal = pd.Series(vx_ideal).to_frame(name="vx_ideal, a.u./year")
    vy_ideal = pd.Series(vy_ideal).to_frame(name="vy_ideal, a.u./year")
    vz_ideal = pd.Series(vz_ideal).to_frame(name="vz_ideal, a.u./year")

    a_ideal = pd.Series(a_ideal).to_frame(name="a_ideal, a.u.")
    e_ideal = pd.Series(e_ideal).to_frame(name="e_ideal")
    i_ideal = pd.Series(i_ideal).to_frame(name="i_ideal, degrees")
    l_ideal = pd.Series(l_ideal).to_frame(name="l_ideal, degrees")
    p_ideal = pd.Series(p_ideal).to_frame(name="p_ideal, degrees")

    an_ideal_x = pd.Series(an_ideal_x).to_frame(name="an_ideal_x")
    an_ideal_y = pd.Series(an_ideal_y).to_frame(name="an_ideal_y")
    an_ideal_z = pd.Series(an_ideal_z).to_frame(name="an_ideal_z")

    en_ideal = pd.Series(en_ideal).to_frame(name="en_ideal")

    df_ts = pd.concat([x_ideal, y_ideal, z_ideal,
                       vx_ideal, vy_ideal, vz_ideal,
                       a_ideal, e_ideal, i_ideal,
                       l_ideal, p_ideal, en_ideal,
                       an_ideal_x, an_ideal_y, an_ideal_z], axis=1)
    df_ts.to_csv(Path(Path.cwd(), "data", "stability", "ideal_data", f"{method_name}",
                      "ideal.txt"), sep="\t")


def get_data_stability_of_method(bodies, problem, list_of_time_step):
    if str(problem.method[0].__name__) == "By_RK_N":
        method_name = "By_RK_" + str(problem.method[1])
    else:
        method_name = problem.method[0].__name__
    df_data = pd.read_csv(Path(Path.cwd(), "data", "stability", "ideal_data",
                               f"{method_name}", "ideal.txt"), header=0, sep="\t")

    x_ideal = [float(i) for i in df_data['x_ideal, a.u.'].tolist()]
    y_ideal = [float(i) for i in df_data['y_ideal, a.u.'].tolist()]
    z_ideal = [float(i) for i in df_data['z_ideal, a.u.'].tolist()]

    vx_ideal = [float(i) for i in df_data['vx_ideal, a.u./year'].tolist()]
    vy_ideal = [float(i) for i in df_data['vy_ideal, a.u./year'].tolist()]
    vz_ideal = [float(i) for i in df_data['vz_ideal, a.u./year'].tolist()]

    a_ideal = [float(i) for i in df_data['a_ideal, a.u.'].tolist()]
    e_ideal = [float(i) for i in df_data['e_ideal'].tolist()]
    i_ideal = [float(i) for i in df_data['i_ideal, degrees'].tolist()]

    l_ideal = [float(i) for i in df_data['l_ideal, degrees'].tolist()]
    p_ideal = [float(i) for i in df_data['p_ideal, degrees'].tolist()]

    an_ideal_x = [float(i) for i in df_data['an_ideal_x'].tolist()][0]
    an_ideal_y = [float(i) for i in df_data['an_ideal_y'].tolist()][0]
    an_ideal_z = [float(i) for i in df_data['an_ideal_z'].tolist()][0]

    en_ideal = [float(i) for i in df_data['en_ideal'].tolist()][0]

    delta_x = [[] for i in range(len(bodies))]
    delta_y = [[] for i in range(len(bodies))]
    delta_z = [[] for i in range(len(bodies))]
    delta_vx = [[] for i in range(len(bodies))]
    delta_vy = [[] for i in range(len(bodies))]
    delta_vz = [[] for i in range(len(bodies))]
    delta_a = [[] for i in range(len(bodies))]
    delta_e = [[] for i in range(len(bodies))]
    delta_i = [[] for i in range(len(bodies))]
    delta_l = [[] for i in range(len(bodies))]
    delta_p = [[] for i in range(len(bodies))]

    delta_en = []
    delta_an_x = []
    delta_an_y = []
    delta_an_z = []

    for time_step in list_of_time_step:
        for i in range(len(bodies)):
            df_data_i = pd.read_csv(
                Path(Path.cwd(), "data", "data out", "objects", f"{method_name}",
                     f"{bodies[i].name}_{time_step}.txt"), header=0, sep="\t")
            x_i = [float(i) for i in df_data_i['X, a.u.'].tolist()]
            y_i = [float(i) for i in df_data_i['Y, a.u.'].tolist()]
            z_i = [float(i) for i in df_data_i['Z, a.u.'].tolist()]

            vx_i = [float(i) for i in df_data_i['V_x, a.u./year'].tolist()]
            vy_i = [float(i) for i in df_data_i['V_y, a.u./year'].tolist()]
            vz_i = [float(i) for i in df_data_i['V_z, a.u./year'].tolist()]

            a_i = [float(i) for i in df_data_i['a, a.u.'].tolist()]
            e_i = [float(i) for i in df_data_i['e'].tolist()]
            i_i = [float(i) for i in df_data_i['i, degrees'].tolist()]
            l_i = [float(i) for i in df_data_i['long_of_asc_node, degrees'].tolist()]
            p_i = [float(i) for i in df_data_i['arg_of_periapsis, degrees'].tolist()]

            delta_x[i].append(abs(x_i[-1] - x_ideal[i]))
            delta_y[i].append(abs(y_i[-1] - y_ideal[i]))
            delta_z[i].append(abs(z_i[-1] - z_ideal[i]))

            delta_vx[i].append(abs(vx_i[-1] - vx_ideal[i]))
            delta_vy[i].append(abs(vy_i[-1] - vy_ideal[i]))
            delta_vz[i].append(abs(vz_i[-1] - vz_ideal[i]))

            delta_a[i].append(abs(a_i[-1] - a_ideal[i]))
            delta_e[i].append(abs(e_i[-1] - e_ideal[i]))
            delta_i[i].append(abs(i_i[-1] - i_ideal[i]))
            delta_l[i].append(abs(l_i[-1] - l_ideal[i]))
            delta_p[i].append(abs(p_i[-1] - p_ideal[i]))

        # df_data_general = pd.read_csv(Path(Path.cwd(), "data", "data out", "general data",
        #                                    f"{method_name}_general_{time_step}.txt"), header=0, sep="\t")
        #
        # an_x_ts = [float(i) for i in df_data_general['angular_momentum_x'].tolist()]
        # an_y_ts = [float(i) for i in df_data_general['angular_momentum_y'].tolist()]
        # an_z_ts = [float(i) for i in df_data_general['angular_momentum_z'].tolist()]
        # en_ts = [float(i) for i in df_data_general['energy'].tolist()]
        #
        # delta_en.append(abs(en_ts[-1] - en_ideal))
        # delta_an_x.append(abs(an_x_ts[-1] - an_ideal_x))
        # delta_an_y.append(abs(an_y_ts[-1] - an_ideal_y))
        # delta_an_z.append(abs(an_z_ts[-1] - an_ideal_z))
    # TODO: cделать общие дельты тоже
    delta_object = [delta_en, delta_an_x, delta_an_y, delta_an_z]
    delta_general = [delta_x, delta_y, delta_z,
                     delta_vx, delta_vy, delta_vz,
                     delta_a, delta_e, delta_i, delta_l, delta_p]
    list_of_time_step = pd.Series(list_of_time_step).to_frame(name="time step")
    names_of_lines = pd.Series(["x", "x", "",
                                "y", "y", "",
                                "z", "z", "",
                                "vx", "vx", "",
                                "vy", "vy", "",
                                "vz", "vz", "",
                                "a", "a", "",
                                "e", "e", "",
                                "i", "i", "",
                                "l", "l", "",
                                "p", "p", ""]).to_frame(name="lines")

    orders = [[] for i in range(len(bodies))]
    for i in range(len(bodies)):
        for k in range(len(delta_object)):
            for j in range(0, len(list_of_time_step) - 1):
                try:
                    orders[i].append(round(np.log10(abs(delta_object[k][i][j] / delta_object[k][i][j + 1])), 4))
                except:
                    orders[i].append("*")
            orders[i].append("-------------")
        orders[i] = pd.Series(orders[i]).to_frame(name=f"{bodies[i].name}")
    orders.insert(0, names_of_lines)
    df_orders = pd.concat(orders, axis=1)
    df_orders.to_csv(Path(Path.cwd(), "data", "stability", "orders.txt"), sep="\t")

    for i in range(len(bodies)):
        delta_x_i = pd.Series(delta_x[i]).to_frame(name="delta_x, a.u.")
        delta_y_i = pd.Series(delta_y[i]).to_frame(name="delta_y, a.u.")
        delta_z_i = pd.Series(delta_z[i]).to_frame(name="delta_z, a.u.")

        delta_vx_i = pd.Series(delta_vx[i]).to_frame(name="delta_vx, a.u./year")
        delta_vy_i = pd.Series(delta_vy[i]).to_frame(name="delta_vy, a.u./year")
        delta_vz_i = pd.Series(delta_vz[i]).to_frame(name="delta_vz, a.u./year")

        delta_a_i = pd.Series(delta_a[i]).to_frame(name="delta_a, a.u.")
        delta_e_i = pd.Series(delta_e[i]).to_frame(name="delta_e")
        delta_i_i = pd.Series(delta_i[i]).to_frame(name="delta_i, degrees")
        delta_l_i = pd.Series(delta_l[i]).to_frame(name="delta_l, degrees")
        delta_p_i = pd.Series(delta_p[i]).to_frame(name="delta_p, degrees")

        df_i = pd.concat([list_of_time_step, delta_x_i, delta_y_i, delta_z_i,
                          delta_vx_i, delta_vy_i, delta_vz_i,
                          delta_a_i, delta_e_i, delta_i_i, delta_l_i, delta_p_i], axis=1)
        df_i.to_csv(Path(Path.cwd(), "data", "stability", "stability_data",
                         f"{method_name}", f"{bodies[i].name}.txt"), sep="\t")

    delta_an_x = pd.Series(delta_an_x).to_frame(name="delta_an_x")
    delta_an_y = pd.Series(delta_an_y).to_frame(name="delta_an_y")
    delta_an_z = pd.Series(delta_an_z).to_frame(name="delta_an_z")
    delta_en = pd.Series(delta_en).to_frame(name="delta_en")

    df = pd.concat([list_of_time_step, delta_en, delta_an_x, delta_an_y, delta_an_z], axis=1)
    df.to_csv(Path(Path.cwd(), "data", "stability", "stability_data",
                   f"{method_name}", "general.txt"), sep="\t")
