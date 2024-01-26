from const import *
from pathlib import Path
import pandas as pd

class Body:
    def __init__(self, vel, half_vel, coord, mass, acs, name, color):
        self.vel = vel
        self.half_vel = half_vel
        self.coord = coord
        self.mass = mass
        self.acs = acs
        self.name = name
        self.color = color
        # self.radius = radius
class Problem:
    def __init__(self, method, time_end, initial_timestep, delta_vel,
                 delta_coord, delta_timestep, timestep_max, timestep_min, dt_output):
        self.method = method
        self.time_end = time_end
        self.initial_timestep = initial_timestep
        self.delta_vel = delta_vel
        self.delta_timestep = delta_timestep
        self.delta_coord = delta_coord
        self.timestep_max = timestep_max
        self.timestep_min = timestep_min
        self.dt_output = dt_output
class Method:
    def __init__(self, name):
        self.name = name

    def add_to_list(self, methods):
        methods.append(self.name)
class Angle:
    def __init__(self, value):
        value.strip('"')
        degrees, min_sec = value.split('°')
        minutes, seconds = min_sec.split("'")
        value = pi * (float(degrees[1:]) + float(minutes) / 60 + float(seconds) / 3600) / 180
        self.value = value
        self.sign = str(degrees[:1])
        self.degrees = float(degrees[1:])
        self.minutes = float(minutes)
        self.seconds = float(seconds)
    # FIXME: create a function that changes angle to hours (def change_to_hours(self):)

def read_strange_data_file(file_name):
    df_data = pd.read_csv(Path(Path.cwd(), "data", "initial data", f"{file_name}"), header=0, sep="\t")
    names = df_data['Name'].tolist()
    colors = df_data['Color'].tolist()
    masses = df_data['Mass'].tolist()
    velocities = df_data['Velocity'].tolist()
    delta = df_data['Latitude'].tolist()
    alpha = df_data['Longitude'].tolist()
    distance = df_data['Distance'].tolist()
    d_alpha = df_data['d/alpha'].tolist()
    d_delta = df_data['d/delta'].tolist()
    # a = np.array....
    for i in range(len(alpha)):
        delta[i] = Angle(delta[i])
        alpha[i] = Angle(alpha[i])
        d_alpha[i] = Angle(d_alpha[i])
        d_delta[i] = Angle(d_delta[i])

    bodies = []
    hour = 3600

    for i in range(len(names)):
        x_i = distance[i] * np.cos(delta[i].value) * np.cos(alpha[i].value)
        y_i = distance[i] * np.cos(delta[i].value) * np.sin(alpha[i].value)
        z_i = distance[i] * np.sin(delta[i].value)

        vel_x_i = change_vel(velocities[i] * np.cos(delta[i].value) * np.cos(alpha[i].value) - distance[i] *
                             d_delta[i].value / hour * np.sin(delta[i].value) * np.cos(alpha[i].value) - distance[i] *
                             d_alpha[i].value / hour * np.cos(delta[i].value) * np.sin(alpha[i].value))
        vel_y_i = change_vel(velocities[i] * np.cos(delta[i].value) * np.sin(alpha[i].value) - distance[i] *
                             d_delta[i].value / hour * np.sin(delta[i].value) * np.sin(alpha[i].value) + distance[i] *
                             d_alpha[i].value / hour * np.cos(delta[i].value) * np.cos(alpha[i].value))
        vel_z_i = change_vel(velocities[i] * np.sin(delta[i].value) + distance[i] *
                             d_alpha[i].value / hour * np.cos(delta[i].value))

        bodies.append(Body([vel_x_i, vel_y_i, vel_z_i], [x_i, y_i, z_i],
                           float(masses[i]), [0, 0, 0], names[i], colors[i]))
    bodies = get_acs_for_all(bodies)
    return bodies

def vector_product(v, u):
    p = [v[1] * u[2] - v[2] * u[1], -(v[0] * u[2] - v[2] * u[0]), v[0] * u[1] - v[1] * u[0]]
    return p
def scalar_product(v, u):
    return v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
def mult(v, a):
    u = [v[0] * a, v[1] * a, v[2] * a]
    return u
def add(a, b):
    c = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    return c
def get_r(a, b):
    r = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return r
def get_mag(r):
    mag = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
    return mag

def get_k(bodies, a_list_1, k_list, h):
    result = copy(bodies)
    for j in range(len(bodies)):
        for i in range(len(k_list)):
            result[j].coord[0] += k_list[i][j].vel[0] * a_list_1[i] * h
            result[j].coord[1] += k_list[i][j].vel[1] * a_list_1[i] * h
            result[j].coord[2] += k_list[i][j].vel[2] * a_list_1[i] * h

            result[j].vel[0] += k_list[i][j].acs[0] * a_list_1[i] * h
            result[j].vel[1] += k_list[i][j].acs[1] * a_list_1[i] * h
            result[j].vel[2] += k_list[i][j].acs[2] * a_list_1[i] * h
    result = get_acs_for_all(copy(result))
    return result
def copy(bodies):
    copied_bodies = []
    for i in range(len(bodies)):
        copied_bodies.append(Body(bodies[i].vel[:],
                                  bodies[i].half_vel[:],
                                  bodies[i].coord[:],
                                  bodies[i].mass,
                                  bodies[i].acs[:],
                                  bodies[i].name,
                                  bodies[i].color))
    return copied_bodies

def get_acs(m, r):
    a = mult(r, -G * m / (get_mag(r)) ** 3)
    return a
def get_total_acs(bodies, i):
    acs_total_i = [0.0, 0.0, 0.0]
    for j in range(len(bodies)):
        if i != j:
            acs_total_i = add(acs_total_i, get_acs(bodies[j].mass, get_r(bodies[i].coord, bodies[j].coord)))
    return acs_total_i
def get_acs_for_all(bodies):
    result = copy(bodies)
    for i in range(len(bodies)):
        result[i].acs = get_total_acs(bodies, i)
    return result
def get_acs_gtr(m, r):
    a = mult(r, -G * m / ((get_mag(r) - r_g) ** 2 * get_mag(r)))
    return a
def get_total_acs_gtr(bodies, i):
    acs_total_i = [0.0, 0.0, 0.0]
    for j in range(len(bodies)):
        if i != j:
            acs_total_i = add(acs_total_i, get_acs_gtr(bodies[j].mass, get_r(bodies[i].coord, bodies[j].coord)))
    return acs_total_i
def get_acs_for_all_gtr(bodies):
    result = copy(bodies)
    for i in range(len(bodies)):
        result[i].acs = get_total_acs_gtr(bodies, i)
    return result

def get_grav_pot(body_coords, body_mass, point_coords):
    r = get_mag(get_r(body_coords, point_coords))
    potential = - G * body_mass / r
    return potential
def get_total_grav_pot(index, bodies):
    total_potential = 0
    for i in range(len(bodies)):
        if i != index:
            total_potential += get_grav_pot(bodies[i].coord, bodies[i].mass, bodies[index].coord)
    return total_potential
def get_Potential_Energy(bodies):
    potential_energy = 0
    for i in range(len(bodies)):
        potential_energy += get_total_grav_pot(i, bodies) * bodies[i].mass / 2
    return potential_energy
def get_Kinetical_Energy(bodies):
    kinetical_energy = 0
    for i in range(len(bodies)):
        kinetical_energy += bodies[i].mass * get_mag(bodies[i].vel) ** 2 / 2
    return kinetical_energy
def get_Energy(bodies):
    energy = get_Kinetical_Energy(bodies) + get_Potential_Energy(bodies)
    return energy

def get_coord_cm(bodies):
    coord_cm = []
    sum_xi_mi = 0
    sum_yi_mi = 0
    sum_zi_mi = 0
    sum_mi = 0
    for i in range(len(bodies)):
        sum_xi_mi += bodies[i].coord[0] * bodies[i].mass
        sum_yi_mi += bodies[i].coord[1] * bodies[i].mass
        sum_zi_mi += bodies[i].coord[2] * bodies[i].mass
        sum_mi += bodies[i].mass
    coord_cm.append(sum_xi_mi / sum_mi)
    coord_cm.append(sum_yi_mi / sum_mi)
    coord_cm.append(sum_zi_mi / sum_mi)
    return coord_cm
def get_momentum(bodies, i):
    momentum = mult(bodies[i].vel, bodies[i].mass)
    return momentum
def get_total_momentum(bodies):
    vect_total_momentum = [0, 0, 0]
    for i in range(len(bodies)):
        vect_total_momentum = add(vect_total_momentum, get_momentum(bodies, i))[:]
    return vect_total_momentum
def get_angular_momentum(bodies, i):
    impulse_moment = vector_product(bodies[i].coord, get_momentum(bodies, i))[:]
    return impulse_moment
def get_total_angular_momentum(bodies):
    total_impulse_moment = [0, 0, 0]
    for i in range(len(bodies)):
        total_impulse_moment = add(total_impulse_moment, get_angular_momentum(bodies, i))[:]
    return total_impulse_moment

def change_vel(vel):
    k = year_s / a_e
    changed_vel = []
    for i in range(len(vel)):
        changed_vel.append(vel[i] * k)
    return changed_vel
def change_mass(mass):
    changed_mass = mass / M_sun
    return changed_mass
def type_float(coordinates):
    float_coordinates = []
    for i in range(len(coordinates)):
        float_coordinates.append(float(coordinates[i]))
    return float_coordinates

def fill_list_3d_func(my_list, function, bodies):
    for i in range(3):
        my_list[i].append(function(bodies)[i])
    return my_list
def fill_coord_list_3d(my_list, body):
    for i in range(3):
        my_list[i].append(body.coord[i])
    return my_list
def fill_vel_list_3d(my_list, body):
    for i in range(3):
        my_list[i].append(body.vel[i])
    return my_list

def get_time_step(bodies, time_step, problem):
    dec_step = False
    inc_step = False
    expected = copy(bodies)
    for i in range(len(bodies)):
        if i > 0:
            expected[i].coord = add(bodies[i].coord, mult(bodies[i].vel, time_step))
            expected[i].vel = add(bodies[i].vel, mult(bodies[i].acs, time_step))
            vel_change_i = abs(get_mag(expected[i].vel) - get_mag(bodies[i].vel))
            coord_change_i = abs(get_mag(expected[i].coord) - get_mag(bodies[i].coord))
            expected_delta_vel_i = 2 * vel_change_i / (get_mag(expected[i].vel) + get_mag(bodies[i].vel))
            expected_delta_coord_i = 2 * coord_change_i / (get_mag(expected[i].coord) + get_mag(bodies[i].coord))

            if expected_delta_vel_i > problem.delta_vel or expected_delta_coord_i > problem.delta_coord:
                dec_step = True
                break
            elif expected_delta_vel_i < problem.delta_vel and expected_delta_coord_i < problem.delta_coord:
                inc_step = True
    if dec_step and time_step - problem.delta_timestep > problem.timestep_min:
        return time_step - problem.delta_timestep
    elif inc_step and time_step + problem.delta_timestep < problem.timestep_max:
        return time_step + problem.delta_timestep
    else:
        return time_step
def clear_all_datafiles(bodies):
    for i in range(len(bodies)):
        f = open(Path(Path.cwd(), "data", "data out", "objects", f"{bodies[i].name}.txt"), "w")
        f.truncate(0)
        f.write("time, years\tX, a.u.\tY, a.u.\tZ, a.u.\tVx, a.u./year\tVy, a.u./year\tVz, a.u./year")
        f.close()
        f = open(Path(Path.cwd(), "data", "data out", "elements", f"elements of {bodies[i].name}.txt"), "w")
        f.truncate(0)
        f.write("time, years\ta, a.u.\te\ti, degrees\tlong_of_asc_node, degrees\targ_of_periapsis, degrees")
        f.close()


    f = open(Path(Path.cwd(), "data", "data out", "center_mass.txt"), "w")
    f.truncate(0)
    f.write("time, years\tcm_x, a.u.\tcm_y, a.u.\tcm_z, a.u.")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "momentum.txt"), "w")
    f.truncate(0)
    f.write("time, years\tmomentum_x, a.u.\tmomentum_y, a.u.\tmomentum_z, a.u.\tmomentum_mag")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "angular momentum.txt"), "w")
    f.truncate(0)
    f.write("time, years\tangular_momentum_x\tangular_momentum_y\tangular_momentum_z, \tangular_momentum_mag")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "energy.txt"), "w")
    f.truncate(0)
    f.write("time, years\tenergy")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "time_step.txt"), "w")
    f.truncate(0)
    f.write("time, years\ttime step, years")
    f.close()
def print_ex_time(execution_time):
    if 3600 > execution_time > 60:
        print(f"Время выполнения программы: {round(execution_time / 60, 1)} минут")
    elif execution_time < 60:
        print(f"Время выполнения программы: {round(execution_time, 1)} секунд")
    else:
        print(f"Время выполнения программы: {round(execution_time / 3600, 1)} часов")
def get_method(method_name, methods, methods_names):
    for i in range(len(methods_names)):
        if method_name == methods_names[i]:
            return methods[i]

def max_mass(bodies):
    masses = []
    for body in bodies:
        masses.append(body.mass)
    return max(masses), masses.index(max(masses))

# def check_collision(bodies, i):
#     collided_bodies = []
#     for i in range(len(bodies)):
#         for j in range(len(bodies)):
#             if get_mag(get_r(bodies[i].coord, bodies[j].coord)) <= bodies[i].radius + bodies[j].radius:
#                 m_max, i_max_mass = max_mass([bodies[i], bodies[j]])
#                 bodies[\\\].vel = mult(add(mult(bodies[i].vel, bodies[i].mass),
#                                                   mult(bodies[j].vel, bodies[j].mass)),
#                                               bodies[i].mass + bodies[j].mass)[:]

def major_axis(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    # FIXME: what I should do with Sun??
    if i == i_max_mass:
        return 0
    else:
        kappa_2 = G * (max_m + bodies[i].mass)
        r = get_mag(get_r(bodies[i].coord, bodies[i_max_mass].coord))
        v = get_mag(get_r(bodies[i].vel, bodies[i_max_mass].vel))
        a = (2 / r - v**2 / kappa_2) ** (-1)
        return a
def eccentricity(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i == i_max_mass:
        return 0
    else:
        a = major_axis(bodies, i)
        p = focal_parameter(bodies, i)
        e = np.sqrt(1 - p / a)
        return e
def focal_parameter(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i == i_max_mass:
        return 0
    else:
        kappa_2 = G * (max_m + bodies[i].mass)
        r = get_r(bodies[i].coord, bodies[i_max_mass].coord)
        v = get_r(bodies[i].vel, bodies[i_max_mass].vel)
        c = get_mag(vector_product(r, v))
        p = c ** 2 / kappa_2
        return p
def true_anomaly(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    kappa_2 = G * (max_m + bodies[i].mass)
    r = get_mag(get_r(bodies[i].coord, bodies[i_max_mass].coord))
    v = get_mag(get_r(bodies[i].vel, bodies[i_max_mass].vel))
    p = focal_parameter(bodies, i)

    theta = np.arctan2(v * r * np.sqrt(p / kappa_2), p - r) * 180 / np.pi
    return theta
def eccentric_anomaly(bodies, i):
    e = eccentricity(bodies, i)
    theta = true_anomaly(bodies, i)

    E = 2 * (np.arctan2(np.tan(theta / 2), np.sqrt((1 + e) / (1 - e)))) * 180 / np.pi
    return E
def middle_anomaly(bodies, i):
    E = eccentric_anomaly(bodies, i)
    e = eccentricity(bodies, i)

    M = (E - e * np.sin(E)) * 180 / np.pi
    return M
def inclination(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i == i_max_mass:
        return 0
    else:
        r = get_r(bodies[i].coord, bodies[i_max_mass].coord)
        v = get_r(bodies[i].vel, bodies[i_max_mass].vel)
        x = get_r(bodies[i].coord, bodies[i_max_mass].coord)[0]
        y = get_r(bodies[i].coord, bodies[i_max_mass].coord)[1]
        v_x = get_r(bodies[i].vel, bodies[i_max_mass].vel)[0]
        v_y = get_r(bodies[i].vel, bodies[i_max_mass].vel)[1]

        i = np.arccos((x * v_y - y * v_x) / (abs(get_mag(vector_product(r, v))))) * 180 / np.pi
        return i
def longitude_of_asc_node(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i == i_max_mass:
        return 0
    else:
        x = get_r(bodies[i].coord, bodies[i_max_mass].coord)[0]
        y = get_r(bodies[i].coord, bodies[i_max_mass].coord)[1]
        z = get_r(bodies[i].coord, bodies[i_max_mass].coord)[2]
        v_x = get_r(bodies[i].vel, bodies[i_max_mass].vel)[0]
        v_y = get_r(bodies[i].vel, bodies[i_max_mass].vel)[1]
        v_z = get_r(bodies[i].vel, bodies[i_max_mass].vel)[2]

        THETA = np.arctan2(z * v_y - y * v_z, z * v_x - x * v_z) * 180 / np.pi
        return THETA
def arg_of_periapsis(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i == i_max_mass:
        return 0
    else:
        inc = inclination(bodies, i)
        z = get_r(bodies[i].coord, bodies[i_max_mass].coord)[2]
        r = get_mag(get_r(bodies[i].coord, bodies[i_max_mass].coord))
        u = np.arcsin(z / (np.sin(inc) * r))

        g = (u - true_anomaly(bodies, i)) * 180 / np.pi

        g = np.arctan2(np.tan(g / 180 * np.pi), 1) * 180 / np.pi
        return g

def get_kepler_elements(bodies, i):
    max_m, i_max_mass = max_mass(bodies)
    if i != i_max_mass:
        r = get_mag(get_r(bodies[i].coord, bodies[i_max_mass].coord))
        r_vector = get_r(bodies[i].coord, bodies[i_max_mass].coord)
        v = get_mag(get_r(bodies[i].vel, bodies[i_max_mass].vel))
        v_vector = get_r(bodies[i].vel, bodies[i_max_mass].vel)
        j = get_mag(vector_product(r_vector, v_vector))

        kappa_2 = G * (max_m + bodies[i].mass)
        a = (2 / r - v ** 2 / kappa_2) ** (-1)

        x = get_r(bodies[i].coord, bodies[i_max_mass].coord)[0]
        y = get_r(bodies[i].coord, bodies[i_max_mass].coord)[1]
        z = get_r(bodies[i].coord, bodies[i_max_mass].coord)[2]
        v_x = get_r(bodies[i].vel, bodies[i_max_mass].vel)[0]
        v_y = get_r(bodies[i].vel, bodies[i_max_mass].vel)[1]
        v_z = get_r(bodies[i].vel, bodies[i_max_mass].vel)[2]

        long_of_asc_node = np.arctan2(z * v_y - y * v_z, z * v_x - x * v_z)

        inc = np.arccos((x * v_y - y * v_x) / (abs(get_mag(vector_product(r_vector, v_vector))))) * 180 / np.pi
        u = np.arcsin(z / (np.sin(inc) * r))
        arg_of_periap = u - true_anomaly(bodies, i)

        cos_2_betta = 2 * (scalar_product(v_vector, r_vector) / (v * r)) ** 2 - 1
        c = 0.5 * np.sqrt((abs(r ** 2 + (2 * a - r) ** 2 + 2 * r * (2 * a - r) * cos_2_betta)))

        p = j ** 2 / kappa_2
        e = np.sqrt(1 - p / a)

        theta = np.arctan2(v * r * np.sqrt(p / kappa_2), p - r)

        E = 2 * np.arctan2(np.tan(theta / 2), np.sqrt((1 + e) / (1 - e)))

        M = E - e * np.sin(E)

        elements = [a, e, i, long_of_asc_node, arg_of_periap]
        return elements
    else:
        # FIXME: what should I do with Sun??
        return [0 for i in range(5)]

def get_major_axis_for_all(bodies):
    all_minor_axis = []
    for i in range(len(bodies)):
        all_minor_axis.append(major_axis(bodies, i))
    return all_minor_axis
def get_eccentricity_for_all(bodies):
    all_eccentricity = []
    for i in range(len(bodies)):
        all_eccentricity.append(eccentricity(bodies, i))
    return all_eccentricity
def get_focal_parameter_for_all(bodies):
    focal_parameter_all = []
    for i in range(len(bodies)):
        focal_parameter_all.append(focal_parameter(bodies, i))
    return focal_parameter_all
def get_true_anomaly_for_all(bodies):
    true_anomaly_all = []
    for i in range(len(bodies)):
        true_anomaly_all.append(true_anomaly(bodies, i))
    return true_anomaly_all
def get_eccentric_anomaly_for_all(bodies):
    eccentric_anomaly_all = []
    for i in range(len(bodies)):
        eccentric_anomaly_all.append(eccentric_anomaly(bodies, i))
    return eccentric_anomaly_all
def get_middle_anomaly_for_all(bodies):
    middle_anomaly_all = []
    for i in range(len(bodies)):
        middle_anomaly_all.append(middle_anomaly(bodies, i))
    return middle_anomaly_all
