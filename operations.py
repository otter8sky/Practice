from const import *
from pathlib import Path

class Body:
    def __init__(self, vel, coord, mass, acs, name, color):
        self.vel = vel
        self.coord = coord
        self.mass = mass
        self.acs = acs
        self.name = name
        self.color = color
class Problem:
    def __init__(self, method, time_end, initial_timestep, delta_vel,
                 delta_coord, delta_timestep, timestep_max, timestep_min):
        self.method = method
        self.time_end = time_end
        self.initial_timestep = initial_timestep
        self.delta_vel = delta_vel
        self.delta_timestep = delta_timestep
        self.delta_coord = delta_coord
        self.timestep_max = timestep_max
        self.timestep_min = timestep_min
class Method:
    def __init__(self, name):
        self.name = name

    def add_to_list(self, methods):
        methods.append(self.name)

def vect(v, u):
    p = [v[1]*u[2] - v[2]*u[1], -(v[0]*u[2] - v[2]*u[0]), v[0]*u[1] - v[1]*u[0]]
    return p
def scal(v, u):
    return v[0]*u[0] + v[1]*u[1] + v[2]*u[2]
def mult(v, c):
    u = [v[0]*c, v[1]*c, v[2]*c]
    return u
def add(a, b):
    c = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    return c
def get_r(a, b):
    r = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return r
def get_mag(r):
    mag = (r[0]**2 + r[1]**2 + r[2]**2)**0.5
    return mag

def get_acs(m, r):
    a = mult(r, -G * m / (get_mag(r))**3)
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
    a = mult(r, -G * m / ((get_mag(r) - r_g)**2 * get_mag(r)))
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
        kinetical_energy += bodies[i].mass * get_mag(bodies[i].vel)**2 / 2
    return kinetical_energy
def get_Energy(bodies):
    energy = get_Kinetical_Energy(bodies) + get_Potential_Energy(bodies)
    return energy

def copy(bodies):
    copied_bodies = []
    for i in range(len(bodies)):
        copied_bodies.append(Body(bodies[i].vel,
                                  bodies[i].coord,
                                  bodies[i].mass,
                                  bodies[i].acs,
                                  bodies[i].name,
                                  bodies[i].color))
    return copied_bodies

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
def get_vect_total_momentum(bodies):
    vect_total_momentum = [0, 0, 0]
    for i in range(len(bodies)):
        vect_total_momentum = add(vect_total_momentum, mult(bodies[i].vel, bodies[i].mass))[:]
    return vect_total_momentum

def clear_all_datafiles(bodies):
    for i in range(len(bodies)):
        f = open(Path(Path.cwd(), "data", "objects", f"{bodies[i].name}.txt"), "w")
        f.truncate(0)
        f.write("time, years\tX, a.u.\tY, a.u.\tZ, a.u.\tVx, a.u./year\tVy, a.u./year\tVz, a.u./year")
        f.close()

    f = open(Path(Path.cwd(), "data", "data out", "center_mass.txt"), "w")
    f.truncate(0)
    f.write("time, years\tcm_x, a.u.\tcm_y, a.u.\tcm_z, a.u.")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "momentum.txt"), "w")
    f.truncate(0)
    f.write("time, years\tmomentum_x, a.u.\tmomentum_y, a.u.\tmomentum_z, a.u.\tmomentum_mag")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "energy.txt"), "w")
    f.truncate(0)
    f.write("time, years\tenergy")
    f.close()

    f = open(Path(Path.cwd(), "data", "data out", "time_step.txt"), "w")
    f.truncate(0)
    f.write("time, years\ttime step, years")
    f.close()

def get_method(method_name, methods, methods_names):
    for i in range(len(methods_names)):
        if method_name == methods_names[i]:
            return methods[i]

def print_ex_time(execution_time):
    if 3600 > execution_time > 60:
        print(f"Время выполнения программы: {round(execution_time / 60, 1)} минут")
    elif execution_time < 60:
        print(f"Время выполнения программы: {round(execution_time, 1)} секунд")
    else:
        print(f"Время выполнения программы: {round(execution_time / 3600, 1)} часов")
