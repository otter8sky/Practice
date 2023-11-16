from const import *

def vect(v, u):
    p = [v[1]*u[2] - v[2]*u[1], -(v[0]*u[2] - v[2]*u[0]), v[0]*u[1] - v[1]*u[0]]
    return p
def skal(v, u):
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

class Body:
    def __init__(self, vel, coord, mass, acs):
        self.vel = vel
        self.coord = coord
        self.mass = mass
        self.acs = acs

def copy(bodies):
    copied_bodies = []
    for i in range(len(bodies)):
        copied_bodies.append(Body(bodies[i].vel, bodies[i].coord, bodies[i].mass, bodies[i].acs))
    return copied_bodies
