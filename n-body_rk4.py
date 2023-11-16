from operations import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
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
    a = mult(r, -G * m / get_mag(r)**3)
    return a
def get_total_acs(i, n, coord_i):
    acs_total_i = [0.0, 0.0, 0.0]
    for j in range(n):
        if i != j:
            acs_total_i = add(acs_total_i, get_acs(Bodies[j].mass, get_r(coord_i, Bodies[j].coord_prev)))
    return acs_total_i
class Body:
    def __init__(self, vel, coord, coord_prev, mass, acs):
        self.vel = vel
        self.coord = coord
        self.coord_prev = coord_prev
        self.mass = mass
        self.acs = acs

pi = 3.1415926535897
time_end = 5
h = 0.0001
G = 4*pi**2
t = 0
acs = [0, 0, 0]

file = open('DATA.txt', 'r')
s = 0
names = []
Bodies = []
colors = []
mas = []

for item in file:
    item = item.strip()
    if s < 5:
        if s == 1 or s == 2:
            item = list(map(float, item.split()))
        elif s == 3:
            item = float(item)
        mas.append(item)
        s += 1
    else:
        names.append(mas[0])
        colors.append(mas[4])
        Bodies.append(Body(mas[1], mas[2], mas[2], mas[3], acs))
        mas = []
        s = 0
file.close()

n = len(Bodies)
Bodies_x = [[] for i in range(n)]
Bodies_y = [[] for i in range(n)]
Bodies_z = [[] for i in range(n)]

while t <= time_end:
    for i in range(n):
        k1 = get_total_acs(i, n, Bodies[i].coord_prev)
        Vel_prediction_1 = add(Bodies[i].vel, mult(k1, h / 2))
        Coord_prediction_1 = add(Bodies[i].coord, mult(Vel_prediction_1, h))

        k2 = get_total_acs(i, n, Coord_prediction_1)
        Vel_prediction_2 = add(Bodies[i].vel, mult(k2, h / 2))
        Coord_prediction_2 = add(Bodies[i].coord, mult(Vel_prediction_2, h))

        k3 = get_total_acs(i, n, Coord_prediction_2)
        Vel_prediction_3 = add(Bodies[i].vel, mult(k3, h))
        Coord_prediction_3 = add(Bodies[i].coord, mult(Vel_prediction_3, h))

        k4 = get_total_acs(i, n, Coord_prediction_3)
        Vel_prediction_4 = add(Bodies[i].vel, mult(k4, h))

        addendum_1 = add(k1, mult(k2, 2))
        addendum_2 = add(addendum_1, mult(k3, 2))
        factor_vel = add(addendum_2, k4)
        Bodies[i].vel = add(Bodies[i].vel, mult(factor_vel, h / 6))

        addendum_1 = add(Vel_prediction_1, mult(Vel_prediction_2, 2))
        addendum_2 = add(addendum_1, mult(Vel_prediction_3, 2))
        factor_coord = add(addendum_2, Vel_prediction_4)
        Bodies[i].coord = add(Bodies[i].coord, mult(factor_coord, h / 6))

        Bodies_x[i].append(Bodies[i].coord[0])
        Bodies_y[i].append(Bodies[i].coord[1])
        Bodies_z[i].append(Bodies[i].coord[2])
    t += h

for i in range(n):
    f = open(f"{i+1}.txt", "w+")
    f.writelines([str(Bodies_x[i]), "\n", str(Bodies_y[i]), "\n", str(Bodies_z[i]), "\n", colors[i], "\n", names[i], "\n"])
    f.close()

