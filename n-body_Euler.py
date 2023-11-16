from operations import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
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
    a = mult(r, -G * m / get_mag(r)**3)
    return a
class Body:
    def __init__(self, vel, coord, coord_prev, mass, acs):
        self.vel = vel
        self.coord = coord
        self.coord_prev = coord_prev
        self.mass = mass
        self.acs = acs

time_end = 5
h = 0.0001

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
        acs_total_i = [0.0, 0.0, 0.0]
        for j in range(n):
            if i != j:
                acs_total_i = add(acs_total_i, get_acs(Bodies[j].mass, get_r(Bodies[i].coord, Bodies[j].coord_prev)))
        Bodies[i].acs = acs_total_i
        Bodies[i].coord = add(Bodies[i].coord, mult(Bodies[i].vel, h))
        Bodies[i].vel = add(Bodies[i].vel, mult(Bodies[i].acs, h))

        Bodies_x[i].append(Bodies[i].coord[0])
        Bodies_y[i].append(Bodies[i].coord[1])
        Bodies_z[i].append(Bodies[i].coord[2])
    for i in range(n):
        Bodies[i].coord_prev = Bodies[i].coord
    t += h

for i in range(n):
    f = open(f"{i+1}.txt", "w+")
    f.writelines([str(Bodies_x[i]), "\n", str(Bodies_y[i]), "\n", str(Bodies_z[i]), "\n", colors[i], "\n", names[i], "\n"])
    f.close()

