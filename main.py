from comp import comp
from comp import read_data_file
from methods import By_Ex_Euler
from methods import By_PC
from methods import By_RK_4
from methods import By_Verlet
from const import *

Methods = [By_Ex_Euler,
           By_PC,
           By_RK_4,
           By_Verlet]
Names = ['Explicit Euler', 'Predictor-Corrector', 'RK-4', 'Verlet']

# choose the method, number of bodies SS, end time and time step
method = By_RK_4
n = 6
time_end = 1
time_step = 0.0001
# variable time step
delta_vel = 0.01
delta_coord = 0.01
delta_timestep = 0.01
timestep_max = 0.01
timestep_min = 0.0000001

Bodies, names, colors = read_data_file('DATA_norm', n)

energy = comp(method, Bodies, t, time_end, time_step, colors, names, delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min)
