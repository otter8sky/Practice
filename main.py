import time
import pandas as pd
from comp import comp
from comp import read_data_file
from comp import read_comp_file
from comp import plot_bodies
from comp import plot_energy
from comp import plot_cm
from comp import plot_momentum
from comp import plot_time_step
from comp import plot_all
from comp import plot_angular_momentum
from comp import plot_elements
from methods import By_Ex_Euler
from methods import By_PC
from methods import By_RK_4
from methods import By_Verlet
from methods import By_Leap_Frog
from methods import By_RK_N
from methods import methods
from methods import methods_names
from operations import *
import winsound


Methods = [By_Ex_Euler,
           By_PC,
           By_RK_4,
           By_Verlet,
           By_Leap_Frog,
           By_RK_N]

start_time = time.perf_counter()

bodies = read_data_file()
# problem = read_comp_file('COMPUTING.txt')

method = 'By_RK_4'
time_end = 1
initial_timestep = 0.0001
delta_vel = 0.001
delta_coord = 0.0001
delta_timestep = 0.0000001
timestep_max = 0.01
timestep_min = 0.0000001
dt_output = 0.01
problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep,
                  delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min, dt_output)

clear_all_datafiles(bodies)
comp(problem, bodies)

# TODO: сделать отдельный файл, где можно просто графики строить

end_time = time.perf_counter()
execution_time = end_time
print_ex_time(execution_time)
#
plot_bodies(bodies, problem)
plot_elements(bodies, problem)
# plot_energy(problem)
# plot_momentum(problem)
# plot_cm(problem)
# plot_time_step(problem)
# plot_angular_momentum(problem)
#
# freq = 500
# dur = 2000
# winsound.Beep(freq, dur)

# plot_all(problem, bodies)

