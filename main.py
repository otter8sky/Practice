from comp import comp
from comp import read_data_file
from comp import read_comp_file
from comp import plot_bodies
from comp import plot_energy
from comp import plot_cm
from comp import plot_momentum
from comp import plot_time_step
from comp import plot_all
from methods import By_Ex_Euler
from methods import By_PC
from methods import By_RK_4
from methods import By_Verlet
from methods import By_Leap_Frog
from methods import methods
from methods import methods_names
from operations import *

Methods = [By_Ex_Euler,
           By_PC,
           By_RK_4,
           By_Verlet,
           By_Leap_Frog]

bodies = read_data_file('DATA.txt')
# problem = read_comp_file('COMPUTING.txt')

method = 'By_Leap_Frog'
time_end = 8
initial_timestep = 0.001
delta_vel = 0.001
delta_coord = 0.001
delta_timestep = 0.0000001
timestep_max = 0.01
timestep_min = 0.0000001
problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep,
                  delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min)

# clear_all_datafiles(bodies)
# comp(problem, bodies)

# plot_bodies(bodies, problem)
# plot_energy(problem)
# plot_momentum(problem)
# plot_cm(problem)
# plot_time_step(problem)
plot_all(problem, bodies)
