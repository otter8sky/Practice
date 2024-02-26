import time
import pandas as pd
from comp import comp
from comp import read_data_file
from comp import get_data_stability_of_method
from comp import descartes_from_kepler
from comp import get_ideal_data_stability_of_method
from plotting import plot_stability_of_method
from comp import read_comp_file
from plotting import plot_bodies
from plotting import plot_energy
from plotting import plot_cm
from plotting import plot_momentum
from plotting import plot_time_step
from plotting import plot_all
from plotting import plot_angular_momentum
from plotting import plot_elements
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
r = 5
v = 6.8
i = 122.7 / 180 * np.pi
omega = 241.7 / 180 * np.pi
big_omega = 24.6 / 180 * np.pi
q = 0.23
mass = 0.0001
color = "purple"
bodies = descartes_from_kepler("Asteroid", bodies, r, v, i, omega, big_omega, q, mass, color)
# TODO: в чём ошибка с РК-7???!!! (ошибка не во мне!)
# TODO:
method = 'By_RK_4'
time_end = 1
list_of_time_step = [1e-3, 1e-4, 1e-5]
initial_timestep = 1e-3
delta_vel = 0.001
delta_coord = 0.0001
delta_timestep = 0.0000001
timestep_max = 0.01
timestep_min = 0.0000001
dt_output = 0.001
problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep,
                  delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min, dt_output)
print("Началось уничтожение Солнечной системы!!")
comp(problem, bodies)
plot_bodies(bodies, problem)

# get_ideal_data_stability_of_method(bodies, problem)

# method = 'By_RK_4'
#
# problem = Problem(get_method(method, methods, methods_names), time_end, initial_timestep,
#                   delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min, dt_output)
# bodies = read_data_file()
# for ts in list_of_time_step:
#     bodies = read_data_file()
#     comp(Problem(get_method(method, methods, methods_names), time_end, ts, delta_vel, delta_coord, delta_timestep,
#                  timestep_max, timestep_min, dt_output), bodies)

# get_data_stability_of_method(bodies, problem, list_of_time_step)
# plot_stability_of_method(bodies, problem)

# comp(Problem(get_method(method, methods, methods_names), time_end, initial_timestep,
#              delta_vel, delta_coord, delta_timestep, timestep_max, timestep_min, dt_output), bodies)



end_time = time.perf_counter()
execution_time = end_time
print_ex_time(execution_time)


# plot_elements(bodies, problem)
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
