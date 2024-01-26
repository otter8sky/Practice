from operations import *
import numpy as np


def By_Ex_Euler(bodies, time_step):
    result = copy(bodies)
    for i in range(len(bodies)):
        result[i].coord = add(result[i].coord, mult(result[i].vel, time_step))
        result[i].vel = add(result[i].vel, mult(result[i].acs, time_step))
    result = get_acs_for_all(result)
    for i in range(len(bodies)):
        bodies[i].vel = result[i].vel[:]
        bodies[i].coord = result[i].coord[:]
        bodies[i].acs = result[i].acs[:]
    return result


def By_PC(bodies, time_step):
    result = copy(bodies)
    k1 = copy(bodies)
    k2 = Euler_byDataOf(k1, k1, time_step)
    for i in range(len(bodies)):
        result[i].coord = add(k1[i].coord, mult(add(k1[i].vel, k2[i].vel), time_step / 2))
        result[i].vel = add(k1[i].vel, mult(add(k1[i].acs, k2[i].acs), time_step / 2))
    result = get_acs_for_all(result)
    for i in range(len(bodies)):
        bodies[i].coord = result[i].coord[:]
        bodies[i].vel = result[i].vel[:]
        bodies[i].acs = result[i].acs[:]
    return result


def Euler_byDataOf(data, bodies, time_step):
    result1 = copy(bodies)
    for i in range(len(bodies)):
        result1[i].coord = add(result1[i].coord, mult(data[i].vel, time_step))
        result1[i].vel = add(result1[i].vel, mult(data[i].acs, time_step))
    result1 = get_acs_for_all(result1)
    return result1


def By_RK_4(bodies, time_step):
    result = copy(bodies)
    k1 = copy(bodies)
    k2 = Euler_byDataOf(k1, k1, time_step / 2)
    k3 = Euler_byDataOf(k2, k1, time_step / 2)
    k4 = Euler_byDataOf(k3, k1, time_step)

    for i in range(len(bodies)):
        addendum_1 = add(k1[i].acs, mult(k2[i].acs, 2))
        addendum_2 = add(addendum_1, mult(k3[i].acs, 2))
        factor_vel = add(addendum_2, k4[i].acs)

        result[i].vel = add(result[i].vel, mult(factor_vel, time_step / 6))

        addendum_1 = add(k1[i].vel, mult(k2[i].vel, 2))
        addendum_2 = add(addendum_1, mult(k3[i].vel, 2))
        factor_coord = add(addendum_2, k4[i].vel)

        result[i].coord = add(result[i].coord, mult(factor_coord, time_step / 6))
    result = get_acs_for_all(result)
    for i in range(len(bodies)):
        bodies[i].coord = result[i].coord[:]
        bodies[i].vel = result[i].vel[:]
        bodies[i].acs = result[i].acs[:]
    return result


def By_Verlet(bodies, time_step):
    result = copy(bodies)
    for i in range(len(bodies)):
        result[i].coord = add(add(bodies[i].coord, mult(bodies[i].vel, time_step)),
                              mult(bodies[i].acs, time_step ** 2 / 2))
        predicted_acs_i = get_total_acs(result, i)
        result[i].vel = add(bodies[i].vel, mult(add(bodies[i].acs, predicted_acs_i), time_step / 2))
    result = get_acs_for_all(result)
    for i in range(len(bodies)):
        bodies[i].coord = result[i].coord[:]
        bodies[i].vel = result[i].vel[:]
        bodies[i].acs = result[i].acs[:]
    return result


def By_Leap_Frog(bodies, time_step):
    result = copy(bodies)
    for i in range(len(bodies)):
        result[i].coord = add(bodies[i].coord, mult(bodies[i].half_vel, time_step))
        result = get_acs_for_all(result)
        result[i].vel = add(bodies[i].half_vel, mult(result[i].acs, time_step / 2))
        result[i].half_vel = add(bodies[i].half_vel, mult(result[i].acs, time_step))

    for i in range(len(bodies)):
        bodies[i].coord = result[i].coord[:]
        bodies[i].half_vel = result[i].half_vel[:]
        bodies[i].vel = result[i].vel[:]
        bodies[i].acs = result[i].acs[:]
    return result


def By_RK_N(bodies, time_step):
    k_list = []
    for i in range(len(b_list)):
        k_list.append(get_k(copy(bodies), a_list[i], k_list, time_step))
    result = get_k(copy(bodies), b_list, k_list, time_step)
    for i in range(len(bodies)):
        bodies[i].coord = result[i].coord[:]
        bodies[i].vel = result[i].vel[:]
        bodies[i].acs = result[i].acs[:]
    return result


methods = [By_Ex_Euler, By_PC, By_RK_4, By_Verlet, By_Leap_Frog, By_RK_N]


def fill_methods_names_list(methods_list):
    methods_names_list = []
    for i in range(len(methods_list)):
        methods_names_list.append(methods_list[i].__name__)
    return methods_names_list


methods_names = fill_methods_names_list(methods)
