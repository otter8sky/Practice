from operations import *

class Body:
    def __init__(self, vel, coord, mass, acs):
        self.vel = vel
        self.coord = coord
        self.mass = mass
        self.acs = acs
def copy(bodies):
    copied_bodies = Body(bodies.vel, bodies.coord, bodies.mass, bodies.acs)
    return copied_bodies

def aaa(body):
    result = copy(body)
    result.vel = [0, 0, 0]
    result.coord = [2, 2, 2]
    global a
    a = copy(result)
    return result


a = Body([1, 1, 1], [1, 1, 1], 1, [1, 1, 1])
aaa(a)
print("a_vel = ", a.vel)
print("a_coord = ", a.coord)
