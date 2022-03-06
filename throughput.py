import math
import numpy as np
def cal_h(x, y, z, source, destination):
    kx = destination[0] - source[0]
    ky = destination[1] - source[1]
    kz = destination[2] - source[2]
    constant = (((kx * x) + (ky * y) + (kz * z)) / (math.pow(kx, 2) + math.pow(ky, 2) + math.pow(kz, 2)))
    h = math.sqrt(math.pow(constant * kx - x, 2) + math.pow(constant * ky - y, 2) + math.pow(constant * kz - z, 2))
    return h

def cal_used_energy_to_move(action):
    energy_move = math.sqrt((math.pow(action[0], 2) + math.pow(action[1], 2) + math.pow(action[2], 2)))
    return energy_move

source = np.array((0,0,0,4))
des = np.array((4,4,4,0))
action = np.array((-1,0,0,-1))

print(cal_h(2,3,3,source,des))

