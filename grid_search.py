from gymExample.gym_example.envs.my_dqn_env import reward_set
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N=2
# ~transmission radius max
R_MAX = 4
# location x,y,z
MIN_LOC = 0
MAX_LOC = 4

MIN_HEIGHT = 1
MAX_HEIGHT = 4

source = np.array((MIN_LOC, MIN_LOC, MIN_LOC, 3))
dest = np.array((MAX_LOC, MAX_LOC, MAX_LOC, 0))

state = np.zeros((N+2, 4))
state[1]=np.array((3, 3, 3, 3))
state[2]=np.array((MIN_LOC, MIN_LOC, MIN_LOC, 3))
state[3]=np.array((MAX_LOC, MAX_LOC, MAX_LOC, 0))

'''agent = np.zeros((6,4))
agent[0] = (1,1,2,3)
agent[1] = (1,2,1,3)
agent[2] = (2,1,1,3)
agent[3] = (2,2,1,3)
agent[4] = (2,1,2,3)
agent[5] = (1,2,2,3)'''

env = reward_set(N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(0, MAX_LOC)
ax.set_ylim3d(MAX_LOC, 0)
ax.set_zlim3d(0, MAX_LOC)

xs = []
ys = []
zs = []
alphas = []
color_list = ("bisque", "navajowhite", "orange", "darkorange", "darkgoldenrod", "olive", "olivedrab","darkgreen","darklategray","blue")
for x in range(0,5,1):
    for y in range(0,5,1):
        for z in range(0,5,1):
            for r in range(0,5,1):
                xs.append(x)
                ys.append(y)
                zs.append(z)
                agent = np.array((x,y,z,r))
                state[0] = agent
                adj_array = env.cal_adjacency(state)
                throughput = env.cal_throughput(adj_array)
                foot = env.cal_foot(state, source, dest, 0)
                dispersed = env.cal_dispersed(0, state[0][3], adj_array)
                reward = throughput+foot+dispersed
                print(state[0], 'throughput=',throughput,'foot=',foot,'dispersed=',dispersed, 'reward=', reward)
                reward_save = (reward+5)/11
                alphas.append(reward_save)
ax.scatter(xs,ys,zs,marker='o', s=100, c='orange', alpha=alphas)
plt.show()

