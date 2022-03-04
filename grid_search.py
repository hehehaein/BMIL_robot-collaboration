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
                if reward <= -3 : color = color_list[0]
                elif reward > -3 : color = color_list[1]
                elif reward > -2 : color = color_list[2]
                elif reward > -1: color = color_list[3]
                elif reward > 0: color = color_list[4]
                elif reward > 1: color = color_list[5]
                elif reward > 2: color = color_list[6]
                elif reward > 3: color = color_list[7]
                elif reward > 4: color = color_list[8]
                elif reward > 5: color = color_list[9]
                ax.scatter(x,y,z,marker='o', s=100, c=color, alpha=0.8)
plt.show()

