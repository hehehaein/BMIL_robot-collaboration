import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns;
sns.set()
plt.ion()

seed = 1
np.random.seed(seed)
random.seed(seed)
R_MAX = 3

MIN_LOC = 0
MAX_LOC = 4

MIN_HEIGHT = 1
MAX_HEIGHT = 4

relay_node = []
array = np.zeros((4, 5, 5), dtype=int)
for i in range(100000):
    relay_node = []

    x = random.randint(MIN_LOC, MAX_LOC)
    y = random.randint(MIN_LOC, MAX_LOC)
    z = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    r = random.randint(0, R_MAX)
    relay_node.append(x)
    relay_node.append(y)
    relay_node.append(z)
    relay_node.append(r)

    array[relay_node[2] - 1][relay_node[0]][relay_node[1]] += 1

frame1 = pd.DataFrame(data=array[0])
frame2 = pd.DataFrame(data=array[1])
frame3 = pd.DataFrame(data=array[2])
frame4 = pd.DataFrame(data=array[3])

plt.figure()
ax = sns.heatmap(frame1, cmap='coolwarm', annot=True)
plt.title('z=1')

plt.figure()
ax = sns.heatmap(frame2, cmap='coolwarm', annot=True)
plt.title('z=2')
plt.figure()
ax = sns.heatmap(frame3, cmap='coolwarm', annot=True)
plt.title('z=3')
plt.figure()
ax = sns.heatmap(frame4, cmap='coolwarm', annot=True)
plt.title('z=4')

plt.ioff()
plt.show()
