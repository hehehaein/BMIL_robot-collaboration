import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;
import pickle
import matplotlib.lines as mlines

NUM_EPISODES = 200000
STEPS = 20
iter = 10

reward = []
throughput = []
with open("rewards_jcci_20.pickle","rb") as f:
    reward.append(pickle.load(f))
with open("throughputs_jcci_20.pickle","rb") as f:
    throughput.append(pickle.load(f))

reward = np.squeeze(reward, axis=0)
throughput = np.squeeze(throughput, axis=0)

def make_list(episode, term):
    i = 0
    n = 0
    list = []
    while i < episode:
        for t in range(term):
            list.append(n)
            i += 1
        n += 1
    return list

def get_mean(array, k):
    means = []
    m = k
    if k == STEPS:
        m = 1
    for n in range(0, NUM_EPISODES // m, 1):
        sum = 0
        for i in range(0, k, 1):
            sum += array[(n * k) + i]
        means.append(sum / k)
    return means

#리워드, hit ratio 같이 뽑기
plt.figure()
sns.set_style('darkgrid')
#throughput -> hit ratio 변환
# hit_ratio = []
# for i in range(len(throughput_count)):
#     tmp = throughput_count[i] / 20
#     hit_ratio.append(tmp)

for i in range(len(reward)):
    reward[i] = round(reward[i], 3)
for i in range(len(throughput)):
    throughput[i] = round(throughput[i], 3)

reward_means = get_mean(reward, STEPS)
throughput_means = get_mean(throughput, STEPS)
print(reward_means[-1])
print(throughput_means[-1])
d = {'100 episode': make_list(NUM_EPISODES,100),
     'throughput': throughput_means,
     'reward': reward_means}
df = pd.DataFrame(data=d)
fig, axe1 = plt.subplots()
axe2 = axe1.twinx()

throughput = sns.lineplot(ax=axe1, data=df, x='100 episode', y='throughput', color='red')
reward = sns.lineplot(ax=axe2, data=df, x='100 episode', y='reward', color='blue')

mark_reward = mlines.Line2D([], [], color='blue', linestyle ='-', label='reward')
mark_throughput = mlines.Line2D([], [], color='red', linestyle ='-', label='throughput average')
mark_upper = mlines.Line2D([],[], color='black', linestyle='--', label='upper bound')

axe1.grid(axis='y')
axe2.grid(axis='x')

axe1.axhline(6.667, c='black', ls ='--')
#axe2.axhline(11.667, c='black', ls ='--')

axe1.tick_params(axis='y')
axe2.tick_params(axis='y')

axe2.set_ylim(-2, 12)
axe1.set_ylim(0, 7)

plt.legend(handles=[mark_reward, mark_throughput,mark_upper], loc='lower right')

plt.show()