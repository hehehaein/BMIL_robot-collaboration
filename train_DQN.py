# -*- coding: utf-8 -*-
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import seaborn as sns; sns.set()
import json

from collections import namedtuple, deque, OrderedDict
from ray.tune.registry import register_env

from gymExample.gym_example.envs.my_dqn_env import My_DQN
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

select_env = "dqn-v0"
register_env(select_env, lambda config: My_DQN())
env = gym.make(select_env).unwrapped

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(seed)

# GPU를 사용할 경우
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 81)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        # x = x.to(device)
        # print('forward\n',x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


BATCH_SIZE = 64
NUM_EPISODES = 1000
STEPS = 20
DISCOUNT_FACTOR = 0.8
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = (EPS_START - EPS_END) / (NUM_EPISODES * STEPS * 0.5)
TARGET_UPDATE = 40
UPDATE_FREQ = 20
BUFFER = 100000
LEARNING_RATE = 1e-4
IS_DOUBLE_Q = False
ZERO = False
SCHEDULER = False
SCHEDULER_GAMMA = 0.95

now = time.localtime()
str = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}-{14}-{15}-{16}'.format(
    NUM_EPISODES, STEPS, DISCOUNT_FACTOR, EPS_DECAY, TARGET_UPDATE, UPDATE_FREQ, '1th',
    BUFFER, LEARNING_RATE, seed, IS_DOUBLE_Q, ZERO, SCHEDULER, SCHEDULER_GAMMA, now.tm_hour, now.tm_min, now.tm_sec)
path = os.path.join(os.getcwd(), 'results')

n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
optimal_net = DQN().to(device)
tmp_net = DQN().to(device)

target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
if SCHEDULER:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
memory = ReplayMemory(BUFFER)

steps_done = 0
epslions = []
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
trainer = Trainer(callbacks=[earlystop])


def select_action(state):
    global steps_done

    # 멈춰있는 action 확률 키우기
    a1 = []  # 빈 리스트 생성
    a2 = []
    a3 = []
    for i in range(n_actions):
        a1.append(1)
        a2.append(3 / 356)
        a3.append(0.0044)
    a2[40] = 1 / 16
    a3[40] = 81 / 125

    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - (EPS_DECAY * steps_done))
    epslions.append(eps_threshold)
    steps_done += 1

    # eps greedy method
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(-1)[1].view(1, 1)
    else:
        if ZERO:
            if (20 * STEPS * NUM_EPISODES) * 0.5 * 0.66 < steps_done:
                return torch.tensor([random.choices(range(0, n_actions), weights=a3)], device=device, dtype=torch.long)
            elif (20 * STEPS * NUM_EPISODES) * 0.5 * 0.33 < steps_done:
                return torch.tensor([random.choices(range(0, n_actions), weights=a2)], device=device, dtype=torch.long)
            else:
                return torch.tensor([random.choices(range(0, n_actions), weights=a1)], device=device, dtype=torch.long)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# 최적화의 한 단계를 수행함
losses = []
scheduler_lrs = []
scheduler_check = False


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # batch-array의 Transitions을 Transition의 batch-arrays로 전환
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    for s in batch.next_state:
        if s is not None:
            non_final_next_states = torch.stack(tuple(torch.Tensor(s)))

    '''propagation의 예측값 저장'''
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택
    # policy_net에 따라 각 배치 상태에 대해 선택된 행동
    state_action_values = policy_net(state_batch).gather(-1, action_batch.squeeze(1))

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    index = torch.zeros(BATCH_SIZE, device=device)
    if IS_DOUBLE_Q:
        index = policy_net(non_final_next_states).max(-1)[1].detach()
        next_state_values = target_net(non_final_next_states)[index]
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Huber 손실 계산
    '''loss 계산'''
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())

    # 모델 최적화
    optimizer.zero_grad()
    '''backward propagation'''
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    '''경사하강법(gradient descent). 옵티마이저는 .grad 에 저장된 변화도(gradients)에 따라 각 매개변수를 조정(adjust)합니다.'''
    optimizer.step()
    if scheduler_check:
        scheduler.step()
        scheduler_lrs.append(scheduler.get_lr())
    else:
        scheduler_lrs.append(LEARNING_RATE)

actions = []
throughputs = []
foots=[]
disperses = []
moves = []
txrs = []
throughput_counts = []
rewards = []
stay_count = []
'''scatters_front = []
scatters_middle = []
scatters_tail = []'''

reward_count = 0
move_count = 0
maxes = []
optimal = 0
stay = 0
opti_count = []
stay_count = []
move_count = 0
for i_episode in tqdm(range(NUM_EPISODES)):
    # 환경과 상태 초기화
    state = env.reset()
    state = torch.Tensor(state)
    throughput_count = 0
    max_count = 0
    # optimal = 0
    stay = 0
    move_count = 0
    max_count = 0
    if SCHEDULER and i_episode == NUM_EPISODES // 2:
        scheduler_check = True
    for t in range(0, STEPS, 1):
        move_count += 1
        throughput_value = 0

        # 행동 선택과 수행
        action = select_action(state)
        actions.append(action.item())
        next_state, reward, done, last_set = env.step(action.item())

        state_reshape = np.reshape(state, (env.N + 2, 4))
        next_state_reshape = np.reshape(next_state, (env.N + 2, 4))

        if next_state_reshape[0][0] == 1 and \
                next_state_reshape[0][1] == 1 and \
                next_state_reshape[0][2] == 1 and \
                next_state_reshape[0][3] == 3:
            if np.array_equal(next_state_reshape, state_reshape):
                stay += 1

        if reward > 6.8:
            max_count += 1
            print(state_reshape[0], next_state_reshape[0],
                  "action:%2d%2d%2d%2d reward:%.6f count:%2d"
                  % (last_set[5], last_set[6], last_set[7], last_set[8], reward, move_count))

        if i_episode >= (NUM_EPISODES - 2):
            print(state_reshape[0], next_state_reshape[0],
                  "action:%2d%2d%2d%2d throughput:%6.3f foot:%6.3f dispersed:%6.3f move:%6.3f txr:%3d"
                  % (last_set[5], last_set[6], last_set[7], last_set[8],
                     last_set[0], last_set[1], last_set[2], last_set[3], last_set[4]))

        if last_set[0] != 0:
            throughput_value = last_set[0]
            throughput_count += 1


        throughput_counts.append(throughput_count)
        throughputs.append(throughput_value)
        foots.append(last_set[1])
        disperses.append(last_set[2])
        moves.append(last_set[3])
        txrs.append(last_set[4])
        rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        # 3D plot
        '''if i_episode == (num_episodes - 1):
            scatters_tail.append(np.array(next_state_reshape[0]))
        elif i_episode == num_episodes // 2:
            scatters_middle.append(np.array(next_state_reshape[0]))
        elif i_episode == 0:
            scatters_front.append(np.array(next_state_reshape[0]))'''

        next_state = torch.Tensor(next_state)
        if done:
            next_state = None
        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)
        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        if i_episode % UPDATE_FREQ == 0:
            optimize_model()
            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

    f1 = open(path + '/' + str, 'w')
    f2 = open(path + '/' + '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}-{14}-{15}-{16}'.format(
        NUM_EPISODES, STEPS, DISCOUNT_FACTOR, EPS_DECAY, TARGET_UPDATE, UPDATE_FREQ, '1th',
        BUFFER, LEARNING_RATE, seed, IS_DOUBLE_Q, ZERO, SCHEDULER, SCHEDULER_GAMMA, now.tm_hour, now.tm_min, now.tm_sec), 'w')
    file_data = OrderedDict()

    if (i_episode % 100 == 0):
        tmp_net.load_state_dict(policy_net.state_dict())
        torch.save({'epi{}'.format(i_episode): tmp_net.state_dict()}, path + '/' + '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}-{14}-{15}-{16}_{17}'.format(
                                                                        NUM_EPISODES, STEPS, DISCOUNT_FACTOR, EPS_DECAY, TARGET_UPDATE, UPDATE_FREQ, '1th',
                                                                        BUFFER, LEARNING_RATE, seed, IS_DOUBLE_Q, ZERO, SCHEDULER, SCHEDULER_GAMMA, now.tm_hour, now.tm_min, now.tm_sec,i_episode))
        file_data["BATCH_SIZE"]=BATCH_SIZE
        file_data["NUM_EPISODES"]=NUM_EPISODES
        file_data["STEPS"]=STEPS
        file_data["DISCOUNT_FACTOR"]=DISCOUNT_FACTOR
        file_data["EPS_START"]=EPS_START
        file_data["EPS_END"]=EPS_END
        file_data["EPS_DECAY"]=EPS_DECAY
        file_data["TARGET_UPDATE"]=TARGET_UPDATE
        file_data["UPDATE_FREQ"]=UPDATE_FREQ
        file_data["BUFFER"]=BUFFER
        file_data["LEARNING_RATE"]=LEARNING_RATE
        file_data["SEED"]=seed
        file_data["IS_DOUBLE_Q"]=IS_DOUBLE_Q
        file_data["ZERO"] = ZERO
        file_data["SCHEDULER"] = SCHEDULER
        file_data["SCHEDULER_GAMMA"] = SCHEDULER_GAMMA
        file_data["ACTION"] = actions
        file_data["THROUGHPUT"] = throughput_value
        file_data["FOOT"] = foots
        file_data["DISPERSED"] = disperses
        file_data["MOVE"] = moves
        file_data["TXR"] = txrs
        file_data["REWARD"] = rewards
        json_data = json.dumps(file_data, ensure_ascii=False)
        with open(path+'/'+str+'epi{}'.format(i_episode)+'_data.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

    # gradient 출력
    '''if i_episode == num_episodes-1:
        for p in policy_net.parameters():
            with torch.no_grad():
                print(p.grad, len(p.grad), p.grad.shape)'''

    # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if stay == 15:
        optimal_net.load_state_dict(policy_net.state_dict())
    stay_count.append(stay)

torch.save({
    'target_net': target_net.state_dict(),
    'policy_net': policy_net.state_dict(),
    'optimal_net': optimal_net.state_dict(),
    'optimizer': optimizer.state_dict()
}, path + '/' + str)

print('Complete')


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

plt.figure()
throughput_count_means2 = get_mean(throughput_counts, STEPS)
d = {'episode': range(NUM_EPISODES),
     '500 episode': make_list(NUM_EPISODES, 500),
     'throughput count': throughput_count_means2}
df = pd.DataFrame(data=d)
th_count = sns.lineplot(data=df, x='500 episode', y='throughput count',ci='sd')
th_count.set(title='throughput count')

plt.figure()
throughput_means2 = get_mean(throughputs, STEPS)
d = {'episode': range(NUM_EPISODES),
     '500 episode': make_list(NUM_EPISODES, 500),
     'throughput': throughput_means2}
df = pd.DataFrame(data=d)
th = sns.lineplot(data=df, x='500 episode', y='throughput',ci='sd')
th.set(title='throughput')

plt.figure()
reward_means2 = get_mean(rewards, STEPS)
d = {'episode': range(NUM_EPISODES),
     '500 episode': make_list(NUM_EPISODES, 500),
     'reward': reward_means2}
df = pd.DataFrame(data=d)
reward = sns.lineplot(data=df, x='500 episode', y='reward',ci='sd')
reward.set(title='reward')

'''plt.figure()
plt.title('stay count')
plt.xlabel('episode')
plt.ylabel('count')
plt.plot(stay_count)'''

'''throughput_means500 = get_mean(throughputs, 500)
plt.figure()
plt.title('throughput value mean 500')
plt.xlabel('500 episode')
plt.ylabel('throughput value mean')
plt.plot(throughput_means500)'''

"""plt.figure()
plt.title('throughput')
plt.xlabel('step')
plt.ylabel('throughput')
x_values = list(range(throughputs.__len__()))
y_values = [y for y in throughputs]
plt.scatter(x_values, y_values, s=40)"""

'''throughput_count_means500 = get_mean(throughput_counts, 500)
plt.figure()
plt.title('throughput count mean 500')
plt.xlabel('500 episode')
plt.ylabel('throughput count mean')
plt.plot(throughput_count_means500)'''

'''plt.figure()
plt.title('reward')
plt.xlabel('step')
plt.ylabel('Reward')
plt.plot(rewards)'''

'''reward_means = get_mean(rewards, STEPS)
plt.figure()
plt.title('reward mean')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot(reward_means)'''

'''reward_means50 = get_mean(reward_means, 50)
plt.figure()
plt.title('reward mean 50')
plt.xlabel('50 episodes')
plt.ylabel('Reward')
plt.plot(reward_means50)'''

'''reward_means500 = get_mean(reward_means, 500)
plt.figure()
plt.title('reward mean 500')
plt.xlabel('500 episodes')
plt.ylabel('Reward')
plt.plot(reward_means500)'''

'''plt.figure()
plt.title('scheduler')
plt.xlabel('episode')
plt.ylabel('lr')
plt.plot(scheduler_lrs)'''

'''plt.figure()
plt.title('eps')
plt.xlabel('step')
plt.ylabel('epsilon')
plt.plot(epslions)

plt.figure()
plt.title('loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.plot(losses)'''


"""def create_sphere(cx, cy, cz, r):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + cx
    y = r * y + cy
    z = r * z + cz
    return (x, y, z)"""

# 3D 그래프 그리기
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(0, env.MAX_LOC)
ax.set_ylim3d(env.MAX_LOC, 0)
ax.set_zlim3d(0, env.MAX_LOC)
color_list = ("olive", "orange", "green", "blue", "purple", "black", "cyan", "pink", "brown", "darkslategray")

nodes = []
nodes.append(env.source)
nodes.append(env.dest)
nodes.append(env.agent2)
"""for i in range(0, env.MAX_STEPS//10, 1):
    for j in range(10):
        # 구 그리기
        (x, y, z) = create_sphere(scatters[i][0], scatters[i][1], scatters[i][2], scatters[i][3])
        ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
        ax.plot_surface(x, y, z, color=color_list[i%6], linewidth=0, alpha=0.1)
        # 점 찍기
        ax.scatter(np.transpose(scatters)[0], np.transpose(scatters)[1], np.transpose(scatters)[2],marker='o', s=60, c=color_list[i])"""
ax.scatter(np.transpose(scatters_front)[0], np.transpose(scatters_front)[1], np.transpose(scatters_front)[2],
           marker='o', s=60, c='orange')
ax.scatter(np.transpose(scatters_middle)[0], np.transpose(scatters_middle)[1], np.transpose(scatters_middle)[2],
           marker='o', s=60, c='red')
ax.scatter(np.transpose(scatters_tail)[0], np.transpose(scatters_tail)[1], np.transpose(scatters_tail)[2], marker='o',
           s=60, c='purple')

ax.scatter(np.transpose(nodes)[0], np.transpose(nodes)[1], np.transpose(nodes)[2], marker='o', s=80, c='cyan')
plt.show()'''
# env.render()
env.close()
plt.ioff()
plt.show()
