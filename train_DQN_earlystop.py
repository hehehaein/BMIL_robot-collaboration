# -*- coding: utf-8 -*-
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os

from collections import namedtuple, deque
from itertools import count


from ray.tune.registry import register_env
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from gymExample.gym_example.envs.my_dqn_env import My_DQN

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
os.environ["PYTHONHASHSEED"]=str(seed)



# GPU를 사용할 경우
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
num_episodes = 128000
STEPS = 20
DISCOUNT_FACTOR = 0.8
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = (EPS_START-EPS_END) / (num_episodes * STEPS * 0.5)
TARGET_UPDATE = 40
UPDATE_FREQ = 20
BUFFER = 100000
LEARNING_RATE = 1e-4

now = time.localtime()
str = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}'.format(
    num_episodes, STEPS, DISCOUNT_FACTOR, EPS_DECAY, TARGET_UPDATE, UPDATE_FREQ, '1th',
    BUFFER, LEARNING_RATE,  seed, now.tm_hour, now.tm_min, now.tm_sec)
path = os.path.join(os.getcwd(), 'results')

n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
optimal_net = DQN().to(device)
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(BUFFER)

steps_done = 0
epslions = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - (EPS_DECAY * steps_done))
    epslions.append(eps_threshold)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(-1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

losses = []
#최적화의 한 단계를 수행함
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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss")])

######################################################################
#
# 아래에서 주요 학습 루프를 찾을 수 있습니다. 처음으로 환경을
# 재설정하고 ``상태`` Tensor를 초기화합니다. 그런 다음 행동을
# 샘플링하고, 그것을 실행하고, 다음 화면과 보상(항상 1)을 관찰하고,
# 모델을 한 번 최적화합니다. 에피소드가 끝나면 (모델이 실패)
# 루프를 다시 시작합니다.
#
# 아래에서 `num_episodes` 는 작게 설정됩니다. 노트북을 다운받고
# 의미있는 개선을 위해서 300 이상의 더 많은 에피소드를 실행해 보십시오.
#
throughputs = []
rewards = []
scatters_front = []
scatters_middle = []
scatters_tail = []

show_state = []
show_next_states = []
i_episode = 0
reward_count = 0
maxs=[]
optimal = 0
stay = 0
opti_count = []
stay_count = []
move_count = 0
for i_episode in tqdm(range(num_episodes)):
    # 환경과 상태 초기화
    state = env.reset()
    state = torch.Tensor(state)
    throughput_count=0
    # print('state type : ',state.type())
    max_count=0
    #optimal = 0
    stay = 0
    move_count=0
    for t in range(0,20,1):
        # 행동 선택과 수행
        action = select_action(state)
        next_state, reward, done, last_set = env.step(action.item())

        state_reshape = np.reshape(state, (env.N+2, 4))
        next_state_reshape = np.reshape(next_state, (env.N+2, 4))

        '''if next_state_reshape[0][0] == 1 and \
                next_state_reshape[0][1] == 1 and\
                next_state_reshape[0][2] == 1 and\
                next_state_reshape[0][3] == 3:
            optimal += 1'''
        if next_state_reshape[0][0] == 1 and \
                next_state_reshape[0][1] == 1 and \
                next_state_reshape[0][2] == 1 and \
                next_state_reshape[0][3] == 3:
            if np.array_equal(next_state_reshape, state_reshape):
                stay += 1

        move_count+=1
        if reward > 6.8:
            max_count +=1
            print(state_reshape[0], next_state_reshape[0],
                  "action:%2d%2d%2d%2d reward:%.6f count:%2d"
                  % (last_set[5], last_set[6], last_set[7], last_set[8], reward, move_count))

        if i_episode > (num_episodes - 2):
            print(state_reshape[0], next_state_reshape[0],
                  "action:%2d%2d%2d%2d throughput:%6.3f foot:%6.3f dispersed:%6.3f move:%6.3f txr:%3d"
                  % (last_set[5], last_set[6], last_set[7], last_set[8],
                     last_set[0], last_set[1], last_set[2], last_set[3], last_set[4]))

        if last_set[0] != 0:
            throughput_count += 1

        rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        if i_episode == (num_episodes - 1):
            scatters_tail.append(np.array(next_state_reshape[0]))
        elif i_episode == num_episodes // 2:
            scatters_middle.append(np.array(next_state_reshape[0]))
        elif i_episode == 0:
            scatters_front.append(np.array(next_state_reshape[0]))

        next_state = torch.Tensor(next_state)
        if done:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        '''if torch.argmax(policy_net(state)) == action:
            max_count += 1'''

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행

        if i_episode % UPDATE_FREQ == 0 :
            optimize_model()
            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

    #opti_count.append(optimal)
    stay_count.append(stay)
    maxs.append(max_count)
    # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if stay > 17:
        optimal_net.load_state_dict(policy_net.state_dict())
    throughputs.append(throughput_count)

f=open(path+'/'+str,'w')
torch.save({
    'target_net': target_net.state_dict(),
    'policy_net': policy_net.state_dict(),
    'optimal_net': optimal_net.state_dict(),
    'optimizer': optimizer.state_dict()
}, path+'/'+str)

print('Complete')

def get_mean(array):
    means = []
    for n in range(0,num_episodes,1):
        sum = 0
        for i in range(0,STEPS,1):
            sum += array[(n*STEPS)+i]
        means.append(sum / STEPS)
    return means

def get_reward_mean2(array):
    means = []
    for n in range(0,num_episodes//50,1):
        sum = 0
        for i in range(50):
            sum+=array[n*(50)+i]
        means.append(sum/50)
    return means

def get_reward_mean3(array):
    means = []
    for n in range(0,num_episodes//500,1):
        sum = 0
        for i in range(500):
            sum+=array[n*(500)+i]
        means.append(sum/500)
    return means

plt.figure()
plt.title('max_count')
plt.xlabel('episode')
plt.ylabel('count')
plt.plot(maxs)

'''plt.figure()
plt.title('optimal count')
plt.xlabel('episode')
plt.ylabel('count')
plt.plot(opti_count)'''

plt.figure()
plt.title('stay count')
plt.xlabel('episode')
plt.ylabel('count')
plt.plot(stay_count)

plt.figure()
plt.title('throughput')
plt.xlabel('episode')
plt.ylabel('throughput_count')
plt.plot(throughputs)

"""plt.figure()
plt.title('throughput')
plt.xlabel('step')
plt.ylabel('throughput')
x_values = list(range(throughputs.__len__()))
y_values = [y for y in throughputs]
plt.scatter(x_values, y_values, s=40)"""

'''plt.figure()
plt.title('reward')
plt.xlabel('step')
plt.ylabel('Reward')
plt.plot(rewards)'''

reward_means = get_mean(rewards)
plt.figure()
plt.title('reward mean')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot(reward_means)

reward_means2 = get_reward_mean2(reward_means)
plt.figure()
plt.title('reward mean2')
plt.xlabel('50 episodes')
plt.ylabel('Reward')
plt.plot(reward_means2)

'''reward_means3 = get_reward_mean3(reward_means)
plt.figure()
plt.title('reward mean3')
plt.xlabel('50 episodes')
plt.ylabel('Reward')
plt.plot(reward_means3)'''

plt.figure()
plt.title('eps')
plt.xlabel('step')
plt.ylabel('epsilon')
plt.plot(epslions)

plt.figure()
plt.title('loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.plot(losses)

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

######################################################################
# 다음은 전체 결과 데이터 흐름을 보여주는 다이어그램입니다.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# 행동은 무작위 또는 정책에 따라 선택되어, gym 환경에서 다음 단계 샘플을 가져옵니다.
# 결과를 재현 메모리에 저장하고 모든 반복에서 최적화 단계를 실행합니다.
# 최적화는 재현 메모리에서 무작위 배치를 선택하여 새 정책을 학습합니다.
# "이전" target_net은 최적화에서 기대 Q 값을 계산하는 데에도 사용되고,
# 최신 상태를 유지하기 위해 가끔 업데이트됩니다.
#
