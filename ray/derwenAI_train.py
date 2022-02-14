from my_env import Example_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import numpy as np

# ~number of relay node
N = 2
# ~transmission radius max
R_MAX = 4
#location x,y,z
MIN_LOC = 0
MAX_LOC = 4

def get_action(eps,action_space):
    action = np.empty(4)

    if (eps > np.random.rand()):  # uniform random action selection
        action_idx = np.random.randint(0, 3)
        for i in range(4):
            action[i] = action_space[int(action_idx)]
    else:
        # 구해놓은 Q table을 사용
        state_action = self.q_table[int(cur_state), :]
        action_idx = np.argmax(state_action)
        action = self.action_space[int(action_idx)]
        next_txr = cur_txr + action

    # 정해진 tx range 범위 내에서만 action 고르도록...?
    while next_txr < 0 or next_txr >= 3:
        action_idx = np.random.randint(0, 3)
        action = self.action_space[int(action_idx)]
        next_txr = cur_txr + action

    return action, action_idx, next_txr

def main ():
    # init directory in which to save checkpoints
    checkpoint_root = "tmp/exa"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "my_env"
    register_env(select_env, lambda config: Example_v0())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} saved {}"
    n_iteration = 5

    # train a policy with RLlib using PPO
    for n in range(n_iteration):
        result = agent.train()
        checkpoint_file = agent.save(checkpoint_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                checkpoint_file # saved file 주소
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(checkpoint_file)
    env = gym.make(select_env)

    state = env.reset_()
    sum_reward = 0
    n_step = 20

    action = np.empty((N, 4),int)
    for step in range(n_step):
        for i in range(N + 2):
            action[i] = agent.compute_action(state)
            state, reward, done = env.step(i,action)
            sum_reward += reward

        env.render(state)

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()