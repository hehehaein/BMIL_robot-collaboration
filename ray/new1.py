from my_env import Example_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil


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
    select_env = "example-v0"
    register_env(select_env, lambda config: Example_v0())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
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
                result["episode_len_mean"], # len 숫자
                checkpoint_file # saved file 주소
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(checkpoint_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done = env.step(i,action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()