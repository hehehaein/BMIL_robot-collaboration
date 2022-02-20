from gym.envs.registration import register

register(
    id="example-v0",
    entry_point="gymExample.gym_example.envs:Example_v0",
)

register(
    id="fail-v1",
    entry_point="gymExample.gym_example.envs:Fail_v1",
)

register(
    id="ppo-v0",
    entry_point="gymExample.gym_example.envs:My_PPO",
)

register(
    id="dqn-v0",
    entry_point="gymExample.gym_example.envs:My_DQN",
)
