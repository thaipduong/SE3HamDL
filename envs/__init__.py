from gym.envs.registration import register

register(
    id='MyPendulum-v1',
    entry_point='envs.pendulum:PendulumEnvV1',
)