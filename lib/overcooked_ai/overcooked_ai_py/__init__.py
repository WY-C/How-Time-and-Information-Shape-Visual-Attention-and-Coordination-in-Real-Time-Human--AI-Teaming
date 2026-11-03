try:
    from gym.envs.registration import register
except ModuleNotFoundError:
    from gymnasium.envs.registration import register

register(
    id='Overcooked-v0',
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)
