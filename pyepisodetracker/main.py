from pyepisodetracker.env import AtariWrapper
import gymnasium as gym
from gymnasium.utils.play import play
from keymap import KEYMAP

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    pass


###############
# Startup gym #
###############
env = AtariWrapper(gym.make("ALE/Freeway-v5", render_mode="rgb_array"))
print(env.render_mode, env.env.render_mode)
play(env, callback=callback, keys_to_action=KEYMAP)
