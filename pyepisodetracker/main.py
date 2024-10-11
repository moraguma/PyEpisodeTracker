from pyepisodetracker.env import EpisodeTrackerWrapper
import gymnasium as gym
from gymnasium.utils.play import play
from keymap import KEYMAP
import cv2
from peasyprofiller.profiller import profiller as pprof

class Monitor():
    def __init__(self):
        self.counter = 0
        self.profile_every = 1000


    def callback(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
        self.counter += 1

        if self.counter % self.profile_every == 0:
            print(f"Plotted @ {self.counter}")
            path = f"results/{self.counter}"
            pprof.plot(path)
            pprof.save_csv(path)


###############
# Startup gym #
###############
cv2.imwrite("test.png", gym.make("ALE/Freeway-v5", render_mode="rgb_array").reset()[0])

env = EpisodeTrackerWrapper(gym.make("ALE/Freeway-v5", render_mode="rgb_array"), 2)
monitor = Monitor()
play(env, callback=monitor.callback, keys_to_action=KEYMAP)
