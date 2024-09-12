from gymnasium import Wrapper, Env
from peasyprofiller.profiller import profiller as pprof
from pyepisodetracker.episode_tracker import EpisodeTracker
import numpy as np

class AtariWrapper(Wrapper):
    def __init__(self, env: Env, verbose: bool=True):
        pprof.start("Environment")
        super().__init__(env)

        self.env = env
        self.frames_per_action = 1

        self.step_report = 100
        self.total_steps = 0

        self.verbose = verbose

        pprof.stop("Environment")
    

    def slice_obs(self, obs):
        return obs[15:196, 8:]


    def reset(self, seed=None):
        pprof.start("Environment")
        self.total_steps = 0

        obs, info = self.env.reset(seed=seed)
        pprof.stop("Environment")

        return self.slice_obs(obs), info
    

    def step(self, action_data):
        pprof.start("Environment")
        i = 0
        reward = 0.0
        while i < self.frames_per_action and reward == 0.0:
            obs, reward, terminated, info, done = self.env.step(action_data)
            i += 1
        
        self.total_steps += 1

        pprof.stop("Environment")

        return self.slice_obs(obs), reward, terminated, info, done
    
class EpisodeTrackerWrapper(AtariWrapper):
    def __init__(self, env: Env, verbose: bool=True):
        super().__init__(env, verbose)

        self.episode_tracker = EpisodeTracker(np.array([np.array([142, 142, 142]), np.array([170, 170, 170]), np.array([214, 214, 214])]))
        self.to_render = None
    

    def step(self, action_data):
        obs, reward, terminated, info, done = super().step(action_data)
        events = self.episode_tracker.process_frame(obs)

        self.to_render = events[0]
        return events, reward, terminated, info, done
    

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        events = self.episode_tracker.process_frame(obs)
        self.episode_tracker.finish_episode()
        
        self.to_render = events[0]
        return events, info
    

    def render(self):
        return self.to_render