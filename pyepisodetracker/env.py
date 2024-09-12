from gymnasium import Wrapper, Env
from peasyprofiller.profiller import profiller as pprof

class AtariWrapper(Wrapper):
    def __init__(self, env: Env, verbose: bool=True):
        pprof.start("Environment")
        super().__init__(env)

        self.env = env
        self.frames_per_action = 4

        self.step_report = 100
        self.total_steps = 0

        self.verbose = verbose

        pprof.stop("Environment")
    

    def reset(self, seed=None):
        pprof.start("Environment")
        self.total_steps = 0

        obs, info = self.env.reset(seed=seed)
        pprof.stop("Environment")

        return obs, info
    

    def step(self, action_data):
        pprof.start("Environment")
        i = 0
        reward = 0.0
        while i < self.frames_per_action and reward == 0.0:
            obs, reward, terminated, info, done = self.env.step(action_data)
            i += 1
        
        self.total_steps += 1

        if self.total_steps % self.step_report == 0 and self.verbose:
            print(f"{self.total_steps} steps")
        pprof.stop("Environment")

        return obs, reward, terminated, info, done