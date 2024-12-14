from gymnasium import Wrapper, Env
from peasyprofiller.profiller import profiller as pprof
from pyepisodetracker.episode_tracker import EpisodeTracker, Event, MovementEvent
import numpy as np

class AtariWrapper(Wrapper):
    def __init__(self, env: Env, frames_per_action: int, verbose: bool=True):
        pprof.start("Environment")
        super().__init__(env)

        self.env = env
        self.frames_per_action = frames_per_action

        self.step_report = 100
        self.total_steps = 0

        self.verbose = verbose

        pprof.stop("Environment")
    

    def slice_obs(self, obs):
        return obs[34:194, 8:]


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
            obs, reward, terminated, truncated, info = self.env.step(action_data)
            i += 1
        
        self.total_steps += 1

        pprof.stop("Environment")

        return self.slice_obs(obs), reward, terminated, truncated, info
    
class EpisodeTrackerWrapper(AtariWrapper):
    def __init__(self, env: Env, frames_per_action: int, max_events_per_cat: np.array, relevant_cat_count: int=2, y_event_ordering: bool=True, verbose: bool=True):
        super().__init__(env, frames_per_action, verbose)

        self.y_event_ordering = y_event_ordering
        self.max_events_per_cat = max_events_per_cat
        self.episode_tracker = EpisodeTracker(np.array([np.array([144, 72, 17])]), relevant_cat_count)
        self.to_render = None


    def create_state(self, event: MovementEvent, episode_tracker: EpisodeTracker):
        pos = event.current_pos
        vel = episode_tracker.get_event_vel(event)
        
        return np.array([pos.x / 182.0, pos.y / 152.0, vel.x / 7.75, vel.y / 7.75])
        


    def step(self, action_data):
        obs, reward, terminated, truncated, info = super().step(action_data)
        render, events, categories = self.episode_tracker.process_frame(obs)

        self.to_render = render
        
        return self.create_observation(events, categories), reward, terminated, truncated, info
    

    def create_observation(self, events, categories):
        # Order categories
        for i in range(1, len(categories)):
            key = categories[i]
            j = i - 1

            while j >= 0 and key.size.area() > categories[j].size.area():
                categories[j + 1] = categories[j]
                j -= 1
            categories[j + 1] = key
        cat_max_pos = {}
        cat_first_pos = {}
        cat_pos = {}
        count = 0
        for i in range(len(categories)):
            cat_max_pos[categories[i]] = self.max_events_per_cat[i]
            cat_pos[categories[i]] = 0
            cat_first_pos[categories[i]] = count
            count += self.max_events_per_cat[i]
        
        # Order events
        compare = "y" if self.y_event_ordering else "x"
        for i in range(1, len(events)):
            key = events[i]
            j = i - 1

            while j >= 0 and getattr(key.current_pos, compare) > getattr(events[j].current_pos, compare):
                events[j + 1] = events[j]
                j -= 1
            events[j + 1] = key

        # Create observation
        obs = np.zeros((np.sum(self.max_events_per_cat), 4))
        for event in events:
            if event.category != "MOVEMENT":
                continue

            cat = event.obj_category
            if cat_pos[cat] >= cat_max_pos[cat]:
                continue

            obs[cat_first_pos[cat] + cat_pos[cat], :] = self.create_state(event, self.episode_tracker)
            
            cat_pos[cat] += 1
        
        return obs


    def reset(self, seed=None):
        obs, info = super().reset(seed)
        self.episode_tracker.finish_episode()
        render, events, categories = self.episode_tracker.process_frame(obs)
        
        self.to_render = render
        return self.create_observation(events, categories), info
    

    def render(self):
        return self.to_render