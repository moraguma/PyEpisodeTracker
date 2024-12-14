from pyepisodetracker.env import EpisodeTrackerWrapper
import gymnasium as gym
from gymnasium.utils.play import play
from keymap import KEYMAP
import cv2
import numpy as np
from torchbringer.servers.torchbringer_agent import TorchBringerAgent
import torch
from peasyprofiller.profiller import profiller as pprof
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    print(obs_tp1)
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENV_ID = "ALE/Pong-v5"
def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = EpisodeTrackerWrapper(env, 1, np.array([1, 1, 1, 1]), relevant_cat_count=4)

    return env


###############
# Startup gym #
###############
env = make_env()
env.reset()
#for i in range(40):
#    obs, reward, terminated, truncated, info = env.step(0)
#for i in range(10):
#    obs, reward, terminated, truncated, info = env.step(3)
#    cv2.imwrite(f"images/test{i}.png", cv2.cvtColor(env.to_render, cv2.COLOR_BGR2RGB))
play(env, callback=callback, keys_to_action=KEYMAP)


########################
# Startup Torchbringer #
########################
config = {
    "type": "dqn",
    "run_name": "PYET DQN",
    "save_every_steps": 1000000,
    "save_path": "PYETDQN",
    "load_path": "PYETDQN",
    "action_space": {
        "type": "discrete",
        "n": 3
    },
    "gamma": 0.99,
    "target_network_update_frequency": 10000,
    "epsilon": {
        "type": "lin_decrease",
        "start": 1.0,
        "end": 0.1,
        "steps_to_end": 1000000
    },
    "batch_size": 32,
    "loss": "smooth_l1_loss",
    "optimizer": {
        "type": "rmsprop",
        "lr": 0.00025,
        "momentum": 0.95
    },
    "replay_buffer_size": 1000000,
    "min_replay_size": 50000,
    "network": [
        {
            "type": "linear",
            "in_features": 48,
            "out_features": 512
        },
        {"type": "relu"},
        {
            "type": "linear",
            "in_features": 512,
            "out_features": 512
        },
        {"type": "relu"},
        {
            "type": "linear",
            "in_features": 512,
            "out_features": 3
        }
    ]
}
agent = TorchBringerAgent()
agent.initialize(config)


#######
# Run #
#######
EPISODE_COUNT = 10000
LOG_INTERVAL = 100
PLOT_INTERVAL = 1000
for i in range(EPISODE_COUNT):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    reward, terminal = torch.tensor([0.0], dtype=torch.float32, device=device), False
    while not terminal:
        pprof.start("RL Agent")
        action = agent.step(obs, reward, terminal).item()
        pprof.stop("RL Agent")

        obs, reward, terminated, truncated, info = env.step(action)
        obs = None if terminated else torch.tensor(obs, dtype=torch.float32, device=device).flatten().unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        terminal = terminated or truncated

    pprof.start("RL Agent")
    agent.step(obs, reward, terminal)
    pprof.stop("RL Agent")

    if i % LOG_INTERVAL == 0:
        print("Finished %d/%d episodes; ETR - %s; ET - %s" % (i + 1, EPISODE_COUNT, pprof.get_etr(i + 1, EPISODE_COUNT), pprof.get_et()))
    if i % PLOT_INTERVAL == 0:
        name = "results/EP%s" % (i)
        pprof.save_csv(name)
        pprof.plot(name)
        
