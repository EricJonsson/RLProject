import random

import gym
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, envID,env):
    """Performs necessary observation preprocessing."""
    if envID in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    elif envID in ['Pong-v0','Breakout-v0']:
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=True)
        return torch.tensor(env._get_obs(), device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
