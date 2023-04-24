# RL_Connect
**Matthieu Annequin - Arthur Guedon - Gillian Keusch - Valentin Odde**

## Introduction
This repository is the code related to project for the Reinforcement Learning Class 

## Installation
To install the project, run :
````
pip install -e .
````

## Try it !
```
from Run.run_N_episodes import run_episode
from Agents.AlphaBetaAgent import AlphaBetaAgent
from Agents.RandomAgent import RandomAgent
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env("rgb_array")
agents = [
    AlphaBetaAgent(env.action_space, env.observation_space, depth=4),
    RandomAgent(env, env.action_space, env.observation_space)
]
env.reset()
run_episode(env, agents, display=True)
```
