import random
import numpy as np

class RandomAgent(): 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def __init__(self, env, action_space, observation_space, name='Random'):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name
        
    def get_action(self, env, obs):
        observation, reward, termination, truncation, info = env.last()
        action_mask = obs['action_mask']
        possible_action = list(np.where(action_mask ==1)[0])
        try:
            action = random.choice(possible_action)
        except:
            action = None
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass