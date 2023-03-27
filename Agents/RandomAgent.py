import random
import numpy as np

class RandomAgent: 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def _init_(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
    def get_action(self, env, obs):
        observation, reward, termination, truncation, info = env.last()
        possible_action = {0,1,2,3,4,5,6}
        action_mask = obs['action_mask']
        possible_action = list(np.where(action_mask ==1)[0])
        try:
            action = random.choice(possible_action)
        except:
            action = None
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass