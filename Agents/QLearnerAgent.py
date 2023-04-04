from collections import defaultdict
import numpy as np
import random

class QLearner(): 
    """
        Stores the data and computes the observed returns.
    """
    def __init__(self, 
                 action_space, 
                 observation_space, 
                 gamma=0.99, 
                 lr=0.1,
                 eps_init=.5, 
                 eps_min=1e-5,
                 eps_step=1-3,
                 name='Q-learning'):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lr = lr
        
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.name = name
        
        self.reset()
    
    def eps_greedy(self, obs, eps=None):
        if eps is None: 
            eps = self.eps
        action_mask = obs['action_mask']
        possible_action = list(np.where(action_mask ==1)[0])
        if np.random.random() < self.eps:
            try:
                action = random.choice(possible_action)
            except:
                action = None
            return action
        else:
            b = self.q_values[str(obs)]
            return np.random.choice(np.flatnonzero(b == np.max(b))) # argmax with random tie-breaking
            #return np.argmax(b)
        
    def get_action(self, env, obs): 
        return self.eps_greedy(obs)
        
    def update(self, obs, action, reward, terminated, next_obs):
        # update the q-values
        estimate_value_at_next_state = (not terminated) * np.max(self.q_values[str(next_obs)])
        new_estimate = reward + self.gamma * estimate_value_at_next_state
        
        self.q_values[str(obs)][action] = (
            (1 - self.lr) * self.q_values[str(obs)][action] 
            + self.lr * new_estimate
        )
        
        self.epsilon_decay()
        
    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)
        
    def reset(self):
        self.q_values = defaultdict(lambda: np.zeros(len(self.action_space)))