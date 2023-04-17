from collections import defaultdict
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models


def convert_obs(obs):
    list_obs = list()
    for ligne in obs['observation']:
        for case in ligne:
            if (case == np.array([0,0])).all():
                list_obs.append(0)
            elif (case == np.array([1,0])).all():
                list_obs.append(1)
            elif (case == np.array([0,1])).all():
                list_obs.append(2)
    return np.array([list_obs])

class DeepQLearner(): 
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
        print('init')
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lr = lr
        
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.name = name
        self.model = tf.keras.models.load_model('my_model\my_model')
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
        self.reset()
    
    def eps_greedy(self, obs, eps=None):
        if eps is None: 
            eps = self.eps
        action_mask = obs['action_mask']
        possible_action = list(np.where(action_mask ==1)[0])
        b = self.q_values[str(obs)]
        observation = convert_obs(obs)
        if (b == np.zeros(7)).all():
            print("here")
            b = self.model.predict(observation)
            action_mask = obs['action_mask']
            possible_action = list(np.where(action_mask ==1)[0])
            action = np.random.choice(np.flatnonzero(b == np.max(b)))
            print(possible_action)
            while action not in possible_action:
                b[0][action] = 0
                action = np.random.choice(np.flatnonzero(b == np.max(b)))
            print(b)
        else :
            print('there')
            action = np.random.choice(np.flatnonzero(b == np.max(b))) # argmax with random tie-breaking
        print(action)
        return action
        #return np.argmax(b)
        
    def get_action(self, env,  obs): 
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