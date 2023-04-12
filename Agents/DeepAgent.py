import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
import numpy as np
import random


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


class DeepAgent(): 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def __init__(self, env, action_space, observation_space, name='Random'):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name
        self.model = tf.keras.models.load_model('my_model\my_model')
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
    def get_action(self, env, obs):
        observation = convert_obs(obs)
        b = self.model.predict(observation)
        action_mask = obs['action_mask']
        possible_action = list(np.where(action_mask ==1)[0])
        action = np.random.choice(np.flatnonzero(b == np.max(b)))
        if action not in possible_action:
            b[0][action] = 0
            print(b)
        action = np.random.choice(np.flatnonzero(b == np.max(b)))
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass