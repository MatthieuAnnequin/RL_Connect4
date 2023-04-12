import tensorflow as tf
from tensorflow import keras
import numpy as np

class Memory:
    def __init__(self): 
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []
        self.info = []
        
    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(float(new_reward))

#train player 1 against random agent
tf.keras.backend.set_floatx('float64')
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

env = make("connectx", debug=True)
memory = Memory()
epsilon = 1

for i_episode in range(40000):
    
    trainer = env.train([None,'random'])
        
    observation = trainer.reset()['board']
    memory.clear()
    epsilon = epsilon * .99985
    overflow = False
    while True:
        action, _ = get_action(player_1_model,observation,epsilon)
        next_observation, dummy, overflow, info = trainer.step(action)
        observation = next_observation['board']
        observation = [float(i) for i in observation]
        done = check_if_done(np.array(observation).reshape(6,7))
        
        
        memory.add_to_memory(np.array(observation).reshape(6,7,1), action)
        if done[0]:
            #train after each game
            
            train_step(player_2_model, optimizer,
                     observations=np.array(memory.observations),
                     actions=np.array(memory.actions),
                     rewards = memory.rewards)
            
            break