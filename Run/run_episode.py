from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def clean_q_values(agent):
    to_delete = list()
    for qval in agent.q_values:
        if np.all(agent.q_values[qval] == np.zeros(7)):
            to_delete.append(qval)
    for to_del in to_delete:
        del agent.q_values[to_del] 


def run_episode(env, agents, clean =True, display=False):
    observation, reward, termination, truncation, info = env.last()
    done = False
    while not done: 
        
        if termination or truncation:
            done = True
            #break
        
        # red play (player 0)
        observation, reward, termination, truncation, info = env.last()
        action = agents[0].get_action(observation)
        if not(termination or truncation):
            env.step(action)
        next_observation, reward, termination, truncation, info = env.last()
        reward = env.rewards['player_0']
        agents[0].update(observation, action, reward, termination, next_observation)
        observation = next_observation
        if display: 
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()

        if clean and agents[0].name == 'Q-learning':
            clean_q_values(agents[0])
            
        #black play (player 1)
        observation, reward, termination, truncation, info = env.last()
        action = agents[1].get_action(env,observation)
        if not(termination or truncation):
            env.step(action)
        next_observation, reward, termination, truncation, info = env.last()
        reward = env.rewards['player_1']
        agents[1].update(observation, action, reward, termination, next_observation)
        observation = next_observation

        if clean and agents[1].name == 'Q-learning':
            clean_q_values(agents[1])

        if display: 
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            
    return env, agents