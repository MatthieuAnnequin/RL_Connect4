from Agents.MCTSLearnerAgent import MCTS
from Agents.RandomAgent import RandomAgent
from IPython.display import clear_output
import matplotlib.pyplot as plt

from pettingzoo.classic import connect_four_v3
import copy

def run_mcts_episode(env, agents, display=False):
    observation, reward, termination, truncation, info = env.last()
    done = False
    while not done: 
        
        if termination or truncation:
            done = True
            #break
        
        # red play (player 0)
        observation, reward, termination, truncation, info = env.last()
        agent_mcts = MCTS(copy.deepcopy(env), time_limit=120, player_role="player_0")
        start_game = agent_mcts.start_the_game()
        print(start_game)
        action = start_game[1]
        if not(termination or truncation):
            env.step(action)
        next_observation, reward, termination, truncation, info = env.last()
        observation = next_observation
        #black play (player 1)
        observation, reward, termination, truncation, info = env.last()
        action = agents[1].get_action(env,observation)
        if not(termination or truncation):
            env.step(action)
        next_observation, reward, termination, truncation, info = env.last()
        reward = env.rewards['player_1']
        agents[1].update(observation, action, reward, termination, next_observation)
        observation = next_observation

        if display: 
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            
    return env, agents






