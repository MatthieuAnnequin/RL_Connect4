from IPython.display import clear_output
import matplotlib.pyplot as plt

def run_episode(env, agents, display=False):
    observation, reward, termination, truncation, info = env.last()
    done = False
    while not done: 
        
        if termination or truncation:
            done = True
            #break
        
        # red play (player 0)
        observation, reward, termination, truncation, info = env.last()
        action = agents[0].get_action(env,observation)
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