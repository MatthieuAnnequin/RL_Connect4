from IPython.display import clear_output
import matplotlib.pyplot as plt

agent_ids = ["player_0", "player_1"]

def run_episode(env, agents, display=False):
    observation, reward, termination, truncation, info = env.last()
    done = False
    while not done: 
        
        if termination or truncation:
            done = True
            #break

        # red play (player 0)
        observation, reward, termination, truncation, _ = env.last()
        if not termination and not truncation:
            action = agents[0].get_action(observation)
            env.step(action)
        next_observation, reward, termination, truncation, _ = env.last()
        reward = env.rewards.get(agent_ids[0])
        agents[0].update(observation, action, reward, termination, next_observation)
        observation = next_observation
        if display: 
            display_board(env)
        #black play (player 1)

        observation, reward, termination, truncation, info = env.last()
        action = agents[1].get_action(env,observation)
        if not(termination or truncation):
            env.step(action)
        next_observation, reward, termination, truncation, _ = env.last()
        reward = env.rewards.get(agent_ids[1])
        agents[1].update(observation, action, reward, termination, next_observation)
        observation = next_observation

        if display: 

            display_board(env)
    return env, agents


def display_board(env):
    clear_output(wait=True)
    plt.imshow(env.render())
    plt.show()
