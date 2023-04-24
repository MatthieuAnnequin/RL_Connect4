from Run.load_agent import load_agent
from Agents.YourAgent import YourAgent
from Run.run_episode import run_episode
from pettingzoo.classic import connect_four_v3
from Run.run_N_episodes import run_N_episodes
from Agents.RandomAgent import RandomAgent
from Agents.DeepAgent import DeepAgent
from Run.test_agent import test_agent

if __name__ == "__main__":
    env = connect_four_v3.env(render_mode='rgb_array')
    env.reset()
    possible_action = {0,1,2,3,4,5,6}
    agent = load_agent('SavedAgents/qlearner_values_2.json', env)
    print(len(agent.q_values))
    agents = [agent,YourAgent()]
    run_N_episodes(env, agents, N_episodes=1)
    #test_agent(env, agents, 100)
    #print(len(agent.q_values))

    #run_episode(env, agents, True, display=True)
