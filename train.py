from pettingzoo.classic import connect_four_v3
from Agents.QLearnerAgent import QLearner
from Agents.RandomAgent import RandomAgent
from Agents.YourAgent import YourAgent
from Run.run_N_episodes import run_N_episodes
from Run.test_agent import test_agent
from Run.run_episode import run_episode

lr = 0.1
gamma = 0.9999
env = connect_four_v3.env(render_mode='rgb_array')
env.reset()
possible_action = {0,1,2,3,4,5,6}

agents = [QLearner(possible_action, env.observation_space, gamma=gamma, lr=lr),RandomAgent()]
run_N_episodes(env, agents, N_episodes=500)

test_agent(env, [agents[0], RandomAgent()], 100)
env.reset()
run_episode(env, [agents[0], YourAgent()], True)