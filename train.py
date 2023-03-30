from pettingzoo.classic import connect_four_v3
from Agents.QLearnerAgent import QLearner
from Agents.RandomAgent import RandomAgent
from Run.run_N_episodes import run_N_episodes
from Run.save_agent import save_agent

lr = 0.1
gamma = 0.9999
env = connect_four_v3.env(render_mode='rgb_array')
env.reset()
possible_action = {0,1,2,3,4,5,6}
agents = [QLearner(possible_action, env.observation_space, gamma=gamma, lr=lr),RandomAgent(possible_action, env.observation_space)]
run_N_episodes(env, agents, N_episodes=1000)
save_agent(agents[0], 'qlearner_values.json')
print(len(agents[0].q_values))
