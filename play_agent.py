from Run.load_agent import load_agent
from Agents.YourAgent import YourAgent
from Run.run_episode import run_episode
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode='rgb_array')
env.reset()
agent = load_agent('qlearner_values.json', env)
print(len(agent.q_values))
agents = [agent,YourAgent()]
run_episode(env, agents, True, display=True)
