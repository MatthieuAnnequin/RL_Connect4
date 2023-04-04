from Agents.MCTSLearnerAgent import MCTS
from Agents.RandomAgent import RandomAgent

from pettingzoo.classic import connect_four_v3


env = connect_four_v3.env(render_mode='rgb_array')
env.reset()

agent = MCTS(env, time_limit=5, player_role="player_0")


env.reset()
for i in range(5):
    agent = MCTS(env, time_limit=5, player_role="player_0")
    action = agent.start_the_game()[1]
    print(action)
    env.step(action)
    #other agent
    env.step(1)

print(agent.tree)   




