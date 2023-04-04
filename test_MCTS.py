from Agents.MCTSLearnerAgent import MCTS
from Agents.RandomAgent import RandomAgent

from pettingzoo.classic import connect_four_v3


env = connect_four_v3.env(render_mode='rgb_array')
env.reset()

agent = MCTS(env, time_limit=5, player_role="player_0")

agent.start_the_game()




