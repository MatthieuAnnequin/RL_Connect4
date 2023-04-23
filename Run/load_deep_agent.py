import json
import numpy
from Agents.DeepQLearnerAgent import DeepQLearner
from pettingzoo.classic import connect_four_v3

def load_deep_agent(path, env):
    possible_action = {0,1,2,3,4,5,6}
    print("Started Reading JSON file")
    with open(path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)

        agent_type = numpy.asarray(decodedArray["agent_type"])
        gamma = numpy.asarray(decodedArray["gamma"])
        lr = numpy.asarray(decodedArray["lr"])

        if agent_type == 'Q-learning':
            agent = DeepQLearner(possible_action, env.observation_space, gamma=gamma, lr=lr)
            saved_q_values = numpy.asarray(decodedArray["q_values"])
            for key, value in saved_q_values.flatten()[0].items():
                agent.q_values[key] = value
            return agent