import json
import numpy
from Agents.QLearnerAgent import QLearner

def load_agent(path, agent):
    if agent.name == 'Q-learning': 
        print("Started Reading JSON file")
        with open(path, "r") as read_file:
            print("Converting JSON encoded data into Numpy array")
            decodedArray = json.load(read_file)

        saved_q_values = numpy.asarray(decodedArray["q_values"])
        for key, value in saved_q_values.flatten()[0].items():
            agent.q_values[key] = value
        return agent