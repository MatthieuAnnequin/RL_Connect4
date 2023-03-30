import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_agent(agent, path):

    if agent.name == 'Q-learning':

        numpyData = {"q_values": agent.q_values, "agent_type": agent.name, "gamma": agent.gamma, "lr": agent.lr}
        print("serialize NumPy array into JSON and write into a file")
        with open(path, "w") as write_file:
            json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
        print("Done writing serialized NumPy array into file")