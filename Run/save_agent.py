import json
from json import JSONEncoder
import numpy
from tqdm import tqdm

def clean_q_values(agent):
    to_delete = list()
    for qval in tqdm(agent.q_values):
        if numpy.all(agent.q_values[qval] == numpy.zeros(7)):
            to_delete.append(qval)
    for to_del in to_delete:
        del agent.q_values[to_del] 

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_agent(agent, path):

    if agent.name == 'Q-learning':
        clean_q_values(agent)

        numpyData = {"q_values": agent.q_values, "agent_type": agent.name, "gamma": agent.gamma, "lr": agent.lr}
        print("serialize NumPy array into JSON and write into a file")
        with open(path, "w") as write_file:
            json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
        print("Done writing serialized NumPy array into file")