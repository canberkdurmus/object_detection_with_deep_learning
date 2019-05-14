import json
import os

filename = 'config.json'


def get_parameter(parameter):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data[parameter]


def set_parameter(parameter, value):
    with open(filename, 'r') as f:
        data = json.load(f)
    data[parameter] = value
    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
