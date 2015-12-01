from __future__ import division, print_function


import sys
import json
import numpy as np
import base64

import hashlib
from time import time

import theano.tensor as T


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}

        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.array(dct['__ndarray__'])

    return dct


def generate_uid_from_string(value):
    """ Creates unique identifier from a string. """
    return hashlib.sha256(value).hexdigest()


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': '), cls=NumpyEncoder))


def load_dict_from_json_file(path):
    """ Loads a dict from a json formatted file. """
    with open(path, "r") as json_file:
        return json.loads(json_file.read(), object_hook=json_numpy_obj_hook)


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


def logsumexp(x, axis=None, keepdims=False):
    """ Theano version of `np.logaddexp.reduce` """
    max_value = T.max(x, axis=axis, keepdims=True)
    res = max_value + T.log(T.sum(T.exp(x-max_value), axis=axis, keepdims=True))
    if not keepdims:
        if axis is None:
            return T.squeeze(res)

        slices = [slice(None, None, None)]*res.ndim
        slices[axis] = 0  # Axis being merged
        return res[tuple(slices)]

    return res
