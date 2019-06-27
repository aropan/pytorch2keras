import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_constant(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert constant layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting constant ...')

    params_list = params['value'].numpy()

    layers[scope_name + '_np'] = params_list  # ad-hoc
    layers[scope_name] = params_list  # ad-hoc
