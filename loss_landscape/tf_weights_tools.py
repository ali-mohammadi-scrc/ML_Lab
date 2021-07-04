import numpy as np


def to1D(x):
    if(not hasattr(x, '__iter__')):
        return [x]
    y = []
    for l in x:
        y += to1D(l)
    return y


def array1D_to_model_weights_by_layer(model, array1D):
    new_weights = []
    start = 0
    for layer in model.layers:
        layer_new_weights = []
        for w in layer.get_weights():
            layer_new_weights.append(np.reshape(
                array1D[start:start+w.size], w.shape))
            start += w.size
        new_weights.append(layer_new_weights)
    return new_weights


def weights_by_layer_to_model_weights(weights_by_layer):
    weights = []
    for layer_weights in weights_by_layer:
        weights += list(layer_weights)
    return weights


def model_weights_to_weights_by_layer(model, weights):
    weights_by_layer = []
    start = 0
    for layer in model.layers:
        n_layer_weights = len(layer.get_weights())
        weights_by_layer.append(weights[start:start+n_layer_weights])
        start += n_layer_weights
    return weights_by_layer
