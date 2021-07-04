import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D


def get_filter_wise_normalized_direction(model):
    def frobenius_norm(x): return np.sqrt(np.sum(x**2))
    normalized_random_direction = []
    for model_layer in model.layers:
        model_layer_weights = model_layer.get_weights()
        if isinstance(model_layer, Dense) or isinstance(model_layer, Conv2D):
            model_kernels = model_layer_weights[0]
            model_biases = model_layer_weights[1]
            n_filters = len(model_biases)
            random_kernels = []
            for i in range(n_filters):
                model_kernel = np.take(model_kernels, i, -1)
                random_kernel = np.random.normal(scale=np.sqrt(
                    2/np.count_nonzero(model_kernel)), size=np.shape(model_kernel))
                normalization_factor = frobenius_norm(
                    model_kernel)/(frobenius_norm(random_kernel)+1e-10)
                random_kernel *= normalization_factor
                random_kernels.append(np.expand_dims(
                    random_kernel, np.ndim(random_kernel)))
            random_kernels = np.concatenate(random_kernels, axis=-1)
            normalized_direction_layer = [
                random_kernels, np.zeros(n_filters)]  # ignore biases
        else:  # ignore BN
            normalized_direction_layer = list(
                0 * np.array(model_layer_weights, dtype=np.object))
        normalized_random_direction += normalized_direction_layer
    return np.array(normalized_random_direction, dtype=np.object)


def loss_val_filter_normalized_dir(model, x, y, steps_res=[-1, 1, 50], n_trials=10):
    steps = np.array(range(steps_res[2]))/(steps_res[2]-1)
    steps *= steps_res[1] - steps_res[0]
    steps += steps_res[0]
    w = np.array(model.get_weights(), dtype=np.object)
    evals = []
    for i in range(n_trials):
        d = get_filter_wise_normalized_direction(model)
        trial_evals = []
        for step in steps:
            model.set_weights(w+step*d)
            trial_evals.append(model.evaluate(x=x, y=y, verbose=0))
        model.set_weights(w)
        evals.append(trial_evals)
    return steps, np.array(evals)


def plot_loss_val(steps, trials_evals, plt_colors=('b', 'r'), plt_axes=None, linestyle='--'):
    if plt_axes is None:
        fig, ax_loss = plt.subplots()
        ax_acc = ax_loss.twinx()
    else:
        ax_loss, ax_acc = plt_axes
    loss_color, acc_color = plt_colors
    for evals in trials_evals:
        ax_loss.plot(steps, evals[:, 0], loss_color+linestyle)
        ax_acc.plot(steps, evals[:, 1], acc_color+linestyle)
