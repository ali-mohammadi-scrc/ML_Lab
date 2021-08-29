import numpy as np
from tensorflow.keras.utils import to_categorical

class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def random_samples_by_class(y, class_number, percentage):
    nominal_y = np.nonzero(y)[1]
    class_indices = np.where(nominal_y == class_number)[0]
    n_random_samples = (percentage*class_indices.size)//100
    random_samples_indices = np.random.choice(
        class_indices, size=n_random_samples, replace=False)
    return random_samples_indices


def size_bias(x, y, class_number, error_percentage):
    samples_to_remove_indices = random_samples_by_class(
        y, class_number, error_percentage)
    x_biased = np.delete(x, samples_to_remove_indices, axis=0)
    y_biased = np.delete(y, samples_to_remove_indices, axis=0)
    return x_biased, y_biased


def gaussian_noise_bias(x, y, class_number, error_percentage, mean='auto', var=0.2):
    samples_to_make_noisy_indices = random_samples_by_class(
        y, class_number, error_percentage)
    samples_to_make_noisy = x[samples_to_make_noisy_indices, :, :, :]
    if mean == 'auto':
        mean = np.mean(samples_to_make_noisy)
    noise = np.random.normal(
        loc=mean, scale=var, size=samples_to_make_noisy.shape)
    x_biased = np.array(x)
    x_biased[samples_to_make_noisy_indices, :, :, :] += noise
    y_biased = np.array(y)
    return x_biased, y_biased


def mislabeling_bias(x, y, error_percentage):
    nominal_y = np.nonzero(y)[1]
    class_numbers = np.unique(nominal_y)
    temp_error_percentage = (
        error_percentage*class_numbers.size)//(class_numbers.size-1)
    samples_to_mislable_indices = np.concatenate([random_samples_by_class(
        y, class_number, temp_error_percentage) for class_number in class_numbers], axis=0)
    y_samples_to_mislable = nominal_y[samples_to_mislable_indices]
    np.random.shuffle(y_samples_to_mislable)
    x_biased = np.array(x)
    y_biased = np.array(y)
    y_biased[samples_to_mislable_indices, :] = to_categorical(
        y_samples_to_mislable, class_numbers.size)
    return x_biased, y_biased


def add_bias(x, y, bias, error_percentage=None, class_name=None):
    if bias == 'size_bias':
        return size_bias(x, y, class_names.index(class_name), error_percentage)
    elif bias == 'gaussian_noise_bias':
        return gaussian_noise_bias(x, y, class_names.index(class_name), error_percentage)
    elif bias == 'mislabeling_bias':
        return mislabeling_bias(x, y, error_percentage)
    return (x, y)
