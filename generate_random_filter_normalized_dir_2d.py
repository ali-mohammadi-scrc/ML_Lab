import numpy as np
import os

from keras_example_cifar10_resnet.resnet import resnet_v1
from keras_example_cifar10_resnet.cifar10 import load_cifar10
from loss_landscape.random_direction import loss_val_filter_normalized_dir_2d

x_test, y_test = load_cifar10(return_part='test')

input_shape = np.shape(x_test[0])
depth = 20
loss = 'categorical_crossentropy'
metrics = ['accuracy', 'MSE']
steps_res = [-1, 1, 50]

base_path = 'Results/Models'
model_base_path = 'Results/Models'

model = resnet_v1(input_shape=input_shape, depth=depth)
model.compile(loss=loss, metrics=metrics)

for model_name in os.listdir(model_base_path):
    print('**** ', model_name, ' ****')

    model_dir = os.path.join(model_base_path, model_name, 'latest.h5')
    save_dir = os.path.join(base_path, model_name,
                            'random_trials_' + str(steps_res[-1]) + '_2d')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.load_weights(model_dir)
    steps, evals = loss_val_filter_normalized_dir_2d(
        model, x_test, y_test, steps_res=steps_res)
    np.save(os.path.join(save_dir, 'evals'), evals, allow_pickle=True)
    np.save(os.path.join(save_dir, 'steps'), steps, allow_pickle=True)
