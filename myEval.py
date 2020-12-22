import keras
import sys
import h5py
import numpy as np
import tensorflow_model_optimization as tfmot

data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def data_preprocess(x_data):
    return x_data / 255


def main():
    x, y = data_loader(data_filename)
    x = data_preprocess(x)

    model_1_path, model_2_path = '', ''
    if model_filename == 'G1':
        model_1_path = './models/sunglasses_bd_net.h5'
        model_2_path = './models/Pruned_B1.h5'
    elif model_filename == 'G2':
        model_1_path = './models/multi_trigger_multi_target_bd_net.h5'
        model_2_path = './models/Pruned_B2.h5'
    elif model_filename == 'G3':
        model_1_path = './models/anonymous_1_bd_net.h5'
        model_2_path = './models/Pruned_B3.h5'
    elif model_filename == 'G4':
        model_1_path = './models/anonymous_2_bd_net.h5'
        model_2_path = './models/Pruned_B4.h5'
    else:
        print('Model Name Invalid')

    with tfmot.sparsity.keras.prune_scope():
        model_1 = keras.models.load_model(model_1_path)
        model_2 = keras.models.load_model(model_2_path)

    pred1 = np.argmax(model_1.predict(x), axis=1)
    pred2 = np.argmax(model_2.predict(x), axis=1)

    res = []

    for i in range(len(x)):
        if pred1[i] == pred2[i]:
            res.append(pred1[i])
        else:
            res.append(1283)

    res = np.array(res)

    acc = np.mean(np.equal(res, y)) * 100

    print('Classification accuracy:', acc)


if __name__ == '__main__':
    main()
