import keras
import sys
import numpy as np
import tensorflow_model_optimization as tfmot
from PIL import Image

image_filename = str(sys.argv[1])


def data_preprocess(x_data):
    return x_data / 255


def main():
    img_pil = Image.open(image_filename)
    x = np.array(img_pil)
    x = x.reshape(1, 55, 47, 3)
    x = data_preprocess(x)

    model_1_path = './models/sunglasses_bd_net.h5'
    model_2_path = './models/Pruned_B1.h5'

    with tfmot.sparsity.keras.prune_scope():
        model_1 = keras.models.load_model(model_1_path)
        model_2 = keras.models.load_model(model_2_path)

    pred1 = np.argmax(model_1.predict(x), axis=1)
    pred2 = np.argmax(model_2.predict(x), axis=1)

    if pred1 == pred2:
        print(pred1[0])
    else:
        print(1283)


if __name__ == '__main__':
    main()
