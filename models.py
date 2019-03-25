from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import backend as k
k.set_image_data_format('channels_first')


def cnn_model(image_size, number_of_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, image_size[0],
                     image_size[1]), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(number_of_classes, activation="softmax"))
    return model