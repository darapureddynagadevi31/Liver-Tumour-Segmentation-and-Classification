from keras import Sequential
from keras.applications import EfficientNetB3
from keras.layers import Dense
import numpy as np
import cv2 as cv
from Evaluation import evaluation


def Model_EfficientnetB7(Data, Target):
    IMG_SIZE = [32, 32, 3]
    Feat1 = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Feat1[i, :] = cv.resize(Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Feat2[i, :] = cv.resize(Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=Target.shape[1], activation='relu'))
    model.add(Dense(units=Target.shape[1], activation='relu'))
    model.add(Dense(units=Target.shape[2], activation='sigmoid'))
    model.summary()
    feature = model.get_weights()

    return feature
