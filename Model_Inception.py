# from concatenate import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
# from keras.layers.merge import concatenate
import numpy as np
import cv2 as cv

from Evaluation import evaluation


# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = np.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def Model_INCEPTION_FEAT(Data, Tar):
    IMG_SIZE = [32, 32, 3]
    Feat1 = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Feat1[i] = cv.resize(Data[i], [IMG_SIZE[0], IMG_SIZE[1]])
    Data = Feat1  # Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    # define model input
    visible = Input(shape=(32, 32, 3))
    # add inception block 1
    layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
    # add inception block 1
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    # create model
    model = Model(inputs=visible, outputs=layer)
    # summarize model
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'])
    try:
        model.fit(Data, Tar)
        f1 = np.asarray(model.get_weights()[-1 - 1])
    except:
        f1 = np.asarray(model.get_weights()[-1 - 1])
    # Fully Connected Layer Dense FC2
    Feat = cv.resize(f1[0, 0], (f1.shape[2], Tar.shape[0]))
    return Feat


def Model_INCEPTION(train_data, train_target, test_data, test_target, Batch):
    IMG_SIZE = [32, 32, 3]
    IMG_SIZE = [224, 224, 3]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    # define model input
    visible = Input(shape=(32, 32, 3))
    # add inception block 1
    layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
    # add inception block 1
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)  # 128
    # create model
    model = Model(inputs=visible, outputs=layer)
    # summarize model
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'])
    try:
        model.fit(train_data, train_target)
    except:
        pred = np.round(model.predict(test_data)).astype('int')

    Eval = evaluation(pred, test_target)
    return Eval, pred
