import keras
import numpy as np
import cv2 as cv
import torch.nn as nn
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
import torch.nn.functional as F
import torch
from Evaluation import evaluation


def GatedDenseBlock(x, in_channels, out_channels, num_layers):
        bn1 = nn.BatchNorm2d(in_channels)
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        num_layers = num_layers

        # define gate mechanisms
        fc1 = nn.Linear(out_channels, out_channels)
        fc2 = nn.Linear(out_channels, out_channels)

        # define self-attention mechanism
        attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid())

        out = conv1(F.relu(bn1(x)))
        out = conv2(F.relu(bn2(out)))
        out_gate = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out_gate = out_gate.view(out_gate.size(0), -1)
        out_gate = torch.sigmoid(fc1(out_gate))
        out_gate = torch.sigmoid(fc2(out_gate))
        out = out * out_gate.expand_as(out)

        # apply self-attention mechanism
        attention = attention(out)
        out = out * attention

        return torch.cat([x, out], 1)




def GatedDenseNet(x, num_classes=10, growth_rate=12, num_layers=100):
    conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1)
    dense1 = _make_dense_block(2 * growth_rate, growth_rate, num_layers)
    dense2 = _make_dense_block(2 * growth_rate + num_layers * growth_rate, growth_rate, num_layers)
    dense3 = _make_dense_block(2 * growth_rate + 2 * num_layers * growth_rate, growth_rate, num_layers)
    bn = nn.BatchNorm2d(2 * growth_rate + 3 * num_layers * growth_rate)
    fc = nn.Linear(2 * growth_rate + 3 * num_layers * growth_rate, num_classes)

    out = conv1(x)
    out = dense1(out)
    out = dense2(out)
    out = dense3(out)
    out = F.avg_pool2d(F.relu(bn(out)), 4)
    out = out.view(out.size(0), -1)
    out = fc(out)
    return out

def _make_dense_block(in_channels, growth_rate, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(GatedDenseBlock(in_channels + i * growth_rate, growth_rate, num_layers))
    return nn.Sequential(*layers)

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(num_of_class=1):
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)

    return model


def Model_Densenet(train_data, train_target, test_data, test_target, batch, sol=None):
    if sol is None:
        sol = 10
    IMG_SIZE = [32, 32, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = GatedDenseNet(train_target.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_target, steps_per_epoch=10, epochs=10)
    pred = model.predict(test_data)
    Eval = evaluation(test_target, pred)
    return Eval, pred

