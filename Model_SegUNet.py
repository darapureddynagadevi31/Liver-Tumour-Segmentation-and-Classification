from keras.layers import Input
from keras.models import Model
# from keras.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.src.layers import Convolution2D, BatchNormalization, Activation, Concatenate, Reshape, MaxPooling2D
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def segunet(input_shape, n_labels, sol, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    inputs = Input(shape=input_shape)

    # encoder
    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPooling2D(pool_size)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # between encoder and decoder
    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(pool_5)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([conv_16, mask_5])
    concat_1 = Concatenate()([unpool_1, conv_13])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(concat_1)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(512, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_19, mask_4])
    concat_2 = Concatenate()([unpool_2, conv_10])

    conv_20 = Convolution2D(512, (kernel, kernel), padding="same")(concat_2)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(512, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(256, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_22, mask_3])
    concat_3 = Concatenate()([unpool_3, conv_7])

    conv_23 = Convolution2D(256, (kernel, kernel), padding="same")(concat_3)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(256, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    conv_25 = Convolution2D(128, (kernel, kernel), padding="same")(conv_24)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_25, mask_2])
    concat_4 = Concatenate()([unpool_4, conv_4])

    conv_26 = Convolution2D(128, (kernel, kernel), padding="same")(concat_4)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation("relu")(conv_26)
    conv_27 = Convolution2D(64, (kernel, kernel), padding="same")(conv_26)
    conv_27 = BatchNormalization()(conv_27)
    conv_27 = Activation("relu")(conv_27)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_27, mask_1])
    concat_5 = Concatenate()([unpool_5, conv_2])

    conv_28 = Convolution2D(int(sol[0]), (kernel, kernel), padding="same")(concat_5)
    conv_28 = BatchNormalization()(conv_28)
    conv_28 = Activation("relu")(conv_28)

    conv_29 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_28)
    conv_29 = BatchNormalization()(conv_29)
    conv_29 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_29)

    outputs = Activation(output_mode)(conv_29)
    print("Build decoder done..")

    segunet = Model(inputs=inputs, outputs=outputs, name="SegUNet")

    return segunet


def Model_SegUNetPlusPlus(image, GT, sol):
    if sol is None:
        sol = [1, 2, 3]
    input_shape = image.shape
    n_labels = 20
    kernel = 3
    pool_size = (2, 2)
    output_mode = 'softmax'
    loss = 'categorical_crossentropy'
    optimizer = 'adadelta'
    # set mode
    model = segunet(input_shape, n_labels, sol, kernel, pool_size, output_mode)
    segunet.summary()

    # compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # set callbacks
    cp_cb = ModelCheckpoint(
        filepath='./Images/segmentations/1/img-1.png',
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
        period=2,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="auto")
    tb_cb = TensorBoard(log_dir='./Segunet/Predicted', write_images=True)

    # fit with genarater
    model.fit_generator(
        generator=image[:50],
        steps_per_epoch=6000,
        epochs=5,
        validation_data=image[50:],
        validation_steps=1000,
        callbacks=[cp_cb, es_cb, tb_cb],
    )
    results = model.predict(GT)
    return results


