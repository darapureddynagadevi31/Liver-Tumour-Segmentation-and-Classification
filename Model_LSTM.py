import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from Evaluation import evaluation


def Model_LSTM(train_data, train_target, test_data, test_target, Batch, sol=25):
    out, model = LSTM_train(train_data, train_target, test_data, Batch, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, Batch, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=Batch, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model


