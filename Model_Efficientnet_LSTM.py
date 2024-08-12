from Evaluation import evaluation
from Model_EfficientnetB7 import Model_EfficientnetB7
from Model_LSTM import Model_LSTM


def Model_Efficientnet_LSTM(Data, Target, Batch):
    Feature = Model_EfficientnetB7(Data, Target)

    learnper = round(Feature.shape[0] * 0.75)
    Train_Data = Feature[:learnper, :]
    Train_Target = Target[:learnper, :]
    Test_Data = Feature[learnper:, :]
    Test_Target = Target[learnper:, :]

    Eval, Pred_lstm = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Batch)
    Pred = evaluation(Pred_lstm, Test_Target)
    return Eval, Pred
