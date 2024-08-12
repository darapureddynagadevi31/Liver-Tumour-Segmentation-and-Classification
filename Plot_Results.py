from itertools import cycle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics

from Image_Results import Image_Results, Sample_Images


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'Resnet', 'Inception', 'Densenet', 'Mobilenet', 'MEB7-RLSTM']
    for i in range(1):
        value = eval[3, :, 4:]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('-------------------------------------------------- ', '-Adadelta Method Comparison  for classification',
              '--------------------------------------------------')
        print(Table)

        # for i in range(eval.shape[0]):
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[2]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if j == 5:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
        Optimizer = ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'AdaGrad']
        plt.plot(Optimizer, Graph[:, 0], color=[0.7, 0, 0], linewidth=3, marker='x', markerfacecolor='b', markersize=16,
                 label="Resnet")
        plt.plot(Optimizer, Graph[:, 1], color=[0.5, 0.5, 0.1], linewidth=3, marker='D', markerfacecolor='red', markersize=12,
                 label="Inception")
        plt.plot(Optimizer, Graph[:, 2], color=[0, 0.5, 0.5], linewidth=3, marker='x', markerfacecolor='green', markersize=16,
                 label="Densenet")
        plt.plot(Optimizer, Graph[:, 3], color='c', linewidth=3, marker='D', markerfacecolor='cyan', markersize=12,
                 label="Mobilenet")
        plt.plot(Optimizer, Graph[:, 4], color='k', linewidth=3, marker='x', markerfacecolor='black', markersize=16,
                 label="MEB7-RLSTM")
        plt.xlabel('Optimizer')
        plt.ylabel(Terms[Graph_Term[j]])
        # plt.tick_params(axis='x', labelrotation=25)
        # plt.ylim([60, 100])
        plt.legend(loc='best')
        path1 = "./Results/%s_line.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MBO', 'AOA', 'AGTO', 'LO', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(1):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='MBO-ASUnet++')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='AOA-ASUnet++')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='AGTO-ASUnet++')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='LO-ASUnet++')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='ELO-ASUnet++')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['Resnet', 'Inception', 'Densenet', 'Mobilenet', 'MEB7-RLSTM']

    # Classifier = ['TERMS', 'Xgboost', 'DT', 'NN', 'FUZZY', 'KNN', 'PROPOSED']
    for a in range(1):  # For 5 Datasets
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        Actual = np.load('Target.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_Segmentation_results():
    Eval_all = np.load('Eval_all_Segmentation.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'MBO', 'AOA', 'AGTO', 'LO', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color=[0.5, 0.5, 0], width=0.10, label="MBO-ASUnet++")
            ax.bar(X + 0.10, stats[i, 1, :], color='g', width=0.10, label="AOA-ASUnet++")
            ax.bar(X + 0.20, stats[i, 2, :], color=[0, 0.5, 0.5], width=0.10, label="AGTO-ASUnet++")
            ax.bar(X + 0.30, stats[i, 3, :], color='m', width=0.10, label="LO-ASUnet++")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="ELO-ASUnet++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=2, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_alg-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 4, :], color=[0.5, 0.6, 0], width=0.10, label="UNet")
            ax.bar(X + 0.10, stats[i, 5, :], color='g', width=0.10, label="ResUnet")
            ax.bar(X + 0.20, stats[i, 6, :], color='m', width=0.10, label="TransUnet")
            ax.bar(X + 0.30, stats[i, 7, :], color=[0, 0.6, 0.7], width=0.10, label="SegUnet++")
            ax.bar(X + 0.40, stats[i, 8, :], color='k', width=0.10, label="ELO-ASUnet++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_met-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_results()
    # plot_Segmentation_results()
    # plotConvResults()
    Plot_ROC_Curve()
    # Image_Results()
    # Sample_Images()
