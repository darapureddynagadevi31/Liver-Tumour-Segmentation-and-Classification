from prettytable import PrettyTable
import numpy as np


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'REF1', 'REF2', 'REF3', 'REF4', 'REF5', 'REF6', 'REF7', 'REF7', 'REF8']
    for i in range(1):
        value1 = eval[0, :, 4:]
        value2 = eval[1, :, 4:]
        values = [value1[0], value1[1], value1[2], value1[3], value1[4], value2[0], value2[1], value2[2], value2[3], value2[4], value2[5]]
        value = np.asarray(values)
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('-------------------------------------------------- ', '-Adadelta Method Comparison  for classification',
              '--------------------------------------------------')
        print(Table)

plot_results()