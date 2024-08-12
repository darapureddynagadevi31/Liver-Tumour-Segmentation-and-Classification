import numpy as np
import os
import cv2 as cv
import nibabel as nib
from tkinter.filedialog import askopenfilename
from numpy import matlib
from AGTO import AGTO
from AOA import AOA
from Global_Vars import Global_Vars
from Image_Results import Image_Results, Sample_Images
from LO import LO
from MBO import MBO
from Model_DenseNet import Model_Densenet
from Model_Efficientnet_LSTM import Model_Efficientnet_LSTM
from Model_Inception import Model_INCEPTION
from Model_MobileNet import Model_MobileNet
from Model_Resnet import Model_Resnet
from Model_SegUNet import Model_SegUNetPlusPlus
from Objective_Function import objfun_Segmentation
from Plot_Results import plot_results, plotConvResults, Plot_ROC_Curve, plot_Segmentation_results
from Proposed import Proposed

# Read Images
an = 0
if an == 1:
    Dir = './Data-image/'
    file_name = os.listdir(Dir)
    Oriimg = []
    for i in range(len(file_name)):  # len(file_name)
        file = Dir + file_name[i]
        file1 = os.listdir(file)
        for j in range(len(file1)):
            files = Dir + file_name[i] + '/' + file1[j]
            # files1 = os.listdir(files)
            image = nib.load(files).get_fdata()
            for k in range(image.shape[2]):
                print(i, k)
                image1 = image[:, :, k].astype(np.uint8)
                im = cv.resize(image1, (128, 128))
                Oriimg.append(im)
    np.save('Images.npy', Oriimg)

# Read Groudtruth
an = 0
if an == 1:
    Dir = './segmentations/'
    file_name = os.listdir(Dir)
    GT = []
    for i in range(len(file_name)):
        file = Dir + file_name[i]
        filename = askopenfilename()
        image = nib.load(filename).get_fdata()
        for k in range(image.shape[2]):
            print(i, k)
            image1 = image[:, :, k].astype(np.uint8)
            im = cv.resize(image1, (128, 128))
            GT.append(im * 255)
    np.save('GT.npy', GT)

# generate Target from GT
an = 0
if an == 1:
    GT = np.load('GT.npy', allow_pickle=True)
    Target = []
    for i in range(len(GT)):
        GT1 = GT[i] * 255
        if max(GT1.flatten()) == 0:
            Tar = 0
        else:
            Tar = 1
        Target.append(Tar)
    Target = np.reshape(Target, (-1, 1))
    np.save('Target.npy', Target)

# Optimization for Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)  # Load the images
    Target = np.load('GT.npy', allow_pickle=True)  # Load the images
    Global_Vars.Feat = Images
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # 1 for Hidden Neuron Count and 1 for Epochs, 1 for Step Per Epochs
    xmin = matlib.repmat([5, 5, 300], Npop, 1)
    xmax = matlib.repmat([255, 50, 1000], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(Chlen):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_Segmentation
    Max_iter = 10

    print("MBO...")
    [bestfit1, fitness1, bestsol1, time1] = MBO(initsol, fname, xmin, xmax, Max_iter)  # MBO

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("AGTO...")
    [bestfit3, fitness3, bestsol3, time3] = AGTO(initsol, fname, xmin, xmax, Max_iter)  # AGTO

    print("LO...")
    [bestfit4, fitness4, bestsol4, time4] = LO(initsol, fname, xmin, xmax, Max_iter)  # LO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    Bestsol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', Bestsol)  # Save the Bestsoluton

# Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('GT.npy', allow_pickle=True)  # Load the images
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    seg_images = Model_SegUNetPlusPlus(Images, Target, BestSol[4, :])
    np.save('Segment5.npy', seg_images)

# Classification by varying batch size
an = 0
if an == 1:
    Data = np.load('Segment5.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)  # Load the images
    Image = []
    for u in range(len(Data)):
        # print(u)
        images1 = np.resize(Data[u], (128, 128))
        Image.append(images1)
    Image = np.asarray(Image)
    Target = np.load('Target.npy', allow_pickle=True)
    EVAL = []
    Batch = [4, 16, 32, 64, 128]
    learnper = round(Image.shape[0] * 0.75)
    Train_Data = Image[:learnper, :]
    Train_Target = Target[:learnper, :]
    Test_Data = Image[learnper:, :]
    Test_Target = Target[learnper:, :]
    for i in range(len(Batch)):
        Eval = np.zeros((5, 14))
        Eval[0, :], pred1 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target, Batch[i])
        Eval[1, :], pred2 = Model_Densenet(Train_Data, Train_Target, Test_Data, Test_Target, Batch[i])
        Eval[2, :], pred3 = Model_INCEPTION(Train_Data, Train_Target, Test_Data, Test_Target, Batch[i])
        Eval[3, :], pred4 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target, Batch[i])
        Eval[4, :], pred5 = Model_Efficientnet_LSTM(Data, Target, Batch[i])
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)

plot_results()
plot_Segmentation_results()
plotConvResults()
Plot_ROC_Curve()
Image_Results()
Sample_Images()
