####Load Libraries
import csv
import numpy as np
import pandas as pd
import glob
import pandas as pd
import matplotlib
import os
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from itertools import repeat
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from numpy import inf
from skimage.measure import *
import cv2
import timeit
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy.linalg import fractional_matrix_power
from scipy.ndimage.filters import gaussian_filter
#from sklearn import preprocessing
import math
from scipy.interpolate import Rbf
import timeit
import os
def get_row_col_mask(data):
    """
    :param data: 2d array
    :return: returns mask by deleting even rows and odd rows
    """
    size=data.shape[0]
    mask = np.ones(((size, size)), np.int)
    mask[1::2] = 0
    mask[:, 1::2] = 0
    target = data[~(mask.astype(bool))]
    return mask, target

def level_data(data):
    temp = data
    temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))

    return temp

def read_dataSatellite(path):
    data = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)[:, :, 1]
    return data
def read_data(file_name):
    data = np.load(file_name) / 255
    return data

def shepardsInterpolation(data,mask,xx,yy,p):
    data=data*mask
    def calcIDWValue(data,ind,xx,yy,p):


        det=(((xx - xx[ind]) ** 2 + (yy - yy[ind]) ** 2)**p)
        weight =1/det
        weight[weight == inf] = 0
        weight=mask*weight
        sum_of_weights = np.sum(weight)
        mul=weight * data
        weighted_values_sum = np.sum(mul)
        return weighted_values_sum / sum_of_weights
    indices = [(i, j) for i, row in enumerate(mask) for j, x in enumerate(row) if x == 0]
    for ind in (indices):
        r = ind[0]
        c = ind[1]
        val=calcIDWValue(data,ind,xx,yy,p)
        data[r,c]=val
    return data
def multiquadricsInterpolation(data,xx,yy,epsilon):
    def deleteMat(arr):
        deleteRow = np.delete(arr, np.s_[::2], 1)
        deleteCol = np.delete(deleteRow, np.s_[::2], 0)
        return deleteCol
    
    newXx = deleteMat(xx)
    newYy = deleteMat(yy)
    newData = deleteMat(data)
    newFunc = Rbf(newXx, newYy, newData, function='multiquadric',epsilon=1)

    val = newFunc(xx, yy)
    return val
def saveSatelliteData(temp1,temp2,file,alg,dataName,folderName):
    
    
    originalfigname = (os.path.join(str(alg)+str(dataName)+str(file) + 'Original.png'))
    preditedfigname = (os.path.join(str(alg)+str(dataName)+str(file) + 'Predicted.png'))
    plt.imsave(originalfigname, temp1,cmap=plt.get_cmap('gray'))

    plt.imsave(preditedfigname, temp2,cmap=plt.get_cmap('gray'))


def saveWeatherData(temp1,temp2,file,alg,dataName,folderName,result_dir):
    file = file.strip(folderName).strip("/")
    print(file)
    originalfigname = result_dir + '/' + file + 'original_' + alg + '.png'
    preditedfigname = result_dir + '/' + file + 'predicted_' + alg + '.png'
    fig1 = plt.figure(frameon=False, facecolor='white')
    clev = np.arange(0, temp1.max(), .01)
    plt.contourf(temp1.reshape(256, 256), levels=clev,extend='both')
    # plt.colorbar()

    plt.axis('off')
    plt.axis('scaled')
    plt.axis('equal')
    fig1.set_size_inches(2.56, 2.56)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig(originalfigname, dpi=1000)
    plt.close()

    fig2 = plt.figure(frameon=False, facecolor='white')
    clev = np.arange(0, temp2.max(), .01)
    plt.contourf(temp2.reshape(256, 256), levels=clev,extend='both')

    # plt.colorbar()
    plt.axis('off')
    plt.axis('scaled')
    plt.axis('equal')
    fig2.set_size_inches(2.56, 2.56)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig(preditedfigname, dpi=1000)
    plt.close()


    return originalfigname,preditedfigname

if __name__ == "__main__":
    """
    g=200


    data=np.arange((g*g)).reshape((g,g))
    print(data)
    size = data.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    mask,target=get_row_col_mask(data)

    print(mask)
    a=timeit.default_timer()
    val=multiquadricsInterpolation(data,xx,yy,epsilon=1)
    print(timeit.default_timer()-a)
    """
    size=256

    yy, xx = np.mgrid[0:size, 0:size]
    p=8
    #data=shepardsInterpolation(data,mask,xx,yy,p)
    #print(data)


    #val=multiquadricsInterpolation(data,xx,yy,epsilon=1)
    #print(val)
    # testing trained model
    final_mae = []
    final_mse = []
    final_r2 = []
    final_time=[]
    num = 0
    alg="Multiquadrics"
    dataName="Smois"
    
    folderName="temp1Test"
    mse_error = []
    psnr_error = []
    ssim_error = []
    
    result_dir=alg+dataName+folderName
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    test_dir = folderName+"/**.npy"
    file_dir=os.path.join(alg+dataName+".txt")
    outF = open(file_dir, "w")
    for file in glob.glob(test_dir):
        num += 1
        data = level_data(read_data(file))
        temp1 = np.copy(data)
        temp2 = np.copy(data)
        ###########
        mask, target = get_row_col_mask(data)
        start = timeit.default_timer()
        #data2 = shepardsInterpolation(data, mask, xx, yy, p)
        
        data2=multiquadricsInterpolation(data,xx,yy,epsilon=1)
        # filledData = fill_by_interp(data, mask)
        stop = timeit.default_timer()
        time=stop-start
        Y_test = np.array(target)
        Y_pred=data2[~(mask.astype(bool))]
        # normalization did not help
        # X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + .0001)
        ###########

        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        final_r2.append(r2)
        final_mse.append(mse)
        final_mae.append(mae)
        final_time.append(time)


        # temp1[1::2,1::2]=Y_test.reshape(128,128)
        print("File name:", file)
        print("r2 :", r2)
        print("MSE:", mse)
        print("mae:", mae)
        print("time:",time)
        outF.write("File name: " + file + "\n")
        outF.write("\n")
        outF.write("r2 : " + str(r2))
        outF.write("\n")
        outF.write("MSE: " + str(mse))
        outF.write("\n")
        outF.write("mae: " + str(mae))
        outF.write("\n")
        outF.write("time: " + str(time))
        outF.write("\n")
        temp2=data2
        # visualizing matrices
        originalfigname,predictedfigname=saveWeatherData(temp1,temp2,file,alg,dataName,folderName,result_dir)
        
        originalImg = plt.imread(originalfigname)
        originalImg = (originalImg[:, :, 0:3]).astype(np.float64)

        predictedImg = plt.imread(predictedfigname )
        predictedImg = (predictedImg[:, :, 0:3]).astype(np.float64)
        mse = compare_mse(originalImg, predictedImg)
        psnr = compare_psnr(originalImg, predictedImg)
        print(psnr)

        print(mse)
        ssim = compare_ssim(originalImg, predictedImg, multichannel=True)
        print(ssim)

        mse_error.append(mse)
        psnr_error.append(psnr)
        ssim_error.append(ssim)


print("mean r2:", np.mean(final_r2))
print("mean mae:", np.mean(final_mae))
print("mean mse:", np.mean(final_mse))

outF.write("\n")
outF.write("mean r2 : " + str(np.mean(final_r2)))
outF.write("\n")
outF.write("mean MSE: " + str(np.mean(final_mse)))
outF.write("\n")
outF.write("mean mae: " + str(np.mean(final_mae)))
outF.write("\n")
outF.write("time: " + str(np.mean(final_time)))
outF.write("\n")

ssimMean = np.mean(ssim_error)
psnrMean = np.mean(psnr_error)
mseMean = np.mean(mse_error)

outF.write("\n IMAGE Results Final")
outF.write("\nssim : " + str(ssimMean))
outF.write("\nMSE: " + str(mseMean))
outF.write("\npsnr: " + str(psnrMean))
outF.close()