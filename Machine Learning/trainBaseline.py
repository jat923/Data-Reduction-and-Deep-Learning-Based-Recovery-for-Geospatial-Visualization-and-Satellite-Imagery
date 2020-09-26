####Load Libraries
import csv
import numpy as np
import pandas as pd
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from itertools import repeat
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy.ndimage.filters import gaussian_filter
#from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import pickle
def get_avail_neighbors(arr, mask, d,s,featureCount):
    """
    :param arr:2d array
    :return: returns neighbours of same size by filling mean values of available neighbours if size is less than 4
    """
    # adding symmetric padding
    p = int((d - 1) / 2)

    pArr = np.pad(arr, p, 'symmetric')
    # pArr=np.pad(arr,2,'symmetric)
    pMask = np.pad(mask, p, 'symmetric')
    indices = [[i, j] for i, row in enumerate(mask) for j, x in enumerate(row) if x == 0]
    final = []
    deleteRow = np.delete(arr, np.s_[::2], 1)
    deleteCol = np.delete(deleteRow, np.s_[::2], 0)
    newArray = np.copy(pArr)
    djArr = gaussian_filter(deleteCol, sigma=s)

    for i in range(0,128):
        for j in range(0, 128):
            newArray[2*i+1,2*j+1] = djArr[i,j]

    for ind in (indices):
        r = ind[0]+p
        c = ind[1]+p

        feat1 = [pArr[i, j] for i in range(r - int(d / 2) , r + int(d / 2)+1 ) for j in
                     range(c - int(d / 2) , c + int(d / 2)+1 ) if pMask[i, j] == 1]

        # what happens with checker board?


        feat2=[newArray[i, j] for i in range(r - int(d / 2) , r + int(d / 2)+1 ) for j in
                     range(c - int(d / 2) , c + int(d / 2)+1 ) if pMask[i, j] == 1]
        availFeat = feat1+feat2


        if len(availFeat) < featureCount:
            le = len(availFeat)
            m = min(availFeat)
            availFeat += ([m] * (featureCount - le))
        if len(availFeat)>=featureCount:
            availFeat=availFeat[:featureCount]
        final.append((availFeat))
    return np.array(final)
def get_row_col_mask(data):
    """
    :param data: 2d array
    :return: returns mask by deleting even rows and odd rows
    """
    mask = np.ones(((256, 256)), np.int)
    mask[1::2] = 0
    mask[:, 1::2] = 0
    target = data[~(mask.astype(bool))]
    return mask, target
def level_data(data):
    temp = data
    #temp = preprocessing.normalize(temp)#np.interp(temp, (temp.min(), temp.max()), (0, 1))
    temp =  np.interp(temp, (temp.min(), temp.max()), (0, 1))

    return temp
def read_data(file_name):
    data = np.load(file_name) / 255
    return data


if __name__ == "__main__":

    # Gather training data
    featureCount = 8#6*2 if w=5 or 4*2 if w=3
    xArr = np.empty((0, featureCount))
    yArr = np.empty((0,))
    num = 0
    w=3
    var_name='albedo'
    dir = "training_matrixData_albedo/**.npy"
    test_dir = "testing_matrixData_Albedo"
    sigma=5#sigma is 3 thats why 4 data points are availabel and another 4 data points are added by using gaussian filter
    
    
    algorithm="BR"  #or 'BR'/mlp

    log=var_name+'_'+algorithm+'_'+str(featureCount)+'_kernel_'+str(w)+'_sigma_'+str(sigma)
    os.mkdir(log)
    for file in glob.glob(dir):
        print("Reading File", file)

        data = level_data(read_data(file))
        # Generate mask based on variance.
        mask, target = get_row_col_mask(data)
        compression = 1 - (np.count_nonzero(mask) / (256 * 256))
        #print("compression:", compression)
        # Calling fill_by_interp to fill the missing positions
        # filledData=fill_by_interp(data,mask)
        # Getting available neighbours for each position
        X = get_avail_neighbors(data, mask, w,sigma,featureCount)
        target = np.array(target)
        features = (X)
        xArr = np.append(xArr, features, axis=0)
        yArr = np.append(yArr, target, axis=0)
        num+=1
        print(num)
        if num==500:
            break


    # normalization did not help
    #xArr = (xArr-xArr.min())/(xArr.max() - xArr.min()+.0001)
    #yArr = (yArr-yArr.min())/(yArr.min().max() - yArr.min()+.0001)
    print(xArr.shape)
    print(yArr.shape)
    # Fitting regression model
    #reg = linear_model.BayesianRidge()
    if algorithm=='mlp':
        print('calling method mlp')
        reg = MLPRegressor(random_state=1, max_iter=500)
    else:
        print('calling method BR')
        reg = linear_model.BayesianRidge()
    reg.fit(xArr, yArr)
    # save the model to disk
    filename = log+'_model.sav'
    pickle.dump(reg, open(filename, 'wb'))
    # testing trained model
    final_mae = []
    final_mse = []
    final_r2 = []
    num = 0
    
    outF = open((log+"/"+var_name+'_'+algorithm+'_'+str(featureCount)+'_kernel_'+str(sigma)+".txt"), "w")
    file_list=test_dir+"/**.npy"
    num2=0
    for file in glob.glob(file_list):
        num += 1
        data = level_data(read_data(file))
        temp1 = np.copy(data)
        temp2 = np.copy(data)
        ###########
        mask, target = get_row_col_mask(data)
        # filledData = fill_by_interp(data, mask)
        X = get_avail_neighbors(data, mask, w,sigma,featureCount)


        Y_test = np.array(target)
        X_test = (X)
        #normalization did not help
        #X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + .0001)
        ###########

        Y_pred = reg.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        final_r2.append(r2)
        final_mse.append(mse)
        final_mae.append(mae)
        file=file.strip(test_dir)
        print(file)
        originalfigname=log+'/'+file+'original_'+algorithm+'.png'
        preditedfigname=log+'/'+file+'predicted_'+algorithm+'.png'
        
        # temp1[1::2,1::2]=Y_test.reshape(128,128)
        print("File name:", file)
        print("r2 :", r2)
        print("MSE:", mse)
        print("mae:", mae)
        outF.write("File name: " + file + "\n")
        outF.write("\n")
        outF.write("r2 : " + str(r2))
        outF.write("\n")
        outF.write("MSE: " + str(mse))
        outF.write("\n")
        outF.write("mae: " + str(mae))
        outF.write("\n")
        outF.write("\n")
        count = 0
        for i in range(256):
            for j in range(256):
                if mask[i, j] == 0:
                    temp2[i, j] = Y_pred[count]
                    count += 1
        # visualizing matrices
        fig1 = plt.figure(frameon=False, facecolor='white')
        clev = np.arange(0, temp1.max(), .01)
        plt.contourf(temp1.reshape(256, 256), levels=clev)
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
        plt.contourf(temp2.reshape(256, 256), levels=clev)
        # plt.colorbar()
        plt.axis('off')
        plt.axis('scaled')
        plt.axis('equal')
        fig2.set_size_inches(2.56, 2.56)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.savefig(preditedfigname, dpi=1000)
        plt.close()



    print(final_r2)
    print(final_mae)
    print(final_mse)

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
    outF.write("\n")
    outF.close()
    outF.close()
    #####################
    algorithm="mlp"  #or 'BR'/mlp

    log=var_name+'_'+algorithm+'_'+str(featureCount)+'_kernel_'+str(w)+'_sigma_'+str(sigma)
    os.mkdir(log)
    if algorithm=='mlp':
        print('calling method mlp')
        reg = MLPRegressor(random_state=1, max_iter=500)
    else:
        print('calling method BR')
        reg = linear_model.BayesianRidge()
    reg.fit(xArr, yArr)
    # save the model to disk
    filename = log+'_model.sav'
    pickle.dump(reg, open(filename, 'wb'))
    # testing trained model
    final_mae = []
    final_mse = []
    final_r2 = []
    num = 0
    
    outF = open((log+"/"+var_name+'_'+algorithm+'_'+str(featureCount)+'_kernel_'+str(sigma)+".txt"), "w")
    file_list=test_dir+"/**.npy"
    num2=0
    for file in glob.glob(file_list):
        num += 1
        data = level_data(read_data(file))
        temp1 = np.copy(data)
        temp2 = np.copy(data)
        ###########
        mask, target = get_row_col_mask(data)
        # filledData = fill_by_interp(data, mask)
        X = get_avail_neighbors(data, mask, w,sigma,featureCount)


        Y_test = np.array(target)
        X_test = (X)
        #normalization did not help
        #X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + .0001)
        ###########

        Y_pred = reg.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        final_r2.append(r2)
        final_mse.append(mse)
        final_mae.append(mae)
        file=file.strip(test_dir)
        print(file)
        originalfigname=log+'/'+file+'original_'+algorithm+'.png'
        preditedfigname=log+'/'+file+'predicted_'+algorithm+'.png'
        
        # temp1[1::2,1::2]=Y_test.reshape(128,128)
        print("File name:", file)
        print("r2 :", r2)
        print("MSE:", mse)
        print("mae:", mae)
        outF.write("File name: " + file + "\n")
        outF.write("\n")
        outF.write("r2 : " + str(r2))
        outF.write("\n")
        outF.write("MSE: " + str(mse))
        outF.write("\n")
        outF.write("mae: " + str(mae))
        outF.write("\n")
        outF.write("\n")
        count = 0
        for i in range(256):
            for j in range(256):
                if mask[i, j] == 0:
                    temp2[i, j] = Y_pred[count]
                    count += 1
        # visualizing matrices
        fig1 = plt.figure(frameon=False, facecolor='white')
        clev = np.arange(0, temp1.max(), .01)
        plt.contourf(temp1.reshape(256, 256), levels=clev)
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
        plt.contourf(temp2.reshape(256, 256), levels=clev)
        # plt.colorbar()
        plt.axis('off')
        plt.axis('scaled')
        plt.axis('equal')
        fig2.set_size_inches(2.56, 2.56)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.savefig(preditedfigname, dpi=1000)
        plt.close()



    print(final_r2)
    print(final_mae)
    print(final_mse)

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
    outF.write("\n")
    outF.close()
    outF.close()