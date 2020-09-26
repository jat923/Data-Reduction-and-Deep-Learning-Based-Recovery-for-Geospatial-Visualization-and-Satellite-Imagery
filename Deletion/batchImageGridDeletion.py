import numpy as np

#import pandas as pd
import math as m
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
def get_var_mask_block(x,g=8):
    """
    Parameters:
        x: 2d array
        g: grid number
    returns:
        mask: mask defining 0 for deleted positions and 1 for available positions
        target: values for 0 positions
    """
    mat_size=x.shape[0]

    mask=np.ones((mat_size,mat_size))
    # taking mat_size/g by mat_size/g blocks from the matrix x
    grids = [x[g*j:g*j+g, g*k:g*k+g] for j in range(mat_size//g) for k in range(mat_size//g)]
    grids=np.array(grids)
    # For all the blocks in grids, calculating variance
    var=np.array([np.std(g) for g in grids]).reshape((mat_size//g),-1)
    pvar=np.copy(var)
    '''
        plt.hist(var)
    plt.title('Height Distribution of US Presidents')
    plt.xlabel('height (cm)')
    plt.ylabel('number')
    plt.show()
    '''

    # Getting the indices of the positions for which variances of neighbours are greater than mean of var
    #var=(var*[var>=(np.mean(var))]).reshape((mat_size//g,mat_size//g),order='A')

    t_val3=np.percentile(var,10)
    t_val = np.median(var)
    t_val2=np.mean(var)
    t_val3=0
    var=(var*[var>=(t_val3)]).reshape((mat_size//g,mat_size//g),order='A')

    # Updating the mask according to variance
    for j in range(mat_size//g):
        for k in range(mat_size // g):
            if var[j, k] == 0:
                mask[g * j:g * j + g, g * k:g * k + g]=0


    return mask
def get_var_mask_pixel(x,g=4):
    """
    Parameters:
        x: 2d array
        g: grid number
    returns:
        mask: mask defining 0 for deleted positions and 1 for available positions
        target: values for 0 positions
    """
    d=g
    p = int((d - 1) / 2)
    #p=1
    mat_size=x.shape[0]
    pArr = np.pad(x, p, 'symmetric')
    mask=np.ones((mat_size,mat_size))
    # taking mat_size/g by mat_size/g blocks from the matrix x
    grids = [pArr[j:j+g, k:k+g] for j in range(mat_size) for k in range(mat_size)]
    var=np.array([np.std(g) for g in grids]).reshape((mat_size),-1)
    grids=np.array(grids)
    # For all the blocks in grids, calculating variance



    # Getting the indices of the positions for which variances of neighbours are greater than mean of var
    #var=(var*[var>=(np.mean(var))]).reshape((mat_size//g,mat_size//g),order='A')
    allper=[np.percentile(var,i) for i in range(1,100,10)]
    t_val3 = np.percentile(var,10)
    t_val = np.median(var)
    t_val2=np.mean(var)
    g=21
    p = int((g - 1) / 2)
    pVar = np.pad(var, p, 'symmetric')
    varGrid=[pVar[j:j+g, k:k+g] for j in range(mat_size) for k in range(mat_size)]
    varOfVar=np.array([np.std(g) for g in varGrid]).reshape((mat_size),-1)
    allper=[np.percentile(varOfVar,i) for i in range(1,100,10)]
    perVar=np.percentile(varOfVar,5)
    cond1=[var>(t_val3)]
    cond2=[varOfVar>perVar]
    var=cond1 and cond2

    mask=var*mask
    return mask
def level_data(data):

    temp = data
    temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))

    return temp
res=256
lowRes=128
num=0
inputDir='temp1'
outputDir='temp1image'
#os.mkdir(outputDir)
#cmap_list=['PiYG','PRGn','BrBG','PuOr','RdGy','RdBu','RdYlGn','RdYlBu', 'Spectral','coolwarm','seismic','bwr']
compression=[]
for file in glob.glob(inputDir+'/**.npy'):
    #cmap_name=random.choice(cmap_list)
    cmap_name='viridis'
    data=level_data(np.load(file))
    matFile=((file.strip(inputDir))[1:])[:-4]
    print(matFile)
    originalfigname = outputDir+'/'+matFile + '.png'


    """
        fig = plt.figure()
    plt.set_cmap(cmap_name)
    clev = np.arange(0, data.max(), .01)
    plt.contourf(data.reshape(res, res),clev,extend='both')
    plt.axis('off')
    plt.axis('scaled')
    plt.axis('equal')
    fig.set_size_inches(2.56, 2.56)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig(originalfigname)
    plt.close()
    """
    temp=np.delete(data,np.s_[1::2],0)
    #print(temp.shape)
    #temp = np.delete(temp, np.s_[1::2], 0)
    #print(temp.shape)
    #temp = np.delete(temp, np.s_[1::2], 1)
    lowData=(np.delete(temp,np.s_[1::2],1))

    ####
    """
    mask = get_var_mask_pixel(lowData,g=4)
    comp2 = np.count_nonzero(mask == 0) / (256 * 256)
    comp1=0.75
    comp=comp2+comp1
    compression.append(comp)
    mask = (np.ones_like(mask) * (~(mask.astype(bool)))).astype(np.uint8)
    np.save('testing_imageData_mask/'+matFile + '_mask.npy', mask)
    print("compression:", comp)
    ##########
    """
    lowResFigname = outputDir+'/'+matFile + '_low.png'

    fig = plt.figure(frameon=False, facecolor='white')
    plt.set_cmap(cmap_name)
    clev = np.arange(0, lowData.max(), .01)
    plt.contourf(lowData.reshape(lowRes, lowRes), clev,extend='both')
    plt.axis('off')
    plt.axis('scaled')
    plt.axis('equal')
    fig.set_size_inches(1.28, 1.28)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig(lowResFigname)
    plt.close()

