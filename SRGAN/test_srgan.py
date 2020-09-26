from skimage.measure import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('genModel2Albedo.h5')
mse_error=[]
psnr_error=[]
ssim_error=[]
dir=glob.glob("testing_imageData_Albedo/**_low.png")
outF = open("outLogFile_smoisRegular.txt", "w")
""
for img in dir:
    print(img)
    high = (img.strip('_low.png')) + '.png'

    print(high)
    originalImg= plt.imread(high)
    originalImg= originalImg[:, :, 0:3]
    lowImg= plt.imread(img)
    lowImg= lowImg[:, :, 0:3]
    
    lowImg=np.expand_dims(lowImg, axis=0)
    predictedImg=model.predict(lowImg)
    predictedImg=predictedImg[0,:,:,:]
    print(np.min(predictedImg))
    print(np.max(predictedImg))
    predictedImg = (predictedImg - np.min(predictedImg)) / (np.max(predictedImg) - np.min(predictedImg))
    print(np.min(predictedImg))
    print(np.max(predictedImg))
    plt.imsave((img.strip('_low.png')) + '_Predicted.png',predictedImg)
    mse = compare_mse(originalImg, predictedImg)
    psnr = compare_psnr(originalImg, predictedImg)
    ssim = compare_ssim(originalImg, predictedImg, multichannel=True)
    outF.write("File name: "+img+"\n")
    outF.write("\n")
    outF.write("ssim : "+str(ssim))
    outF.write("MSE: "+str(mse))
    outF.write("psnr: "+str(psnr))
    outF.write("\n")
    mse_error.append(mse)
    psnr_error.append(psnr)
    ssim_error.append(ssim)
ssimMean=np.mean(ssim_error)
psnrMean=np.mean(psnr_error)
mseMean=np.mean(mse_error)

outF.write("\nFinal")
outF.write("\nssim : "+str(ssimMean))
outF.write("\nMSE: "+str(mseMean))
outF.write("\npsnr: "+str(psnrMean))

