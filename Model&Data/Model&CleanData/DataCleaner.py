import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import shutil
import pywt
import pandas as pd
import json

def CropImages(path):
    image = cv2.imread(path)
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayImg, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largestContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largestContour)
        croppedImage = image[y:y+h, x:x+w]
        return croppedImage
    else:
        return None
def WaveletTransform(img, mode="haar", level=1):
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)   
    imArray /= 255;
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffsH=list(coeffs)  
    coeffsH[0] *= 0;  
    imArrayH=pywt.waverec2(coeffsH, mode);
    imArrayH *= 255;
    imArrayH =  np.uint8(imArrayH)
    return imArrayH

def FeaturizeImage(path):
    croppedImage = CropImages(path)
    if croppedImage is not None:
        scalledImg = cv2.resize(croppedImage, (32,32))
        FMImage = WaveletTransform(scalledImg,"db1",5)
        SFMImage = cv2.resize(FMImage, (32,32))
        FinalImage = np.vstack((scalledImg.reshape(32*32*3,1),SFMImage.reshape(32*32*1,1)))
        return FinalImage
    else:
        return None
        

test = CropImages("Model&Data/Model&CleanData/dataset/Kirmizi_Pistachio/kirmizi (1).jpg")
plt.imshow(test)
# plt.show()

x,y = [],[]
if os.path.exists("Model&Data/Model&CleanData/CleanData"):
    shutil.rmtree("Model&Data/Model&CleanData/CleanData")
os.mkdir(("Model&Data/Model&CleanData/CleanData"))
imgPaths = []
classifDict = {}
for imgPath in os.scandir("Model&Data/Model&CleanData/dataset/"):
    print(imgPath)
    if imgPath.is_dir():
        imgPaths.append(imgPath.path)
countClassif = 0
for imgPath in imgPaths:
    count = 0
    classif = imgPath.split("/")[-1]
    classifDict[classif] = countClassif
    countClassif += 1
    croppedFolder = "Model&Data/Model&CleanData/CleanData/" + classif
    os.mkdir(croppedFolder)
    for img in os.scandir(imgPath):
        newImg = CropImages(img.path)
        if newImg is not None:
            x.append(FeaturizeImage(img.path))
            y.append(classifDict[classif])
            imgName = classif + str(count) + ".png"
            newimgPath = croppedFolder + "/" + imgName
            cv2.imwrite(newimgPath, newImg)
            count += 1

print(x[0])
print(y[0])
print(y[1])
print(y[1200])
print(y)
x = np.array(x).reshape(len(x), 4096).astype(float)
np.savez("Model&Data/Model&CleanData/CleanData/data", x=x, y=y)
with open("classDict.json","w") as j:
    j.write(json.dumps(classifDict))


            
            
