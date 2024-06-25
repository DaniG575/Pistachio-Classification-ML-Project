import joblib
import json
import numpy as np
import base64
import cv2
import pywt

__classifDict = {}
__inverseClassif = {}
__model = None
__scaler = None

def B64ToCv2(b64str):
	data = b64str.split(",")[-1]
	npArr = np.frombuffer(base64.b64decode(data), np.uint8)
	img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
	return img
def Predict(b64string, path = None):
	LoadData()
	croppedImg = CropImages(path,b64string)
	finalImage = FeaturizeImage(croppedImg)
	finalImage = finalImage.reshape(1,32*32*4).astype(float)
	finalImage = __scaler.transform(finalImage)
	print(finalImage)
	prediction = {"prediction": ToClassif(__model.predict(finalImage)[0]), "probabilities": np.around(__model.predict_proba(finalImage)*100,2).tolist()[0]}
	return prediction
def ToClassif(predictionNum):
	return __inverseClassif[predictionNum]
def CropImages(path, b64str):
	image = B64ToCv2(b64str)
	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, binary = cv2.threshold(grayImg, 1, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largestContour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largestContour)
		croppedImage = image[y:y+h, x:x+w]
		return croppedImage
	else:
		return croppedImage
	
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

def FeaturizeImage(croppedImage):
	if croppedImage is not None:
		scalledImg = cv2.resize(croppedImage, (32,32))
		FMImage = WaveletTransform(scalledImg,"db1",5)
		SFMImage = cv2.resize(FMImage, (32,32))
		FinalImage = np.vstack((scalledImg.reshape(32*32*3,1),SFMImage.reshape(32*32*1,1)))
		print(FinalImage)
		return FinalImage
	else:
		return None

def LoadData():
	global __model
	global __classifDict
	global __inverseClassif
	global __scaler
	with open("App/Server/Resources/classDict.json","r") as j:
		__classifDict = json.load(j)
	with open("App/Server/Resources/model.pkl","rb") as j:
		__model = joblib.load(j)
	with open("App/Server/Resources/scaler.pkl","rb") as j:
		__scaler = joblib.load(j)
	__inverseClassif = {n:c for c,n in __classifDict.items()}
	
