### Introduction
This is a small project created to learn how to build an ML model and how to deploy it including creating a web application for it. 
This model tries to predict if an image of a pistachio is a kirmizi or a siirt pistachio with about 96% accuracy for clear images

### Problems
If you run the web application locally there is an issue where the images provided by the web don't work and just user uploaded images work because of a problem with the CORS policy which resolving goes out of the scoope of this problem.

### Improval
The methods used for this project consists in a support vector machine to classify the type of pistachio.
However, this model is just a simple algorithm that despite performing remarkably well for the task it was designed, it doesn't mean that the same methods can be applied to more complex tasks.
To achieve more complex image classification, a more complex algorythm or deep learning would be necessary.

### Try it out
The code is split between a file used to clean the data, train the model and evaluate the model which is not needed to run the application and the app file where all the necessary components to run the application are. 

1- copy the repository or fork the repo 

2- make sure you have Python 3.12 installed and install all the requirements in requirements.txt 

3- run the file ServerApp.py (App --> Server --> app.py) 

4- open the app.html file (App --> Client --> app.html) 

OR you can try the web version using this link:

### Atribution
DATASET: https://www.muratkoklu.com/datasets/
Citation Request:
1: SINGH D, TASPINAR YS, KURSUN R, CINAR I, KOKLU M, OZKAN IA, LEE H-N., (2022). Classification and Analysis of Pistachio Species with Pre-Trained Deep Learning Models, Electronics,11 (7), 981. https://doi.org/10.3390/electronics11070981. (Open Access)
DOI: https://doi.org/10.3390/electronics11070981

2: OZKAN IA., KOKLU M. and SARACOGLU R. (2021). Classification of Pistachio Species Using Improved K-NN Classifier. Progress in Nutrition, Vol. 23, N. 2. https://doi.org/10.23751/pn.v23i2.9686. (Open Access)
DOI: https://doi.org/10.23751/pn.v23i2.9686
