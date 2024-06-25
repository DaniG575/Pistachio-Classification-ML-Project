import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = np.load("Model&Data/Model&CleanData/CleanData/data.npz")
x,y = data["x"], data["y"]
scaler = StandardScaler()
x = scaler.fit_transform(x)
xtrain,xeval,ytrain,yeval = train_test_split(x,y,test_size=0.2, random_state=0)
model = SVC(kernel="rbf", C=0.5)
model.fit(xtrain,ytrain)
print(model.score(xeval,yeval))

# Finding the best model and params with hyperparameter tunning

def HyperParameterTune(model, input, target, splits, cost, params):
	if cost == 0:
		clf = GridSearchCV(model, params, cv=splits, return_train_score=False)
		clf.fit(input, target)
		df = pd.DataFrame(clf.cv_results_)
		df = pd.concat([df.filter(regex="^param"), df["mean_test_score"]])
		bp = clf.best_params_
		bs = clf.best_score_
	if cost == 1:
		rs = RandomizedSearchCV(model, params, cv=splits, return_train_score=False, n_iter=8)
		rs.fit(input, target)
		df = pd.DataFrame(rs.cv_results_)
		df = pd.concat([df.filter(regex="^param"), df["mean_test_score"]])
		bp = rs.best_params_
		bs = rs.best_score_
	return bs,bp

def EvalModelsParams(input, target, splits, costs):
	scores = []
	model_params = {
		'svm': {
			'model': SVC(),
			'params' : {
				'C': [0.2,0.5,1,2,10,5],
				'kernel': ['rbf',"poly"]
			}  
		},
		'random_forest': {
			'model': RandomForestClassifier(),
			'params' : {
				'n_estimators': [5,10,50]
			}
		},
		'logistic_regression' : {
			'model': LogisticRegression(solver='liblinear'),
			'params': {
				'C': [0.2,0.5,1,2,10,5]
			}
		},
	}
	for model_name, params in model_params.items():
		best_score, best_params = HyperParameterTune(params["model"], input, target, splits, costs, params["params"] )
		scores.append({
			'model': model_name,
			'best_score': best_score,
			'best_params': best_params
		})
	df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
	return df

# print(EvalModelsParams(x,y,5,1))

# Best model svm with rbf C = 2
model = SVC(kernel="rbf", C=2, probability=True)
model.fit(x,y)
print(x[0])
print(model.score(x,y))
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")