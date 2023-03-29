import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from warnings import filterwarnings
filterwarnings('ignore')


df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=42)
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled[0:5]
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train_scaled, y_train)
y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test, y_pred)

mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],"hidden_layer_sizes": [(10,10,10),(100,100,100),(100,100),(3,5),(5, 3)],"solver" : ["lbfgs","adam","sgd"],"activation": ["relu","logistic"]}


mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params,cv = 10,n_jobs = -1,verbose = 2)
mlpc_cv_model.fit(X_train_scaled, y_train)
print("En iyi parametreler: " + str(mlpc_cv_model.best_params_))

mlpc_tuned = MLPClassifier(activation = "logistic",alpha = 0.1,hidden_layer_sizes = (100, 100, 100),solver = "adam")
mlpc_tuned.fit(X_train_scaled, y_train)
y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test, y_pred)













