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

svc_model = SVC(kernel = "rbf").fit(X_train, y_train)

print(svc_model)


y_pred = svc_model.predict(X_test)
accuracy_score(y_test, y_pred)

svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],"gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}

svc = SVC()
svc_cv_model = GridSearchCV(svc, svc_params,cv = 10,n_jobs = -1,verbose = 2)
svc_cv_model.fit(X_train, y_train)
print("En iyi parametreler: " + str(svc_cv_model.best_params_))

svc_tuned = SVC(C = 10, gamma = 0.0001).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

















