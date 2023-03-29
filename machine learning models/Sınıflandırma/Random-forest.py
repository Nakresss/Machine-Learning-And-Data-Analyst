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
#X = df["Pregnancies"]
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
print(rf_model)


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

rf_params = {"max_depth": [2,5,8,10],"max_features": [2,5,8],"n_estimators": [10,500,1000],"min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model,rf_params,cv = 10,n_jobs = -1,verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("En iyi parametreler: " + str(rf_cv_model.best_params_))


rf_tuned = RandomForestClassifier(max_depth = 10,max_features = 8,min_samples_split = 10,n_estimators = 1000)
rf_tuned.fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},index = X_train.columns)
Importance.sort_values(by = "Importance",axis = 0,ascending = True).plot(kind ="barh", color = "r")
plt.xlabel("Değişken Önem Düzeyleri")






















