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

diabetes = pd.read_csv("diabetes.csv")
df = diabetes.copy()
df = df.dropna()
df.head()
df.info()


df["Outcome"].value_counts()
df["Outcome"].value_counts().plot.barh()
df.describe().T


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


loj = sm.Logit(y, X)
loj_model= loj.fit()
loj_model.summary()

from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model

loj_model.intercept_
loj_model.coef_

y_pred = loj_model.predict(X)
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
loj_model.predict(X)[0:10]
loj_model.predict_proba(X)[0:10][:,0:2]

y[0:10]
y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:,1]

y_probs[0:10]
y_pred = [1 if i > 0.5 else 0 for i in y_probs]
y_pred[0:10]
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
loj_model.predict_proba(X)[:,1][0:5]


logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30,random_state = 42)
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)



accuracy_score(y_test, loj_model.predict(X_test))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()


















