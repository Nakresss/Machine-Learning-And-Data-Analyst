import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from CatBoost import CatBoostRegresson

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

catb = CatBoostRegresson()
catb_model = catb.fit(X_train,y_train)


y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


catb_grid ={'iterations':[200,550,1000,2000],'learning_rate':[0.01,0.03,0.05,0.1],'depth':[3,4,5,6,7,8]}
catb = CatBoostRegresson()
catb_cv_model = CatBoostRegresson(catb,catb_grid,cv=5, n_jobs=-1,verbose=2)
catb_cv_model.fit(X_trian,y_train)

catb_cv_model.best_params_
catb_tuned = CatBoostRegresson(iterations= 200,learning_rate=0.01,depth=8)
catb_tuned = catb_tuned.fit(X_train,y_train)

y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
