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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

hit = pd.read_csv("hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])
y = df["Salary"]
x_ = df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
x = pd.concat([X_, dms[["League_N","Division_W","NewLeague_N"]]], axis = 1)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
mlp_model.n_layers_
mlp_model.hidden_layer_sizes
mlp_model = MLPRegressor(hidden_layer_sizes=(100,20)).fit(X_train_scaled, y_train)
mlp_model.n_layers_
mlp_model.hidden_layer_sizes

y_pred = mlp_model.predict(X_train_scaled)[0:5]
np.sqrt(mean_squared_error(y_test,y_pred))

print(mlp_model)
mlp_params = {'alpha': [0.1,0.01,0.02,0.005],'hidden_layer_sizes':[(20,20),(100,50,150),(300,200,150,)],'activation':['relu','logistic']}

mlp_cv_model = GridSearchCV(mlp_model, mlpparams,cv=10)
mlp_cv_model.fit(X_train_scaled,y_train)
mlp_cv_model.best_params_

mlp_tuned = MLPRegressor(alpha= 0.02,hidden_layes_sizes =(100,50,150))
mlp_tuned.fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))

















