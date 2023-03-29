from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans

df = pd.read_csv("USArrests.csv").copy()
df.head()
df.index = df.iloc[:,0]
df.index

df.head()
df = df.iloc[:,1:5]

df.isnull().sum()
df.describe().T
df.hist(figsize = (10,10));

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
print(kmeans)


k_fit = kmeans.fit(df)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_


kmeans = KMeans(n_clusters = 2)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_


plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")
merkezler = k_fit.cluster_centers_
plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5)

from mpl_toolkits.mplot3d import Axes3D
kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
merkezler = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=kumeler)
ax.scatter(merkezler[:, 0], merkezler[:, 1], merkezler[:, 2],marker='*',c='#050505',s=1000)

kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
pd.DataFrame({"Eyaletler" : df.index, "Kumeler": kumeler})[0:10]
df["kume_no"] = kumeler
df.head()

df["kume_no"] = df["kume_no"] + 1
df.head()

from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,50))
visualizer.fit(df) 
visualizer.poof()  
kmeans = KMeans(n_clusters = 4)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
pd.DataFrame({"Eyaletler" : df.index, "Kumeler": kumeler})[0:10]





















