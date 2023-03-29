from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv("USArrests.csv").copy()
df.index = df.iloc[:,0]
df = df.iloc[:,1:5]
#del df.index.name
df.head()

from sklearn.preprocessing import StandardScaler

df = StandardScaler().fit_transform(df)
df[0:5,0:5]

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)

bilesen_df = pd.DataFrame(data = pca_fit,columns = ["birinci_bilesen","ikinci_bilesen","ucuncu_bilesen"])
bilesen_df.head()
pca.explained_variance_ratio_
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))









