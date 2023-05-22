# Temel bazı properations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

"""
plt.scatter([1,2,3,4,5],[1,8,2,5,6])
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
"""


data = pd.read_csv("maas.csv")
print(data.head())                                      #ilk 5 veriyi çeker
print("\n--------------------------\n",data.tail())     #son 5 veriyi çeker
print("\n--------------------------\n",data.Tecrube)
print("\n--------------------------\n",data.iloc[:,0])
print("\n--------------------------\n",data.iloc[:,[0,1]])

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
# rastgele veri seçmek için kullanılıyor.
# verilerin %30 test için ayrıldı. Diğerleri için eğitim için ayrıldı.

xnorm = (x - np.min(x)) / (np.max(x) - np.min(x))
ynorm = (y - np.min(y)) / (np.max(y) - np.min(y))

plt.figure(0)
plt.plot(xnorm, ynorm)

xtrain, xtest = sc.fit_transform(xtrain), sc.fit_transform(xtest)
ytrain, ytest = sc.fit_transform(ytrain), sc.fit_transform(ytest)
plt.figure(1)
plt.scatter(xtrain, ytrain)
plt.figure(2)
plt.scatter(xtest, ytest)
plt.show()
