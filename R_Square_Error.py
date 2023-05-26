# R Square Error Teorisi
# R square error teorisinde oluşturulan line ile gerçek değer arasındaki
# farkın karesi alınıp toplanır. 1'e ne kadar yakın sonuç çıkarsa 
# o kadar iyi bir yöntem kullanılmış olur.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("maas.csv")
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

lr = LinearRegression()
lr.fit(xtrain, ytrain)
yHead = lr.predict(xtest)

print(r2_score(ytest, yHead))





