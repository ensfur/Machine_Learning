# Multi Linear Regression
# Birden fazla sütunun sonuç üzerindeki etkisini gözlemlemek için kullanılır.
# f(x,y,z) = t

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("reklam.csv")
lr = LinearRegression()

x = data.iloc[:,1:4].values
y = data.iloc[:,-1].values.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

lr.fit(x, y)

print(lr.predict(xtest))




