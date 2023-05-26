# Simple Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("maas.csv")

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

lr = LinearRegression()
lr.fit(xtrain, ytrain)
yHead = lr.predict(xtest)

plt.figure("data")
plt.scatter(x, y)

plt.figure("test data and prediction")
plt.scatter(xtest, ytest)

#plt.figure("prediction")
plt.plot(xtest, yHead, color="red")
plt.show()
