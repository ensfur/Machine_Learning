# R Square Error Teorisi

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





