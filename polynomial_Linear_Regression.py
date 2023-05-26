# Polynomial Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pr = PolynomialFeatures(degree=6)

data = pd.read_csv("sicaklik.csv")
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

xPoly = pr.fit_transform(x)
lr = LinearRegression()
lrFitted = lr.fit(xPoly,y)

plt.scatter(x,y)
plt.plot(x, lrFitted.predict(xPoly), color="red")
plt.show()
