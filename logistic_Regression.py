# Logistic Regression (classification)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

sc = StandardScaler()
lr = LogisticRegression()

data = pd.read_csv("urun.csv")
x = data.iloc[:,0:2].values
y = data.iloc[:,-1].values.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

S = data[data.iloc[:,2]==0]
B = data[data.iloc[:,2]==1]

plt.figure()
plt.scatter(S.iloc[:,0].values, S.iloc[:,1].values, color="red")
plt.scatter(B.iloc[:,0].values, B.iloc[:,1].values, color="blue")

xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.fit_transform(xtest)

lr.fit(xtrain1, ytrain)

yHead = lr.predict(xtest1)

print(lr.score(xtest1,ytest))

cm = confusion_matrix(ytest, yHead)
print(cm)

plt.show()

