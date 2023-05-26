# Random Forest Regression
# Random Forest Regression yönteminde diğer öğrenme algoritmalarının
# ortalaması alınıyor. Bu şekilde daha kararlı bir sonuç elde ediliyor.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("bilet.csv")
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

rf = RandomForestRegressor(n_estimators=100, random_state=22)

rf.fit(x, y)

plt.scatter(x, y)
plt.plot(x, rf.predict(x))
plt.show()

















