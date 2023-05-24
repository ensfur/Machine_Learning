# Decision Tree Regression
# Bu algoritmada Tree içinde gezinerek doğru değeri bulmaya çalışıyor.
# Ara değer bulmaz. Treede en uygun değeri bulup onu sonuç olarak gösterir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("bilet.csv")
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

dt = DecisionTreeRegressor()
dt.fit(x, y)

plt.figure("Tree Regression")
plt.scatter(x,y)
plt.plot(x, dt.predict(x), color="red")

x1 = np.arange(min(x), max(x), 0.1).reshape(-1, 1)

plt.figure("Modulation Tree Regression")
plt.scatter(x,y)
plt.plot(x1, dt.predict(x1), color="red")

plt.show()
























