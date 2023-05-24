# Support Vector Regression
# Normalize edilmiş datayı öğrenirken daha doğru sonuç elde ediyor.
# Grafikte biraz daha aykırı duran sonuçlarda o kadar başarılı değil.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv("sicaklik.csv")
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

sc = StandardScaler()
x1 = sc.fit_transform(x)
y1 = sc.fit_transform(y)

sv = SVR(kernel="rbf")
sv.fit(x1, y1)


plt.scatter(x1,y1)
plt.plot(x1, sv.predict(x1), color="red")
plt.show()













