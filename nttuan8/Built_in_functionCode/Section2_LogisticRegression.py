import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model

data = pd.read_csv("D:\Download\dataset2.csv").values
X = data[:, 0 : 2].reshape(-1, 2)
y = data[:, 2]

plt.scatter(X[:10, 0], X[:10, 1], s = 15, c = 'green', label = 'Yes')
plt.scatter(X[10:, 0], X[10:, 1], s = 15, c = 'red', label = 'No')
plt.legend(loc = 1)
plt.xlabel('Salary (Million)')
plt.ylabel('Experience (Year)')

Model = linear_model.LogisticRegression()
Model.fit(X, y)
predicted_val = Model.predict(np.array([[10, 1]]).reshape(-1, 2))
print(int(predicted_val))

plt.show()