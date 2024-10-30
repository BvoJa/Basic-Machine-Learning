import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv("D:\Download\dataset1.csv").values
N = data.shape[0]
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X = np.hstack((np.ones((N, 1)), X))
w = np.array([0., 1.0]).reshape(-1, 1)
numOfIteration = 100
learning_rate = 0.000001
cost = np.zeros((numOfIteration, 1))

for i in range(numOfIteration):
    r = np.dot(X, w) - y
    cost[i] = 0.5 * np.sum(r * r)
    w -= learning_rate * np.dot(X.T, r)
    
predictedVal = np.dot(X, w)
plt.scatter(data[:, 0], data[:, 1], c = 'blue')
plt.plot((X[0][1], X[N - 1][1]), (predictedVal[0], predictedVal[N - 1]), c = 'red')
plt.show()
print(w)

