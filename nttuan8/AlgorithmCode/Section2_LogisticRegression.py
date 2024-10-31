import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = pd.read_csv("D:\Download\dataset2.csv").values
row, col = data.shape
X = data[:, 0 : col - 1].reshape(-1, col - 1)
y = data[:, 2].reshape(-1, 1)
plt.scatter(X[:10, 0], X[:10, 1], s = 15, c = 'green', label = 'Yes')
plt.scatter(X[10:, 0], X[10:, 1], s = 15, c = 'red', label = 'No')
plt.legend(loc = 1)
plt.xlabel('Salary (Million)')
plt.ylabel('Experience (Year)')

X = np.hstack((np.ones((row, 1)), X))
w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
numOfIteration = 1000
learning_rate = 0.01
cost = np.zeros((numOfIteration, 1))

for i in range(numOfIteration):
    y_predict = Sigmoid(np.dot(X, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
    w -= learning_rate * np.dot(X.T, y_predict - y)

t = 0.5 
plt.plot((4, 10), (-(w[0] + 4 * w[1] + np.log(1 / t - 1)) / w[2], -(w[0] + 10 * w[1]+ np.log(1 / t - 1)) / w[2]), 'blue')
plt.show()



