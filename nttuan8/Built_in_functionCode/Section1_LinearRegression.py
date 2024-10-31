import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\Download\dataset1.csv")
X, y = df['Area'], df['Price']    

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42)
plt.scatter(X_train, y_train, c = 'red')
plt.scatter(X_test, y_test, c = 'blue')
plt.xlabel('Area')
plt.ylabel('Cost')

slope, intercept, r, p, std_err = stats.linregress(X_train, y_train)
def Func(x):
    return slope * x + intercept 
print(Func(50))

myModel = LinearRegression()
X_train_val = X_train.values.reshape(-1, 1)
myModel.fit(X_train_val, y_train)
print(myModel.predict([[50]]))

plt.plot(X_train, list(map(Func, X_train)))
plt.show()
