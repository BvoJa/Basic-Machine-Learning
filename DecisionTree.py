import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import tree 

data = pd.read_csv("D:\Download\DecisionTreedata.csv")
num = {'UK' : 0, 'USA' : 1, 'N' : 1}
data['Nationality'] = data['Nationality'].map(num)
num = {'NO' : 0, 'YES' : 1}
data['Go'] = data['Go'].map(num)

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = data[features]
y = data['Go']

dtree = tree.DecisionTreeClassifier()
dtree.fit(X, y)
tree.plot_tree(dtree, feature_names = features)
plt.show()