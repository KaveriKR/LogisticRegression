import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('haberman.data')
print(data.shape)
data.head()
X = data[data.columns[0:2]]
y = data.iloc[:,-1]
X1 = data.loc[ y ==1]
X2 = data.loc[ y ==2]
print(X)
X11 = X1[X1.columns[0]]
X12 = X1[X1.columns[1]]
X13 = X1[X1.columns[2]]

X21 = X2[X2.columns[0]]
X22 = X2[X2.columns[1]]
X23 = X2[X2.columns[2]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X11,X12,X13,color='#00ee00')
ax.scatter(X21,X22,X23,color='#ee0000')

plt.show()


def sigmoid(X,weights):
    z = np.dot(X,weights)
    return 1.0 / (1.0 + np.exp(-z))

def gradient_ascent(X,h,y):
    return np.dot(X.T, y- h)

def update_weight(weight, learning_rate, gradient):
    return weight + learning_rate * gradient

def decision_boundary(s):

 if s >=0.5:
  print(2)
 else:
     print (1)

theta = np.zeros(2)
num_iter = 100000

for i in range(num_iter):
    h = sigmoid(X, theta)
    gradient = gradient_ascent(X,h,y)
    theta = update_weight(theta,0.1,gradient)
    

     
print(theta)
     
s = sigmoid([72,58,0],theta)
decision_boundary(s)
