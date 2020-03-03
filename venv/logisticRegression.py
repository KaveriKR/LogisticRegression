import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv', sep=',')
data.head()


X = data[data.columns[0:8]]
y = data[data.columns[8]]


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X,y,color='#ef1234')
# plt.show()


def sigmoid(X,weights):
    z = np.dot(X,weights)

    return 1.0 / (1.0 + np.exp(-z))



def loss (h,y):
    return  (-y * np.log(h) - (1-y) * np.log(1-h)).mean()

def gradient_descent(X ,h,y):
    return np.dot(X.T,(h-y))/y.shape[0]

def update_weight_loss(weight,learning_rate,gradient):
    return weight - learning_rate* gradient


num_iter = 10000
theta = np.zeros(8)

for i in range(num_iter):
    h = sigmoid(X,theta)
    gradient = gradient_descent(X,h,y)
    theta = update_weight_loss(theta, 0.1,gradient)

def decision_boundary(s):

 if s >=0.5:
  print(1)
 else:
     print (0)

s= sigmoid([3,126,88,41,235,39.3,0.704,27],theta)
decision_boundary(s)



