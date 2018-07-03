
# coding: utf-8

# In[1]:


import numpy as np
import os


# In[2]:


X = []
Y = []
with open("hw2_lssvm_all.dat", "r") as f:
    for line in f:
        l = line.split()
        x = l[:-1] + [1]
        y = l[-1]
        X.append(x)
        Y.append(y)
        
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)


# In[3]:


print (X.shape)
print (Y.shape)


# In[4]:


x_train, y_train, x_test, y_test = X[:400], Y[:400], X[400:], Y[400:]


# In[5]:


from sklearn.linear_model import RidgeClassifier

class RidgeRegression(object):
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda

    def fit(self, X, y):
        C = X.T.dot(X) + self.lmbda*np.eye(X.shape[1])
        self.w = np.linalg.inv(C).dot(X.T.dot(y))

    def predict(self, X):
        return np.sign(X.dot(self.w))


# In[6]:


alpha = [0.01, 0.1, 1, 10, 100]


# In[7]:


Ein = []

for a in alpha:
    clf = RidgeRegression(lmbda=a)
    clf.fit(x_train, y_train)
    
    pred = clf.predict(x_train)
    err = np.sum(pred != y_train) / 400
    Ein.append(err)
        
print (Ein)


# In[8]:


Eout = []

for a in alpha:
    clf = RidgeRegression(lmbda=a)
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    err = np.sum(pred != y_test) / 100
    Eout.append(err)
        
print (Eout)

