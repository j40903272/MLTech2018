
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
        x = l[:-1]
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


from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

class KernelRidgeRegression(object):
    
    def __init__(self, lmbda=0.1, gamma=0.1):
        self.lmbda = lmbda
        self.gamma = gamma

    def fit(self, X, y):
        C = self.K(X) + self.lmbda*np.eye(X.shape[0])
        self.w = np.linalg.solve(C, y)
        self.X_fit_ = X

    def predict(self, X):
        K = self.K(X, self.X_fit_)
        return np.sign(np.dot(K, self.w))
    
    def K(self, X, Y=None):
        K = rbf_kernel(X, Y, gamma=self.gamma)
        return K


# In[6]:


gamma = [32, 2, 0.125]
alpha = [0.001, 1, 1000]


# In[7]:


Ein = []
for g in gamma:
    for a in alpha:
        clf = KernelRidgeRegression(lmbda=a, gamma=g)
        clf.fit(x_train, y_train)
        
        
        pred = clf.predict(x_train)
        err = np.sum(pred != y_train) / 400
        
        Ein.append(err)
        
print (Ein)


# In[8]:


Eout = []
for g in gamma:
    for a in alpha:
        clf = KernelRidgeRegression(lmbda=a, gamma=g)
        clf.fit(x_train, y_train)
        
        pred = clf.predict(x_test)
        err = np.sum(pred != y_test) / 100
        Eout.append(err)
        
print (Eout)

