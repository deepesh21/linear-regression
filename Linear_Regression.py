#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,40,90,160,250,360,490,640,810,1000])


# In[3]:


len(x)


# In[4]:


len(y)


# In[5]:


plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression')
plt.show()


# In[6]:


x=np.c_[np.ones(x.shape[0]),x]


# In[7]:


x.shape


# In[8]:


y.shape


# In[9]:


alpha=0.001
m=y.size
np.random.seed(10)
theta=np.random.rand(2)
theta.shape


# In[10]:


def gradient_descent(x, y, m, theta,  alpha):
    cost_list = []   
    theta_list = []  
    prediction_list = []
    run = True
    cost_list.append(1e10)    
    i=0
    while run:
        prediction = np.dot(x, theta)  
        prediction_list.append(prediction)
        error = prediction - y          
        cost = 1/(2*m) * np.dot(error.T, error)   
        cost_list.append(cost)
        print(cost)
        print("hjvbfdhjvb")
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   
            run = False
        
        i+=1
    cost_list.pop(0)    
    return prediction_list, cost_list, theta_list


# In[11]:


prediction_list, cost_list, theta_list = gradient_descent(x, y, m, theta, alpha)
theta = theta_list[-1]


# In[12]:


ax1=plt.subplot(121)
plt.xlabel('x')
plt.ylabel('y')
plt.title("data points")
ax1.scatter(x[:,1],y,color='C1')
plt.subplot(122)
plt.plot(prediction_list[-1])
plt.xlabel('x')
plt.ylabel('y')
plt.title("regression line")
plt.show()


# In[13]:


plt.plot(cost_list)
plt.xlabel('x')
plt.ylabel('y')
plt.title("cost function")
plt.show()


# In[14]:


plt.plot(theta_list)
plt.xlabel('x')
plt.ylabel('y')
plt.title("gradient descent")
plt.show()


# In[ ]:




