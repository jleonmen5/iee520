#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import KFold


# In[11]:


#Visualization Setup


# In[12]:


#Load Data
df = pd.read_csv("Avila-DataSet for miniprojects.csv")
x = df.drop("Class: A, B, C, D, E, F, G, H, I, W, X, Y", axis=1)
y = df["Class: A, B, C, D, E, F, G, H, I, W, X, Y"]


# In[13]:


#Split Data 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state=3)


# In[14]:


#Create Scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[15]:


#Create a Model
model = MLPClassifier(hidden_layer_sizes=(220,220,220), random_state=3)


# In[ ]:


start_time = time.time()
model.fit(x_train,y_train)
run_time = time.time() - start_time
print("Run time: ", run_time)
y_hat = model.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_hat))


# In[8]:


#Create Scaler for Cross Validation
# scaler = StandardScaler()
# scaler.fit(x)
# x= scaler.transform(x)
