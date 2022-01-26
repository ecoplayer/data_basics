#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load CSV Using Python Standard Library
import csv
import numpy as np
filename= "data/pima-indians-diabetes.csv"
raw_data=open(filename, "r")
reader=csv.reader(raw_data, delimiter= ",", quoting =csv. QUOTE_NONE)
x=list(reader)
data=np.array(x).astype("float")
print(data)
print(data.shape)


# In[3]:


# Load CSV using NumPy
import numpy as np
filename= "data/pima-indians-diabetes.csv"
raw_data=open(filename, "rb")
data=np.loadtxt(raw_data, delimiter = ",")
print(data)
print(data.shape)


# In[9]:


# Load CSV using Pandas
import pandas as pd
filename= "data/pima-indians-diabetes.csv"
nombres= ["preg","plas","pres","skin","test","mass","pedi","age","class"]
df=pd.read_csv(filename, names=nombres,delimiter=",")
print(df.shape)
df.head(3)


# In[10]:


# load dataset from url with pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
dfab=pd.read_csv(url, header=None)
dfab


# <div style="text-align: right"> <font size=5>
#     <a href="#indice"><i class="fa fa-arrow-circle-up" aria-hidden="true" style="color:#004D7F"></i></a>
# </font></div>
# 
# ---

# In[ ]:


#Dataset descriptions


# In[16]:


import pandas as pd
filename= "data/housing.csv"
columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
df=pd.read_csv(filename,names=columns,sep='\s+') #space separated
print(df.shape)
df.head(3)


# In[22]:


df.describe()


# In[ ]:


#It might be a good practice to make a function to automate the process.


# In[25]:


#Iris: multiclass clasification
filename="data/iris.data.csv"
names=["Sepal.length","Sepal.width","Petal.lenght","Petal.width","class"]
df=pd.read_csv(filename,names=names)
df


# In[27]:


df.groupby("class").size() #balanced class, 50 samples per class


# In[1]:


#Sonar, Mines vs. Rocks: Binary classification
filename="data/sonar.all-data.csv"
df=pd.read_csv(filename,header=None)
df


# In[4]:


df.groupby(60).size() #unbalance class M=111 R=97


# In[8]:


#Boston House Price: Regression Problem
filename_reg = 'data/housing.csv'
names_reg = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
df_reg = pd.read_csv(filename_reg, delim_whitespace=True, names=names_reg)
df_reg


# In[ ]:




