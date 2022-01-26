#!/usr/bin/env python
# coding: utf-8

# The first thing to do when we're working with ML is to visualize our data to know its behavior and distribution.
# This first observation of data makes it possible to learn more about them being the fastest and most useful way to know which techniques are the most appropriate in _pre_ and _pos_ processing.

#dataset load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
filename= "data/pima-indians-diabetes.csv"
nombres= ["preg","plas","pres","skin","test","mass","pedi","age","class"]
df=pd.read_csv(filename, names=nombres,delimiter=",")



# Uivariate plots allow us to visualize individual attributes without interactions;
# the main purpose of these plots is to learn something about the distribution, trend and spread of each attribute.

# # From the shape, you can get a quick idea of whether an attribute is Gaussian, skewed, or even has an exponential distribution.

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms
fig= plt.figure (figsize =(10,10))
ax=fig.gca() #formateo del tamaño
df.hist(ax=ax)
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
# Univariate Histograms with seaborn library

f, axes =plt.subplots(3,3, figsize=(14,14))
sns.distplot(df["preg"], ax= axes[0,0])
sns.distplot(df["plas"], ax= axes[0,1])
sns.distplot(df["pres"], ax= axes[0,2])
sns.distplot(df["skin"], ax= axes[1,0])
sns.distplot(df["test"], ax= axes[1,1])
sns.distplot(df["mass"], ax= axes[1,2])
sns.distplot(df["pedi"], ax= axes[2,0])
sns.distplot(df["age"], ax= axes[2,1])
sns.distplot(df["class"], ax= axes[2,2])


get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate density line
fig=plt.figure(figsize=(16,16))
ax=fig.gca()
df.plot(ax=ax, kind="density", subplots=True, layout= (3,3), sharex=False)
plt.show



#Density line with seaborn
get_ipython().run_line_magic('matplotlib', 'inline')
#You override the histogram to only see the density line, rug parameter is to see where you have the data.
f, axes =plt.subplots(3,3, figsize=(14,14))
sns.distplot(df["preg"], hist= False, rug = True,ax= axes[0,0])
sns.distplot(df["plas"], hist= False, rug = True,ax= axes[0,1])
sns.distplot(df["pres"], hist= False, rug = True,ax= axes[0,2])
sns.distplot(df["skin"], hist= False, rug = True,ax= axes[1,0])
sns.distplot(df["test"], hist= False, rug = True,ax= axes[1,1])
sns.distplot(df["mass"], hist= False, rug = True,ax= axes[1,2])
sns.distplot(df["pedi"], hist= False, rug = True,ax= axes[2,0])
sns.distplot(df["age"], hist= False, rug = True,ax= axes[2,1])
sns.distplot(df["class"], hist= False, rug = True,ax= axes[2,2])

#Here's a note for the Sheila of the future: add a loop to automate graph generation.

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Boxplot
fig=plt.figure(figsize=(13,13))
ax=fig.gca()
df.plot(ax=ax, kind="box", subplots=True, layout=(3,3), sharex= True)
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Boxplots with Seaborn
f, axes =plt.subplots(3,3, figsize=(14,14))
sns.boxplot(df["preg"], ax= axes[0,0])
sns.boxplot(df["plas"], ax= axes[0,1])
sns.boxplot(df["pres"], ax= axes[0,2])
sns.boxplot(df["skin"], ax= axes[1,0])
sns.boxplot(df["test"], ax= axes[1,1])
sns.boxplot(df["mass"], ax= axes[1,2])
sns.boxplot(df["pedi"], ax= axes[2,0])
sns.boxplot(df["age"], ax= axes[2,1])
sns.boxplot(df["class"], ax= axes[2,2])

# Visualización Multivariable

# Multivariate plots are plots in which we can analyze the relationship or interactions between attributes.
# The objective is to learn something about distribution, trend and distribution in groups of data, usually pairs of attributes.
# <a id="section31"></a>

get_ipython().run_line_magic('matplotlib', 'inline')
# Correlation matrix matplolib
correlations=df.corr()
correlations
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(nombres)
ax.set_yticklabels(nombres)
plt.show


get_ipython().run_line_magic('matplotlib', 'inline')
# Correlatioin matrix matplolib with seaborn
plt.figure(figsize=(10,10))
ax=sns.heatmap(df.corr(),vmax=1,square=True,annot=True, cmap="viridis")
plt.title("Matriz de correlación")
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
# Dispersion matrix matplolib
from pandas.plotting import scatter_matrix
plt.rcParams["figure.figsize"]=[20,15]
scatter_matrix(df)
plt.show()


get_ipython().run_line_magic('matplotlib', 'inline')
# dispersion matrix with Seaborn
sns.pairplot(df)


get_ipython().run_line_magic('matplotlib', 'inline')
# Dispersion matrix by class with Seaborn
sns.pairplot(df, hue="class") #If I want to see with histograms diag_kind="hist"

#This type of graphs help us to understand the overlapping and separation of the class with the characteristic attributes


get_ipython().run_line_magic('matplotlib', 'inline')
# Boxplot for class with Iris dataset

filename="data/iris.data.csv"
names=["Sepal.length","Sepal.width","Petal.lenght","Petal.width","class"]
df=pd.read_csv(filename,names=names)


plt.figure(1)
plt.subplots(figsize=(20,20))

plt.subplot(421) #identificador de cada subplot, debe ser diferente
sns.boxplot(x="class", y="Sepal.length", data=df)
plt.title("Sepal.length")
plt.grid(True)

plt.subplot(422)
sns.boxplot(x="class", y="Sepal.width", data=df)
plt.title("Sepal.width")
plt.grid(True)

plt.subplot(423)
sns.boxplot(x="class", y="Petal.lenght", data=df)
plt.title("Petal.lenght")
plt.grid(True)

plt.subplot(424)
sns.boxplot(x="class", y="Petal.width", data=df)
plt.title("Petal.width")
plt.grid(True)

plt.show()

