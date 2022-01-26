#!/usr/bin/env python
# coding: utf-8

import pandas as pd
filename="data/pima-indians-diabetes.csv"
nombres= ["preg","plas","pres","skin","test","mass","pedi","age","class"]
df=pd.read_csv(filename, names=nombres,delimiter=",")
df.head()

# Dimensions of your data
df.shape

# Data Types for Each Attribute
df.dtypes

# Statistical Summary
pd.set_option("display.width",100) #same columns width.
pd.set_option("precision",3) #num of decimals per measurement.
df.describe()

# Class Distribution
#if there is a difference between classes greater than 50%, the imbalance must be corrected.
#Take into consideration which is your class column
df.groupby("class").size()

# Pairwise Pearson correlations, correlation matrix
pd.set_option("display.width",100) #same columns width.
pd.set_option("precision",3)
df.corr()

# Skew for each attribute
#The skew result shows a positive (right) or negative (left) skew. Values closer to zero show less skew.
df.skew()