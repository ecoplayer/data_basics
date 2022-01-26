#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing: Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#dataset load
df=pd.read_csv("data/countries.csv", sep=";")
df


# basic info
print("Filas y columnas",df.shape)
print("Nombres columnas",df.columns)
df.info()
df.describe()
df.dtypes()
df.isnull().sum() #to see NaNs

# Another very important aspect is to see the type of data that each characteristic (column) has. There are three essential aspects in this section:
# * Columns that do not have the correct type must be transformed. For example sometimes many columns are given as categorical when their values are numeric.
# * When we have numeric and categorical characteristics in the same dataset, we must standardize it, i.e., either all categorical or all numeric.
# * See if any column has NaN values that we will have to treat.

#heatmap plot
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(6,6))
ax=sns.heatmap(df.corr(),vmax=1,square=True,annot=True, cmap="viridis")
plt.title("Matriz de correlación")
plt.show()


#Load a second dataset

df_pop= pd.read_csv("data/countries2.csv")
df_pop
#Filtered by Country to create a new df
df_pop_es= df_pop[df_pop["country"]=="Spain"]
df_pop_es.head()

#barplot of the evolution of the population in Spain
df_pop_es.drop(["country"], axis=1)["population"].plot(kind="bar")
#lineplot of the evolution of the population in Spain
df_pop_es.drop(["country"], axis=1)["population"].plot()


# Barplot with two countries

df_pop_arg= df_pop[df_pop["country"]=="Argentina"]
years=df_pop_es["year"].unique()
pop_ar=df_pop_arg["population"].values
pop_es=df_pop_es["population"].values


df_plot=pd.DataFrame({"Argentina":pop_ar,"Spain":pop_es},index=years)
df_plot

df_plot.plot(kind="bar")

#Barplot with more countries filtered by "languages"

df_espanol= df[df["languages"].str.contains("es") | df["languages"].str.contains("ES")]
df_espanol.plot(x="alpha_3",y=["population","area"],kind="bar")
plt.show()

#OUTLIER DETECTION FUNCTION:
#mean +- 2times standar deviation
anomalies=[]

def find_anomalies(data):
    #set upper and lower limits to 2 std
    data_std=data.std()
    data_mean=data.mean()
    anomaly_cut_off=data_std*2
    lower_limit=data_mean - anomaly_cut_off
    upper_limit=data_mean + anomaly_cut_off
    print(lower_limit.iloc[0])
    print(upper_limit.iloc[0])
    
    #generate outliers
    for index, row in data.iterrows():
        outlier=row #obener la primera columna
        #print(outlier)
        if (outlier.iloc[0]> upper_limit.iloc[0]) or (outlier.iloc[0]<lower_limit.iloc[0]):
            anomalies.append(index)
    return anomalies

find_anomalies(df_espanol.set_index("alpha_3")[["population"]])

print(df_espanol[df_espanol["alpha_3"]=="BRA"].index,df_espanol[df_espanol["alpha_3"]=="USA"].index)


# In this problem I delete BRA y USA:
df_espanol.drop([30,233],inplace=True)
#And again I do a barplot
df_espanol.set_index("alpha_3")[["population","area"]].sort_values(["population"]).plot(kind="bar",rot=65,figsize=(12,5))


# segment by ranges to make a better lecture of the data. In this case by area>110000

df_2=df_espanol.set_index("alpha_3")
df_2=df_2[df_2["area"]>110000]
df_2[["area","population"]].sort_values(["area"]).plot(kind="bar",rot=65,figsize=(10,5))