
#Dataset Load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = "data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
print(data.shape)

data.describe()

# separate array into input and output components
X = data.drop(["class"],axis=1).values 
Y = data["class"].values

#another way to separate X,Y
# array=data.values
# X=array[:,0:8]
# Y=array[:,8]


# In the following lines you can see how the original data looks like and compare it with each transformation.

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(14, 14))
sns.distplot(data["preg"], ax=axes[0, 0])
sns.distplot(data["plas"], ax=axes[0, 1])
sns.distplot(data["pres"], ax=axes[0, 2])
sns.distplot(data["skin"], ax=axes[1, 0])
sns.distplot(data["test"], ax=axes[1, 1])
sns.distplot(data["mass"], ax=axes[1, 2])
sns.distplot(data["pedi"], ax=axes[2, 0])
sns.distplot(data["age"], ax=axes[2, 1])
sns.distplot(data["class"], ax=axes[2, 2])


# MINMAXSCALER -- NORMALIZATION, Rescale data (between 0 and 1)
# This transformation is useful for Linear Regression, Neural Networks and kNN.

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler() #default feature_range=(0,1)
X_esc=scaler.fit_transform(X)
np.set_printoptions(precision=3)#set num of decimals

print(X_esc)

# Numpy Array to dataframe
transform_df=pd.DataFrame(X_esc,columns=names[:-1])
transform_df


get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms plot
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(14, 14))
sns.distplot(transform_df["preg"], ax=axes[0, 0])
sns.distplot(transform_df["plas"], ax=axes[0, 1])
sns.distplot(transform_df["pres"], ax=axes[0, 2])
sns.distplot(transform_df["skin"], ax=axes[1, 0])
sns.distplot(transform_df["test"], ax=axes[1, 1])
sns.distplot(transform_df["mass"], ax=axes[1, 2])
sns.distplot(transform_df["pedi"], ax=axes[2, 0])
sns.distplot(transform_df["age"], ax=axes[2, 1])

# STANDARDSCALER -- Standardize data (0 mean, 1 stdev)

# Suitable for algorithms that assume a Gaussian distribution on the input variables and work best with rescaled data,
# such as Linear Regression, Logistic Regression and Linear Discriminant Analysis.
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X)
X_esc=scaler.transform(X)
# summarize transformed data
print(X_esc.shape)

# Array to dataframe
std_df=pd.DataFrame(X_esc,columns=names[:-1])
std_df

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms plots
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(14, 14))
sns.distplot(std_df["preg"], ax=axes[0, 0])
sns.distplot(std_df["plas"], ax=axes[0, 1])
sns.distplot(std_df["pres"], ax=axes[0, 2])
sns.distplot(std_df["skin"], ax=axes[1, 0])
sns.distplot(std_df["test"], ax=axes[1, 1])
sns.distplot(std_df["mass"], ax=axes[1, 2])
sns.distplot(std_df["pedi"], ax=axes[2, 0])
sns.distplot(std_df["age"], ax=axes[2, 1])

# Normalize data (length of 1)
# This preprocessing method can be useful for sparse data sets (many zeros) with attributes of varying scales when using algorithms
# that weight input values such as Neural Networks and algorithms that use distance measures such as k-Nearest Neighbours.
from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(X)
X_normal=scaler.transform(X)
# summarize transformed data
print(X_normal)

# Array to dataframe
Normal_df = pd.DataFrame(X_normal, columns=names[:-1])
Normal_df

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms plots
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(14, 14))
sns.distplot(Normal_df["preg"], ax=axes[0, 0])
sns.distplot(Normal_df["plas"], ax=axes[0, 1])
sns.distplot(Normal_df["pres"], ax=axes[0, 2])
sns.distplot(Normal_df["skin"], ax=axes[1, 0])
sns.distplot(Normal_df["test"], ax=axes[1, 1])
sns.distplot(Normal_df["mass"], ax=axes[1, 2])
sns.distplot(Normal_df["pedi"], ax=axes[2, 0])
sns.distplot(Normal_df["age"], ax=axes[2, 1])

# BINARIZATION
#In this case all values equal to or less than 0 are marked with 0 and all values above 0 are marked with 1.
#https://www.geeksforgeeks.org/sklearn-binarizer-in-python/

from sklearn.preprocessing import Binarizer
# summarize transformed data
binarizer=Binarizer(threshold=0.0).fit(X) #if value>0.0 --> 1, else 0
X_bin=binarizer.transform(X)
print(X_bin)

# Array to dataframe
binary_df = pd.DataFrame(X_bin, columns=names[0:8])
binary_df


get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms plots
import matplotlib.pyplot as plt
import seaborn as sns
f, axes = plt.subplots(3, 3, figsize=(14, 14))
sns.distplot(binary_df["preg"], ax=axes[0, 0])
sns.distplot(binary_df["plas"], ax=axes[0, 1])
sns.distplot(binary_df["pres"], ax=axes[0, 2])
sns.distplot(binary_df["skin"], ax=axes[1, 0])
sns.distplot(binary_df["test"], ax=axes[1, 1])
sns.distplot(binary_df["mass"], ax=axes[1, 2])
sns.distplot(binary_df["pedi"], ax=axes[2, 0])
sns.distplot(binary_df["age"], ax=axes[2, 1])


# BOX-COX
# Useful when an attribute is skewed, which is when an attribute has a Gaussian-like distribution but it is shifted.
# For this type of case, the Box-Cox transformation is used, which assumes that all values are positive.
# It is only applied in attributes where it is necessary due to the presence of bias.

# Univariate Histograms to see the skew in pedi and age
plt.figure(1)
plt.subplots(figsize=(20,20))
plt.subplot(421)
sns.distplot(data['pedi'])
plt.title('Pedigree')
plt.grid(True)
plt.subplot(422)
sns.distplot(data['age'])
plt.title('Age')
plt.grid(True)
plt.show()

# Box-Cox Transform
from sklearn.preprocessing import PowerTransformer
# extract features with skew
features=data[["pedi","age"]]

#instatiate 
pt=PowerTransformer(method="box-cox",standardize=True)
#Fit the data to the powertransformer
skl_boxcox=pt.fit(features)

#Lets get the Lambdas that were found
calc_lambdas_bc=skl_boxcox.lambdas_
print (calc_lambdas_bc)

#Transform the data 
skl_boxcox=pt.transform(features)

#Pass the transformed data into a new dataframe 
df_features=pd.DataFrame(data=skl_boxcox,columns=["pedi","age"])
df_features

# Pass to the original dataframe the transform columns
data.drop(["pedi","age"],axis=1,inplace=True)
#Dataframe concat
df_data=pd.concat([data,df_features],axis=1)
cols=df_data.columns.tolist()
# Left "class" in the last column
cols=cols[-1:]+cols[:-1]
cols=cols[-1:]+cols[:-1]
df_data=df_data[cols]
df_data


# Univariate Histograms, plotting again
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.figure(1)
plt.subplots(figsize=(20,20))
plt.subplot(421)
sns.distplot(df_data['pedi'])
plt.title('Pedigree')
plt.grid(True)
plt.subplot(422)
sns.distplot(df_data['age'])
plt.title('Age')
plt.grid(True)
plt.show()


#Yeo Johnson Transform
# It is similar to Box-Cox, but supports raw values equal to zero or negative.
from sklearn.preprocessing import PowerTransformer
# extract features with skew
features=data[["pedi","age"]]
#instatiate 
pt= PowerTransformer(method="yeo-johnson", standardize=True)
#Fit the data to the powertransformer
skl_yeo=pt.fit(features)
#Lets get the Lambdas that were found
calc_lambdas_yo=skl_yeo.lambdas_
print (calc_lambdas_yo)

#Transform the data 
skl_yeo=pt.transform(features)
#Pass the transformed data into a new dataframe 
df_features=pd.DataFrame(data=skl_yeo, columns=["pedi","age"])
# Pass to the original dataframe the transform columns
data.drop(["pedi","age"],axis=1,inplace=True)
# Dataframe Concat
df_data=pd.concat([data,df_features],axis=1)
cols=df_data.columns.tolist()
# Pasa e útimo elemento al primero de la lista (2 veces)
cols=cols[-1:]+cols[:-1]
cols=cols[-1:]+cols[:-1]
# Sobreescrimos
df_data=df_data[cols]
df_data

get_ipython().run_line_magic('matplotlib', 'inline')
# Univariate Histograms plotting
plt.figure(1)
plt.subplots(figsize=(20,20))
plt.subplot(421)
sns.distplot(df_data['pedi'])
plt.title('Pedigree')
plt.grid(True)
plt.subplot(422)
sns.distplot(df_data['age'])
plt.title('Age')
plt.grid(True)
plt.show()

print(f"Lambda de Box-Cox: {calc_lambdas_bc}")
print(f"Lambda de Yeo-Johnson: {calc_lambdas_yo}")

# Coeficiente Box-Cox y Yeo-Johnson

# These transformations arise because in certain occasions a logarithmic or square root transformation
# is used to the data but without knowing very well which of the two will have a better incidence.
# These two transformations perform that analysis and establish the best transformation.

# The `PowerTransform` has a property called `lambdas_`, which is used to control the nature of the transformation.
# Here are some common values for _lambda:_
# * _lambda = -1_ is a reciprocal transformation.
# * _lambda = -0.5_ is a reciprocal square root transformation.
# * _lambda = 0.0_ is a logarithmic transformation.
# * _lambda = 0.5_ is a square root transformation.
# * _lambda = 1.0_ is not a transformation.

# Depending on the value of the lambda, one or the other transformation applies.



