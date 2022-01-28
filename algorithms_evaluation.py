
# Clasification problem
import numpy as np
import pandas as pd
filename_clas = 'data/pima-indians-diabetes.data.csv'
names_clas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
df_clas = pd.read_csv(filename_clas, names=names_clas)

#X_clas = df_clas.drop(["class"],axis=1).values 
#Y_clas = df_clas["class"].values
array_clas = df_clas.values
X_clas = array_clas[:,0:8]
Y_clas = array_clas[:,8]
Y_clas.shape

# Cross Validation Classification Accuracy
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score #their own metrics
from sklearn.linear_model import LogisticRegression

kfold= KFold (n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(solver="lbfgs", max_iter=1000)
results = cross_val_score (model, X_clas, Y_clas, cv=kfold, scoring="accuracy")
print(f"Accuracy: {results.mean()*100.0:,.2f}% ({results.std()*100.0:,.2f})")

df_clas.groupby("class").size()
#we can also call accuracy_score from sklearn.metrics


# Train test split Validation Classification Kappa coef
#accuracy when we have unbalance 30-70% between class
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=0.33, random_state=7)

model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train, Y_train)
predicted=model.predict(X_test)
cohen_score=cohen_kappa_score(Y_test,predicted)
print(f"Cohen score: {cohen_score*100.0:,.2f}%")


# Cross Validation Classification ROC AUC
#Metric for binary classification problems.
#An area of 1.0 represents a model that made all predictions perfectly.
# An area of 0.5 represents a model that is as good as random.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score #tiene métricas por defecto
from sklearn.linear_model import LogisticRegression

kfold= KFold (n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(solver="lbfgs", max_iter=1000)
results = cross_val_score (model, X_clas, Y_clas, cv=kfold, scoring="roc_auc")
print(f"AUC: {results.mean():,.2f} ({results.std()})")

#Note for the Shei of the future: you could include the line plot

# Train test split Validation Classification Confusion Matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=0.33, random_state=7)

model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train, Y_train)
predicted=model.predict(X_test)
conf_matrix = confusion_matrix(Y_test,predicted)
print(conf_matrix)

#Note for the Shei of the future: you could include the confusion matrix pretty

# <a id="section25"></a>
# ## <font color="#004D7F"> 2.5. Reporte de clasificación</font>

a función `clasification_report()` muestra _precisión, recall, F1-score_ y el soporte para cada clase.
#

#Train test split Validation Classification Report
#shows precission, recall, f1_score and support per class and overall
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=0.33, random_state=7)

model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train, Y_train)
predicted=model.predict(X_test)
report = classification_report(Y_test,predicted)
print(report)




# Regression problem
import pandas as pd
filename_reg = 'data/housing.csv'
names_reg = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
df_reg = pd.read_csv(filename_reg, delim_whitespace=True, names=names_reg)

X_reg = df_reg.drop(["MEDV"],axis=1).values
Y_reg = df_reg["MEDV"].values
#array_reg = df_reg.values
#X_reg = array_reg[:,0:13]
#Y_reg = array_reg[:,13]
Y_reg.shape


#Train Test split, MAE logistic Regression
#the lower the better
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=0.33, random_state=7)
model = LinearRegression()
model.fit(X_train, Y_train)
yhat= model.predict(X_test)
MAE = mean_absolute_error(Y_test,yhat)
print (MAE)

# Cross Validation Regression MAE
from sklearn.model_selection import cross_val_score, KFold

#Negative MAE
kfold= KFold (n_splits=10, shuffle=True, random_state=7)
model= LinearRegression()
scoring= "neg_mean_absolute_error"
results=cross_val_score(model,X_reg,Y_reg, cv=kfold, scoring=scoring)
print(f"Neg MAE: {results.mean():,.2f}({results.std():,.2f})")

#Train test split MSE Regression problem
#the lower the better
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=0.33, random_state=7)
model = LinearRegression()
model.fit(X_train, Y_train)
yhat= model.predict(X_test)
MSE = mean_squared_error(Y_test,yhat)
print (MSE)

# Cross Validation Regression MSE
from sklearn.model_selection import cross_val_score, KFold


#Negative MAE
kfold= KFold (n_splits=10, shuffle=True, random_state=7)
model= LinearRegression()
scoring= "neg_mean_squared_error"
results=cross_val_score(model,X_reg,Y_reg, cv=kfold, scoring=scoring)
print(f"Neg MSE: {results.mean():,.2f}({results.std():,.2f})")

#Train test split R2_score Regresion model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#closer to 1, the better
X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=0.33, random_state=7)
model = LinearRegression()
model.fit(X_train, Y_train)
yhat= model.predict(X_test)
r2 = r2_score(Y_test,yhat)
print (r2)

# Cross Validation Regression MSE
from sklearn.model_selection import cross_val_score, KFold

#r2_SCORE
kfold= KFold (n_splits=10, shuffle=True, random_state=7)
model= LinearRegression()
scoring= "r2"
results=cross_val_score(model,X_reg,Y_reg, cv=kfold, scoring=scoring)
print(f"r2: {results.mean():,.3f}({results.std():,.3f})")