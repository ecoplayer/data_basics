#Dataset Load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filename = "data/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
print(data.shape)

# separate array into input and output components
X = data.drop(["class"],axis=1).values 
Y = data["class"].values


# Evaluate using K_FOLD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 

num_folds=10
seed=7
kfold=KFold(n_splits=num_folds, random_state=seed,shuffle=True)
model=LogisticRegression(solver="lbfgs", max_iter=1000)

results=cross_val_score(model,X,Y,cv=kfold)
print(f"Accuracy: {results.mean()*100.0:,.2f}% ({results.std()*100.0:,.2f}%)")


# Evaluate using Repeated KFOLD
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 

num_folds=10
seed=7
num_repeated=5
rep_kfold=RepeatedKFold(n_splits=num_folds, random_state=seed,n_repeats=num_repeated)
model=LogisticRegression(solver="lbfgs", max_iter=1000)

results=cross_val_score(model,X,Y,cv=rep_kfold)
print(f"Accuracy: {results.mean()*100.0:,.2f}% ({results.std()*100.0:,.2f}%)")

# Evaluate using Leave One Out Cross Validation
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 

loocv=LeaveOneOut()
model=LogisticRegression(solver="lbfgs", max_iter=1000)
results=cross_val_score(model,X,Y,cv=loocv)
print(f"Accuracy: {results.mean()*100.0:,.2f}% ({results.std()*100.0:,.2f}%)")


# Evaluate using a train trest splitting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

test_size=0.33
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
model=LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train,Y_train)
results= model.score(X_test,Y_test)
y_hat=model.predict(X_test)
print(f"Accuracy: {results*100.0:,.2f}% ({results*100.0:,.2f}%)")

print(confusion_matrix(Y_test, y_hat))


# Evaluate using Shuffle Split Cross Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

test_size=0.33
seed=7
n_splits=10

Kfold=ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=seed)
model=LogisticRegression(solver="lbfgs", max_iter=1000)
reults=cross_val_score(model, X,Y, cv= kfold)

print(f"Accuracy: {results.mean()*100.0:,.2f}% ({results.std()*100.0:,.2f}%)")