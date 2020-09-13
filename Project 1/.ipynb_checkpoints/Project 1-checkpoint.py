# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# #Project 1

# %%
#Import libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# %%
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
#Import data
df = pd.read_csv("Avila-DataSet for miniprojects.csv")
df.head()


# %%
df.columns


# %%
df.info()


# %%
df.describe()


# %%
#Prep data for modeling
x = df.drop("Class: A, B, C, D, E, F, G, H, I, W, X, Y", axis = 1)
y = df["Class: A, B, C, D, E, F, G, H, I, W, X, Y"]
y


# %%
#Model the data
model = GaussianNB()
model.fit(x,y)
yhat = model.predict(x)


# %%
#Evaluate the model
metrics.accuracy_score(y,yhat)


# %%
print(metrics.classification_report(y,yhat))


# %%
plot_confusion_matrix(model,x,y)


# %%
# 80 / 20 Split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 520)


# %%
model_split = GaussianNB()
model_split.fit(x_train,y_train)
yhat_split = model.predict(x_test)


# %%
#Evaluate the model
metrics.accuracy_score(y_test,yhat_split)


# %%
print(metrics.classification_report(y_test,yhat_split))


# %%
#Cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=520)

for train, test in kfold.split(x,y):
    model_cv = GaussianNB()
    model.fit(x.iloc[train], y.iloc[train])
    y_hat_cv = model.predict(x.iloc[test])
    probV = metrics.accuracy_score(y.iloc[test],y_hat_cv)
    probV


# %%
#Evaluate model
metrics.accuracy_score(y.iloc[test],y_hat_cv)


