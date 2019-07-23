# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:46:38 2019

@author: 17pd29
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
df.columns=['A','B','C','D','E','F','G','H','I','J','K']
l=pd.get_dummies(df['K'])

df['Malign']=list(l[4])
df['Benign']=list(l[2])
df=df.drop(['K'],axis=1)
df=df.replace('?',np.nan)
df.dropna(inplace=True)
df['G']=df['G'].astype(int)
X = np.asarray(df.iloc[:,1:10])
y = np.asarray(df.iloc[:, 11])

cv = KFold(n_splits=5)
log = LogisticRegression(random_state=0, solver='lbfgs')

for train_index, test_index in cv.split(X):
#    print("Train Index: ", train_index)
#    print("Test Index: ", test_index)
#   
    y_train, y_test = y[train_index], y[test_index]
#    print(y_train)
#    print(y_test)
    X_train, X_test = X[train_index], X[test_index]
    model = log.fit(X_train, y_train)
#    for i in range(len(X_test)):
#        print('prediction: ',int(model.predict([X_test[i]])),'actual: ',y_test[i])
    print(model.score(X,y))