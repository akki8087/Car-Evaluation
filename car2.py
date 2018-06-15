# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 22:19:13 2018

@author: NP
"""


import pandas as pd
import numpy as np
import math
data = pd.read_csv('car.data.txt',names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','class'])

col = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
X = data[col]
y = data['class']
from sklearn.model_selection import train_test_split

X_d = pd.get_dummies(X,drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X_d, y, test_size = 0.2, random_state=1)

'''
#Xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
classifier = XGBClassifier(learning_rate =0.01, n_estimators=15)
'''
'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
'''

# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',random_state = 0)

'''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
'''
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#print(classifier.feature_importances_)
print(classifier.score(X_test,y_test))

