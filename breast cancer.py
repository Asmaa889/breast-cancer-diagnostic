# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:10:56 2021

@author: GM
"""

############### """bad modeling""" ################
"""https://www.kaggle.com/aryantiwari123/breast-cancer-eda-and-prediction-98/notebook#1.-Logistic-Regression"""
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
data = pd.read_csv("E:\course\python for machine\projects\\breast cancer diagnostic\\breast.csv") 
#print(data.isnull().sum())
data.drop('Unnamed: 32', axis = 1, inplace = True) #cleaning data
print(data)
x = data.drop(columns = 'diagnosis')
y = data['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2)

# """choosing the model"""

model = LogisticRegression()
model.fit(X_train,Y_train)
""" accuracy"""
accuracy = accuracy_score(Y_test, model.predict(X_test))
print(accuracy)