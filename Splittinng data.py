# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:50:06 2020

@author: Edgar.Lizarraga
"""

import pandas as pd
import numpy as np
heart_desease = pd.read_csv(r'C:\Users\Edgar.Lizarraga\Desktop\ML training\zero-to-mastery-ml-master\data\heart-disease.csv')


# we need to find now the values for x and y

X = heart_desease.drop("target", axis=1)
y = heart_desease['target']


from sklearn.model_selection import train_test_split

#need to test and train, x and y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X_train.shape

len(X_train.columns)



# =============================================================================
# We need to clean transform and reduce data, using car data files
# =============================================================================

car_sales = pd.read_csv(r'C:\Users\Edgar.Lizarraga\Desktop\ML\car-sales-extended.csv')

X = car_sales.drop("Price", axis=1)
y = car_sales['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Buil a machine learning model
#when predicting a number we can use from sklearn ensemble  and randomforestregressor

from sklearn.ensemble import RandomForestRegressor
#this will turn into error because data still with strings
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)

#using sklearn to convert strings into numbers

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#we need to categorize the data
categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                    remainder = "passthrough")

#we weill trabnsfor the test data

transformed_x = transformer.fit_transform(X)
transformed_x

pd.DataFrame(transformed_x)


#re fit the model

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_x,
                                                    y, test_size=0.2)

model.fit(X_train, y_train)

































    