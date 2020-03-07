# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:08:27 2020

@author: Edgar.Lizarraga
"""

# =============================================================================
# Intro to Scikit - Learn (sklarn)
# 
# Steps::
#     
# 0. And end to end scikit learn workflow
# 1. Getting the data ready
# 2. Choose the right estimator/algorithm for our problems
# 3. Fit the model/algo and use it to make predictions on our data
# 4. evaulating a model
# 5. Improve a model
# 6. Save amd load a trained model
# 7. Put all together
# =============================================================================

# =============================================================================
# 0. End to End
# =============================================================================


import pandas as pd
import numpy as np
import pickle

heart_desease = pd.read_csv(r'C:\Users\Edgar.Lizarraga\Desktop\ML training\zero-to-mastery-ml-master\data\heart-disease.csv')
heart_desease.head()

#we first need to find x and y values: for x;;

x = heart_desease.drop("target", axis=1)

#we now need to create Y, TARGET COLUMN
y = heart_desease['target']

#this is a classification algorithm because we want to know if somebody has a heart desease or not.
#import classification ML model named RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

#which parameters clf contain?

clf.get_params()

# =============================================================================
# 3. Fit the model to the training data
# =============================================================================
from sklearn.model_selection import train_test_split

#me need to split the data into 2 paths, training and testing

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf.fit(X_train, y_train)

#Make a prediction and needs to be at the Y level

y_label = clf.predict(np.array([0,2,3,4]))

print(X_train)

y_preds = clf.predict(x_test)

# =============================================================================
# 4. Evaluate the model on the training data and test data
# =============================================================================

clf.score(X_train, y_train)
clf.score(x_test, y_test)

#80% accuracy, we can improve the model

# =============================================================================
# 5. Improve the model
#     Try different amount of n_estimatros to find the best fit
# =============================================================================

np.random.seed(42)

for i in range(10,100,10):
    print(f"Trying model with {i} estimators.....")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) * 100:.2f}%")
    print("")
    
# =============================================================================
# 6. Save the model & load it
# =============================================================================

pickle.dump(clf, open("random_forst_model_1.pk1", "wb"))

loaded_model = pickle.load(open("random_forst_model_1.pk1", "rb"))
loaded_model.score(x_test, y_test)










