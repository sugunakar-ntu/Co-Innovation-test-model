# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:18:23 2021

@author: abhayaradhya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:53:59 2021

@author: abhayaradhya
"""

import pandas as pd
import sklearn
from sklearn import svm,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

# create dataframe of the data
df = pd.read_csv('train.csv')

# separate features and labels
labels = df.iloc[:,-1]
features = df.iloc[:,1:-1]
min_max_scaler = preprocessing.MinMaxScaler()
norm_features = min_max_scaler.fit_transform(features)

# create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(norm_features, labels, test_size=0.2, random_state=0)


def buildSVMclassifier(X_train, X_test, y_train, y_test):    
    # build SVM model
    SVM = svm.LinearSVC()
    SVM.fit(X_train, y_train)
    
    # save model to pickle
    pickle.dump(SVM,open('model.sav','wb'))
    
    
buildSVMclassifier(X_train, X_test, y_train, y_test)


