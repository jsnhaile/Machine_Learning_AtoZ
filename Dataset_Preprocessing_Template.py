# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:14:07 2017

@author: 581881
"""

#this is the data preprocessing template
"""things included:
    -importing the libraries
    -importing the dataset
    -creation of the independent variable matrix
    -creation of the depended variable set
    -splitting the data into training and splitting
    
    things removed include (refer to 'Data_Preprocessing Unit - Udemy.py' for code for these:
        -importing impute to remove missing data (can use later)
        -encoding categorical data (a lot of the times we won't be dealing with categorical data)
        -feature scaling (other machine learning models include this as a process - commented out)
        """
#import the libraries
import numpy as np #import numpy for stats
import pandas as pd #import pandas for EDA
import matplotlib.pyplot as plt #import matplotlib for visualizations

#import the dataset
dataset = pd.read_csv('Data.csv')

#Create feature matrix and dependent variable set 
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

#splitting the data using the train test split
#import the train_test_splt
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""