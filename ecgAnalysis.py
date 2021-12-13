#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Torin Stott, Seth Spillers
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
######################################################
# DATA PREPROCESSING
######################################################
missing_val_format = ["?"]
# File contains trailing comma, this line fixes that 
df = pd.read_csv('echocardiogram.data', error_bad_lines=False, na_values=missing_val_format)
# file reading is a little messy but column names are provided so we specify them here
df.columns = ['survival', 'still-alive', 'age-at-heart-attack', 'pericardial-effusion',
                'fractional-shortening', 'epss', 'lvdd', 'wall-motion-score', 'wall-motion-index',
                'mult', 'name', 'group', 'alive-at-1']
# Drop all unnecessary columns
df = df.drop(['name'], axis=1)
df = df.drop(['group'], axis=1)
df = df.drop(['mult'], axis=1)
# not all columns have values filled here we replace those
# with a median value from same column
df = df.fillna(df.mean())
dt = df.values
######################################################
# END OF DATA PREPROCESSING
######################################################

def runKNN():
    X = dt[:,:9]
    y = dt[:,9].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # KNN PREDICTION
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(X_train, y_train)
    y_pred = cross_val_predict(classifier, X, y, cv=5)
    # print(accuracy_score(y,y_pred) * 100)
    print('Mean Squared error is: ', mean_squared_error(y,y_pred)) 

def findBestK():
    X = dt[:,:9]
    y = dt[:,9].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # FINDING OPTIMAL K VALUE
    error = []
    for k in range(1,51):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, X, y, cv=5)
        error.append(mean_squared_error(y,y_pred))
    plt.title('K_neighbor plot')
    plt.plot(range(1,51), error)

def runPerceptron():
    X = dt[:,:9]
    y = dt[:,9].astype(np.int32)
    # PERCEPTRON
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)
    sc = StandardScaler()
    sc.fit(X_train)
    std_x_train = sc.transform(X_train)
    std_x_test = sc.transform(X_test)
    iterations = 10
    learningRate = .1
    # create model
    percep = Perceptron(n_iter_no_change=iterations, eta0=learningRate, random_state=1)
    # fit
    percep.fit(std_x_train, y_train)
    y_vals = percep.predict(std_x_test)
    print('Mean Squared error is: ', mean_squared_error(y_test,y_vals)) 
    
    # Plot decision boundary
    pca = PCA(n_components = 2)
    X_train2 = pca.fit_transform(X_train)
    percep.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train, clf=percep, legend=2)
    plt.title("Perceptron Classifier")
    plt.show()
    

def runSVM():
    X = dt[:,:9]
    y = dt[:,9].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # Create the SVM
    svm = SVC(random_state=42, kernel='rbf')
    pca = PCA(n_components = 2)
    X_train2 = pca.fit_transform(X_train)

    # Fit the data to the SVM classifier
    svm = svm.fit(X_train, y_train)

    # Generate predictions
    y_pred = svm.predict(X_test)

    # Evaluate by means of accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')
    print('Mean Squared error is: ', mean_squared_error(y_test,y_pred))
    # Plot decision boundary
    svm.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train, clf=svm, legend=2)
    plt.show()
    X_folds = np.array_split(X, 5)
    y_folds = np.array_split(y, 5)
    scores = list()
    for k in range(5):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(svm.fit(X_train, y_train).score(X_test, y_test))
    print('SVM scores: ' , scores)

# runKNN()






