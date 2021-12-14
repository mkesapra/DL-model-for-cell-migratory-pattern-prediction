#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:57:59 2021

@author: kesaprm


To check which model gives best accuracy: Deep learning or Random Classifier
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras.utils import normalize, to_categorical
from sklearn.utils import shuffle

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout, Flatten


#to convert a string to a list
#from ast import literal_eval

# #k-fold cross validation using sklearn
# kf = KFold(n_splits=5, shuffle=True, random_state=2652124)
# k_trn =[]; k_tst =[];
# for trn, tst in kf.split(clusters.traj,clusters.patternNo):
#     print("%s %s" % (trn, tst))
#     k_trn.append(trn)
#     k_tst.append(tst)

dfData = pd.read_pickle("clusters.txt") 
dfData_val = pd.read_pickle("clusters_validation.txt") 


#adding 4 ppoints each between 2 frames for every cell to make the no. of frames = 180, i.e. 3 hours
fr_36_data = dfData.traj #fr_36_data[0] is x and fr_36_data[1] is y
fl = [] #final array after adding all the 4 new elements between 2 frames. Total 180 xl and 180 yl

for i in range(len(fr_36_data)):
    xl = [] # temporary array to hold x values
    yl = [] # temporary array to hold y values
    for j in range(len(fr_36_data[i][0]) - 1): # no. of frames
        xl.append(fr_36_data[i][0][j]) # to take the 1st value of an array
        yl.append(fr_36_data[i][1][j])
        for r in range(4) :
            xl.append(xl[-1] + (fr_36_data[i][0][j+1] - fr_36_data[i][0][j]) / 5) 
            yl.append(yl[-1] + (fr_36_data[i][1][j+1] - fr_36_data[i][1][j]) / 5)
    fl.append([xl,yl])

dfData['training_traj'] = fl # data augmented 

# split the 240 frames data to 180-training(3hrs) and 60-testing(last 1 hr)
fr_240_data = dfData_val.traj
training_180_data = []
testing_60_data = []
for i in range(len(fr_240_data)):
    xl_180 = []; xl_60 = [];
    yl_180 = []; yl_60 = [];
    for j in range(len(fr_240_data[i][0]) - 1):
        xl_180 = fr_240_data[i][0][:180]
        yl_180 = fr_240_data[i][1][:180]
        xl_60 = fr_240_data[i][0][181:]
        yl_60 = fr_240_data[i][1][181:]
    training_180_data.append([xl_180,yl_180])
    testing_60_data.append([xl_60,yl_60]) 
    
dfData_val['training_traj'] = training_180_data
dfData_val['testing_traj'] = testing_60_data


# Create Train data from 36 frames data+ 240 frames data. Total number of cells are  97 + 23 = 120, with 180 datapoints(first 3 hours) each
train_36_arr = np.array(fl)
train_240_arr = np.array(training_180_data)

train = np.concatenate((train_36_arr,train_240_arr))

# Train labels

train_36_labels = np.array(dfData.patternNo)
train_240_labels = np.array(dfData_val.patternNo)

train_240_labels[train_240_labels == 9.] = 4.

train_labels =  np.concatenate((train_36_labels,train_240_labels))

# Vectorize data 1st cell, 2nd x,y ,3rd timeframe  -- This part is not working
# def vectorize_sequences(sequences, dimension=180,features = 2):
#     results = np.zeros((len(sequences), features, dimension))
#     for j in range(0,features-1):
#         for i, sequence in enumerate(sequences):
#             results[i, j, sequence[j][i]] = 1.
#     return results


# x_train = vectorize_sequences(train)

# x_train = np.vectorize(train)

#Normalize each feature to have a mean of 0 and to have a standard deviation of 1.
train -= train.mean(axis=0)
train /= train.std(axis=0)

#train,test, train_labels,test_labels =  X[train], X[test], y[train], y[test]
#train, test = train_test_split(data, test_size=0.2) # split 80:20 train:test 
#train,test, train_labels,test_labels = train_test_split(clusters.traj,clusters.patternNo, shuffle=True, random_state=2652124)

#shuffle the training data to improve the val accuracy scoere -- Used instead of #np.random.shuffle(rank_3_tensor_train)
train, train_labels = shuffle(train, train_labels)


from sklearn.model_selection import train_test_split
x_grid, x_not_use, y_grid, y_not_use = train_test_split(train, train_labels, test_size=0.2, random_state=42)

x_grid = np.expand_dims(x_grid, axis=3)

def feature_extractor():       
    activation = 'sigmoid'
    feature = Sequential()
    
    feature.add(Conv2D(6, 3, activation = activation, padding = 'same', input_shape = (2,180, 1)))
    feature.add(BatchNormalization())
    
    feature.add(Conv2D(6, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
    feature.add(BatchNormalization())
    feature.add(MaxPooling2D())
    
    feature.add(Flatten())
    
    return feature

feature_extractor = feature_extractor()
print(feature_extractor.summary())



X_for_RF = feature_extractor.predict(x_grid) #This is our X input to RF and other models


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
RF_model.fit(X_for_RF, y_grid) #For sklearn no one hot encoding

X_test_feature = feature_extractor.predict(np.expand_dims(x_not_use, axis=3))


#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_feature)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_not_use, prediction_RF))


#Confusion Matrix - verify accuracy of each class
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_not_use, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)


from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1],  #Regularization parameter. Providing only two as SVM is slow
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [10,20,30]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1]  #Regularization. . Providing only two as LR can be slow
        }
    }
}


scores = []

from sklearn.model_selection import GridSearchCV

for model_name, mp in model_params.items():
    grid =  GridSearchCV(estimator=mp['model'], 
                         param_grid=mp['params'], 
                         cv=5, n_jobs=16, 
                         return_train_score=False)
    
    grid.fit(X_for_RF, y_grid)
    
    scores.append({
        'model': model_name,
        'best_score': grid.best_score_,
        'best_params': grid.best_params_
    })

import pandas as pd    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df)






