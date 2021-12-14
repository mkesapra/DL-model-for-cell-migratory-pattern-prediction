#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:40:53 2021

@author: kesaprm

Divided the data as in Abhijeet's excel as FrameNo, Cell No, x,    y,  pattern
                                            1,      1,       2.3, 3.4, Spinning
                                            2,      1,       3.5, 6.8, Spinning
                                            .,      1,       4.6, 5.5, Spinning
                                            .,      1,       3.8, 6.8, Spinning
                                          180,      1,       2.7, 3.6, Spinning
                                          
Got 100% accuracy. This method works in the case when x and y values are not depenedent on their previous values.
This method considers each row as a different independent datapoint, which does not work in our case.
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


dfData = pd.read_pickle("clusters.txt") 
dfData_val = pd.read_pickle("clusters_validation.txt") 

dfData = dfData[dfData.pattern != 'Mix']


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

train_36_labels = np.array(dfData.pattern)
train_240_labels = np.array(dfData_val.pattern)

#train_240_labels[train_240_labels == 9.] = 4.

train_labels =  np.concatenate((train_36_labels,train_240_labels))


#test

# test_60_arr = np.array(testing_60_data)
# test_labels = train_240_labels

# create a training dataframe
#create lists back to insert to dTframe



d1 = {'traj':dfData.training_traj, 'pattern': dfData.pattern}
df_train1 = pd.DataFrame(data=d1)

d2 = {'traj':dfData_val.training_traj, 'pattern': dfData_val.pattern}
df_train2 = pd.DataFrame(data=d2)



df_train = pd.concat([df_train1,df_train2], ignore_index=True)


#df = df.fillna(0)

dictD = {'CellNo': [], 'x': [], 'y': [], 'pattern': []}


for k in range(len(df_train.traj)-1):
        for i in range(len(df_train.traj[0][0])-1):
            dictD['CellNo'].append(k)
            dictD['x'].append(df_train.traj[k][0][i])
            dictD['y'].append(df_train.traj[k][1][i])
            dictD['pattern'].append(df_train.pattern[k])
          
df = pd.DataFrame(data = dictD)

# sizes = df['pattern'].value_counts(sort=1)
# print(sizes)

#STEP 4: Convert non-numeric to numeric, if needed.
#Sometimes we may have non-numeric data, for example batch name, user name, city name, etc.
#e.g. if data is in the form of YES and NO then convert to 1 and 2

df.pattern[df.pattern == 'Wandering'] = 1
df.pattern[df.pattern == 'Spinning'] = 3
df.pattern[df.pattern == 'Bidirectional'] = 2
print(df.head())


#Y is the data with dependent variable, this is the Productivity column
Y = df["pattern"].values  #At this point Y is an object not of type int
#Convert Y to int
Y=Y.astype('int')

#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ["pattern"], axis=1)  
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
# Train the model on training data
model.fit(X_train, y_train)

prediction_test = model.predict(X_test)


from sklearn import metrics
#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)