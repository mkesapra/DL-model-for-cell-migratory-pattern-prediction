#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:10:36 2021

@author: kesaprm

This method uses Neural network based approach to genrate features and feeds these features to random forest classifier.


"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from sklearn.utils import shuffle

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import seaborn as sns

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

test_60_arr = np.array(testing_60_data)
test_labels = train_240_labels

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


#Reassign to new names test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train, train_labels_encoded, test_60_arr, test_labels_encoded

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


#############################
x_train = np.expand_dims(x_train, axis=3)



activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 2, activation = activation, padding = 'same', input_shape = (2,180,1)))
feature_extractor.add(BatchNormalization())

# feature_extractor.add(Conv2D(32, 2, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(BatchNormalization())
# feature_extractor.add(MaxPooling2D())

# feature_extractor.add(Conv2D(32, 2, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(BatchNormalization())


# feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(BatchNormalization())
# feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(2, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(3, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 


##########################################
#Train the CNN model
#Not using the test data
#splitting the training to validation
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

x_train,y_train, y_train_one_hot,y_train_labels_shuffled = shuffle(x_train,y_train, y_train_one_hot,train_labels)


x_train = x_train[20:]
y_train = y_train[20:]
y_train_one_hot = y_train_one_hot[20:]
x_test = x_train[:20]
y_test = y_train_labels_shuffled[:20]
y_test_one_hot = y_train_one_hot[:20]


history = cnn_model.fit(x_train, y_train_one_hot, epochs=50, validation_data = (x_test, y_test_one_hot))

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def smooth_curve(points, factor=0.9): 
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else: 
            smoothed_points.append(point)
    return smoothed_points

smooth_loss_hist = smooth_curve(loss)
smooth_val_loss_hist = smooth_curve(val_loss)
plt.plot(range(1, len(smooth_loss_hist) + 1), smooth_loss_hist, 'c-', label='Training loss')
plt.plot(range(1, len(smooth_val_loss_hist) + 1), smooth_val_loss_hist, 'purple', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


smooth_acc_hist = smooth_curve(acc)
smooth_val_acc_hist = smooth_curve(val_acc)
plt.plot(range(1, len(smooth_acc_hist) + 1), smooth_acc_hist, 'c-', label='Training acc')
plt.plot(range(1, len(smooth_val_acc_hist) + 1), smooth_val_acc_hist, 'purple', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#de-encode to normal labels
prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)

################################
#Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(x_train) #This is out X input to RF

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 1000, random_state = 42)

# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding

#Send test data through same feature extractor process
X_test_feature = feature_extractor.predict(x_test)

prediction_RF = RF_model.predict(X_test_feature)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_RF)*100)


from sklearn.metrics import classification_report

