#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:21:27 2021

@author: kesaprm

This file contains:
    
    Data & Labels
   =============== 
    1) data is clusters.traj is a column with values of x,y as ([x1,x2,x3..x36],[y1,y2,y3,..y36])
    2) labels are clusters.patternNo with distinct number for each cluster
    
    Pre-processing
   ================
   1) Split the data into training and testing using sklearn.model_selection train_test_split with shuffle and randomstate(to have the same test values)
   2) Used sklearn shuffle to shuffle the training data, so that it could be used while splitting validation data. This has improved val accuracy
   3) Converted the train/test data into tensors and normalized them. Train/test data are in the shape of TensorShape([72, 2, 37])/ TensorShape([25, 2, 37])
   4) Converted the train/test labels using one-hot coding. Train/test labels are in the shape of (72, 2, 46)/(25, 2, 46) 
   5) 46 in the last arg of train/test labels is the number of neurons in the output layer of the model
    
    Build Model
   =============
   1) Used keras model with 2 dense-64 neuron layers with relu activation
   2) The output layer is 1 dense-46 neuron layer with softmax activation
   3) Compiled the model with optimizer-rmsprop, loss-categorical_crossentropy and metrics-accuracy
    
    Training - Validation Split
   =============================
   1) Shuffled the training data in Pre-processing step-2
   2) Used k-fold cross validation with k=5 to split the training and validation data
    
    Fit Model
   ===========
   1) Number of Epochs taken = 500
   2) Fit model on validation data to check the loss and accuracy scores and tune the params accordingly.
    
    
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras.utils import normalize, to_categorical
from sklearn.utils import shuffle

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



# tensor creation
rank_3_tensor_train = tf.constant([train,])
rank_3_tensor_train = tf.reshape(rank_3_tensor_train,[rank_3_tensor_train.shape[1], rank_3_tensor_train.shape[2], rank_3_tensor_train.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# rank_3_tensor_test = tf.constant([test,])
# rank_3_tensor_test = tf.reshape(rank_3_tensor_test,[rank_3_tensor_test.shape[1], rank_3_tensor_test.shape[2], rank_3_tensor_test.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

#Not required to create tensors for labels, we use one-hot encode
# train_labels = tf.constant([train_labels,])
# train_labels = tf.reshape(train_labels,[train_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# test_labels = tf.constant([test_labels,])
# test_labels = tf.reshape(test_labels,[test_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# normalizing and using one hot encode
#rank_3_tensor_train = normalize(rank_3_tensor_train)
#rank_3_tensor_test = normalize(rank_3_tensor_test)

#to one hot labels
def to_one_hot(labels,dimension=5,features=2):
    results = np.zeros((len(labels),features, dimension))
    for j in range(1,features):        
        for i, label in enumerate(labels):           
            label = int(label)
            results[i, j, label] = 1.
    return results

#labels train and test
hot_train_labels = to_one_hot(train_labels)
#hot_test_labels = to_one_hot(test_labels)


# #labels train and test using to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler


epochs=10
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8   
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu',kernel_initializer='he_uniform',  input_shape=(2, 180)))
    model.add(layers.Flatten())
    #model.add(layers.Dense(4, kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    return model
    
## Validating the approach by taking out 20 datapoints for validation -- Manual split of the training & validation data
# x_val = rank_3_tensor_train[:20]
# partial_x_train = rank_3_tensor_train[20:]

# y_val = train_labels[:20]
# partial_y_train = train_labels[20:]

# #k-fold cross validation from the text book
k = 5
num_val_samples = len(rank_3_tensor_train)//k
num_epochs = 10
all_scores = []
all_loss_histories = [];all_acc_histories = []; val_loss_histories =[];val_acc_histories =[]
#np.random.shuffle(rank_3_tensor_train)

validation_scores = []



def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate


lr_rate = LearningRateScheduler(exp_decay)
callbacks_list = [lr_rate]

for fold in range(k):
    #print('processing fold #', i)
    val_data = rank_3_tensor_train[num_val_samples * fold: num_val_samples * (fold+1)] # selects the validation-data partition
    val_labels = hot_train_labels[num_val_samples * fold: num_val_samples * (fold+1)] 
    
    partial_train_data = np.concatenate( [rank_3_tensor_train[:fold * num_val_samples],rank_3_tensor_train[(fold + 1) * num_val_samples:]], axis=0)
    partial_train_labels = np.concatenate( [hot_train_labels[:fold * num_val_samples], hot_train_labels[(fold + 1) * num_val_samples:]], axis=0)
    
    model =  build_model() # creates a brand-new instance of the model(untrained)
    history = model.fit(partial_train_data,
                    partial_train_labels,
                    validation_data=(val_data, val_labels),
                    epochs = num_epochs,
                    callbacks=callbacks_list,
                    batch_size = 5,
                    verbose = 1)
    #val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
    #all_scores.append(val_mae)
    loss_hist = history.history['loss']
    all_loss_histories.append(loss_hist)
    
    val_loss = history.history['val_loss']
    val_loss_histories.append(val_loss)
    
    acc_hist = history.history['accuracy']
    all_acc_histories.append(acc_hist)
    
    val_hist = history.history['val_accuracy']
    val_acc_histories.append(val_hist)
    
    
avg_loss_hist = [ np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]
avg_val_loss_hist = [ np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]

avg_acc_hist = [ np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
avg_val_acc_hist = [ np.mean([x[i] for x in val_acc_histories]) for i in range(num_epochs)]
    

# ##Traing the model wihtout shuffle and without k-fold cross validation

# history = model.fit(partial_train_data,
#                     partial_train_labels,
#                     epochs = num_epochs,
#                     batch_size = 1,
#                     verbose = 0)


##Plotting the training and validation loss
# loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(1, len(avg_loss_hist) + 1)

plt.plot(epochs, avg_loss_hist, 'c-', label='Training loss')
plt.plot(epochs, avg_val_loss_hist, 'purple', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.xlim(0,20)
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

smooth_loss_hist = smooth_curve(avg_loss_hist[10:])
smooth_val_loss_hist = smooth_curve(avg_val_loss_hist[10:])
plt.plot(range(1, len(smooth_loss_hist) + 1), smooth_loss_hist, 'c-', label='Training loss')
plt.plot(range(1, len(smooth_val_loss_hist) + 1), smooth_val_loss_hist, 'purple', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



##Plotting the training and validation accuracy
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

plt.plot(epochs, avg_acc_hist, 'co-', label='Training acc')
plt.plot(epochs, avg_val_acc_hist, 'purple', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


smooth_acc_hist = smooth_curve(avg_acc_hist[10:])
smooth_acc_loss_hist = smooth_curve(avg_val_acc_hist[10:])
plt.plot(range(1, len(smooth_acc_hist) + 1), smooth_acc_hist, 'c-', label='Training acc')
plt.plot(range(1, len(smooth_acc_loss_hist) + 1), smooth_acc_loss_hist, 'purple', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
