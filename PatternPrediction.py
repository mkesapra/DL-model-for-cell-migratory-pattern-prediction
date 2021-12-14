#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:42:27 2021

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

cluster_x = columns.CellCx[columns.Cluster == 3]
cluster_y = columns.CellCy[columns.Cluster == 3]
imageFrom = columns.imageName[columns.Cluster == 3]


cluster_size = cluster_x.size


traj =[]

# #traj in the shaope of [[[x_11,y_11],[x_12,y_12]...,[x_136,y_136]],[[x_21,y_21]...[x_236,y_236]],...[[x_361,y_361],..[x_3636,y_3636]]]
# for k in range(0, cluster_size):
#     #for i in range(0,num_frames):
#     x = cluster_x.iloc[k]
#     y = cluster_y.iloc[k]
#     traj.append(np.column_stack((x, y)))

#traj in the shape of [[[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]],[[10, 11, 12, 13, 14],[15, 16, 17, 18, 19]],[[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

for k in range(0, cluster_size):
    x = cluster_x.iloc[k]
    y = cluster_y.iloc[k]
    traj.append([x,y])
    

d = {'traj': traj , 'pattern': 'Spinning','parentImg':imageFrom, 'patternNo': 9.}
cluster0 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Wandering','parentImg':imageFrom, 'patternNo': 1.}
cluster1 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Bidirectional','parentImg':imageFrom, 'patternNo': 2.}
cluster2 =  pd.DataFrame(data=d)

d = {'traj': traj , 'pattern': 'Mix','parentImg':imageFrom , 'patternNo': 3.}
cluster3 =  pd.DataFrame(data=d)


clusters = pd.concat([cluster0,cluster1,cluster2,cluster3], ignore_index=True)
clusters.to_csv('clusters.txt')


from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras.utils import normalize, to_categorical
from sklearn.utils import shuffle
 

# #k-fold cross validation using sklearn
# kf = KFold(n_splits=5, shuffle=True, random_state=2652124)
# k_trn =[]; k_tst =[];
# for trn, tst in kf.split(clusters.traj,clusters.patternNo):
#     print("%s %s" % (trn, tst))
#     k_trn.append(trn)
#     k_tst.append(tst)


#train,test, train_labels,test_labels =  X[train], X[test], y[train], y[test]
#train, test = train_test_split(data, test_size=0.2) # split 80:20 train:test 
train,test, train_labels,test_labels = train_test_split(clusters.traj,clusters.patternNo, shuffle=True, random_state=2652124)

#shuffle the training data to improve the val accuracy scoere -- Used instead of #np.random.shuffle(rank_3_tensor_train)
train, train_labels = shuffle(train, train_labels)

# tensor creation
rank_3_tensor_train = tf.constant([train,])
rank_3_tensor_train = tf.reshape(rank_3_tensor_train,[rank_3_tensor_train.shape[1], rank_3_tensor_train.shape[2], rank_3_tensor_train.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

rank_3_tensor_test = tf.constant([test,])
rank_3_tensor_test = tf.reshape(rank_3_tensor_test,[rank_3_tensor_test.shape[1], rank_3_tensor_test.shape[2], rank_3_tensor_test.shape[3]]) # 77 cells, 2 column arrays- x & y, 37 time points 

#Not required to create tensors for labels, we use one-hot encode
# train_labels = tf.constant([train_labels,])
# train_labels = tf.reshape(train_labels,[train_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# test_labels = tf.constant([test_labels,])
# test_labels = tf.reshape(test_labels,[test_labels.shape[1]]) # 77 cells, 2 column arrays- x & y, 37 time points 

# normalizing and using one hot encode
rank_3_tensor_train = normalize(rank_3_tensor_train)
rank_3_tensor_test = normalize(rank_3_tensor_test)

#to one hot labels
def to_one_hot(labels,dimension=46,features=2):
    results = np.zeros((len(labels),features, dimension))
    for j in range(1,features):        
        for i, label in enumerate(labels):           
            label = int(label)
            results[i, j, label] = 1.
        return results

#labels train and test
hot_train_labels = to_one_hot(train_labels)
hot_test_labels = to_one_hot(test_labels)


# #labels train and test using to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(2, 37)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    
    model.compile(optimizer='rmsprop',
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
num_epochs = 500
all_scores = []
all_loss_histories = [];all_acc_histories = []; val_loss_histories =[];val_acc_histories =[]
#np.random.shuffle(rank_3_tensor_train)

validation_scores = []

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
                    batch_size = 1,
                    verbose = 0)
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