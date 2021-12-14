#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 15:36:12 2021

@author: kesaprm

Trying autokeras

datastructure of form in tuples: Pattern, [(x_traj0, y_traj0), (x_1,y_1)...(x_36,y_36)]
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import autokeras as ak

dfData = pd.read_pickle("clusters.txt") 


dfData = dfData[dfData.pattern != 'Mix']



#tuples (x,y)
tup1 = []
for i in range(len(dfData)):
    tupxy = [];
    for k in range(len(dfData.traj[0][0])):
        tupxy.append((dfData.traj[i][0][k],dfData.traj[i][1][k]))
    tup1.append(tupxy)
        

dfData['tup_traj'] = tup1


df = dfData.drop(labels = ['traj','parentImg','patternNo'],axis =1)


sns.countplot(x='pattern', data =df)

#Normalize
trajs_list = df['tup_traj'].values.tolist() # convert the dataframe column to list
trajs = np.array(trajs_list) 
#"""Normalize each feature to have a mean of 0 and to have a standard deviation of 1."""
trajs -= trajs.mean()
trajs /= trajs.std()

traj_labels = df['pattern']


plt.plot(dfData.traj[0][0],dfData.traj[0][1])   #68,74,78,79,80,81 wrong
# plt.plot(-trajs[103][0],trajs[103][1])




#data augmentation
ntrajs = []
ntraj_labels = []


for cl in trajs :
    temp_aug = []
    for a in range(8):
        temp_aug.append([])
    for co in cl :
        temp_aug[0].append((co[0],co[1]))  #(x,y)
        temp_aug[1].append((co[1],co[0]))  #(y,x)
        temp_aug[2].append((-co[0],co[1])) #(-x,y)
        temp_aug[3].append((-co[0],-co[1])) #(-x,-y)
        temp_aug[4].append((co[0],-co[1])) #(x,-y)
        
        temp_aug[5].append((-co[1],co[0])) #(-y,x)
        temp_aug[6].append((-co[1],-co[0])) #(-y,-x)
        temp_aug[7].append((co[1],-co[0])) #(y,-x)
    for ma in temp_aug :
        ntrajs.append(ma)

for a in range(len(trajs)):
    for b in range(8):
        ntraj_labels.append(traj_labels[a])
    
ntrajs = np.array(ntrajs)
ntraj_labels = np.array(ntraj_labels)
    
#Deep learnining model
from sklearn.utils import shuffle

#shuffle the training data to improve the val accuracy scoere -- Used instead of #np.random.shuffle(rank_3_tensor_train)
all_data, all_labels = shuffle(ntrajs, ntraj_labels, random_state=42)


#Split train:test = 80:20
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

#Change the string labels to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

##Fit data to model and split train to validation 
x_val = x_train[:100]
partial_x_train = x_train[100:]

y_val = y_train_one_hot[:100]
partial_y_train = y_train_one_hot[100:]


# # define the search
# clf = ak.StructuredDataClassifier(max_trials= 5)
# clf.fit(partial_x_train,partial_y_train, verbose =1, epochs = 50, validation_data = (x_val, y_val))

#Model
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization

#x_train = np.expand_dims(x_train, axis=3)

#activation = 'sigmoid'

feature_extractor = Sequential()


feature_extractor.add(Dense(10, activation='relu',kernel_initializer='he_uniform',  input_shape=(37,2)))

#feature_extractor.add(BatchNormalization())

feature_extractor.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
feature_extractor.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))

#feature_extractor.add(BatchNormalization())
#feature_extractor.add(Dropout(0.2))

feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
#feature_extractor.add(BatchNormalization())
feature_extractor.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))

#feature_extractor.add(Conv2D(32, 2, activation = activation, padding = 'same', input_shape = (2,37,1)))
#feature_extractor.add(BatchNormalization()) # This reduces the covariate shift. We use the mean and variance of the values in the current batch in batch normalization

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
#x = Dense(2, activation = activation, kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(3, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 




history = cnn_model.fit(partial_x_train, partial_y_train, epochs=500, validation_data = (x_val, y_val))

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
cm = confusion_matrix(y_test, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Bidirectional','Spinning','Wandering'], yticklabels=['Bidirectional','Spinning','Wandering'])

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction_NN, normalize=False)
