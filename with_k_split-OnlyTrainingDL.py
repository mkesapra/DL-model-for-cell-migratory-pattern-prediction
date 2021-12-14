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



def build_model():
    feature_extractor = Sequential()
    feature_extractor.add(Dense(10, activation='relu',kernel_initializer='he_uniform',  input_shape=(37,2)))
    feature_extractor.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
    feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
    feature_extractor.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
    feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
    feature_extractor.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))
    feature_extractor.add(Flatten())
    x = feature_extractor.output  
    
    prediction_layer = Dense(3, activation = 'softmax')(x)
    
    cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
    cnn_model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(cnn_model.summary()) 
    return cnn_model

# #k-fold cross validation from the text book
k = 5
num_val_samples = len(x_train)//k
num_epochs = 500
all_scores = []
all_loss_histories = [];all_acc_histories = []; val_loss_histories =[];val_acc_histories =[]
#np.random.shuffle(rank_3_tensor_train)

validation_scores = []





for fold in range(k):
    #print('processing fold #', i)
    val_data = x_train[num_val_samples * fold: num_val_samples * (fold+1)] # selects the validation-data partition
    val_labels = y_train_one_hot[num_val_samples * fold: num_val_samples * (fold+1)] 
    
    partial_train_data = np.concatenate( [x_train[:fold * num_val_samples],x_train[(fold + 1) * num_val_samples:]], axis=0)
    partial_train_labels = np.concatenate( [y_train_one_hot[:fold * num_val_samples], y_train_one_hot[(fold + 1) * num_val_samples:]], axis=0)
    
    model =  build_model() # creates a brand-new instance of the model(untrained)
    history = model.fit(partial_train_data,
                    partial_train_labels,
                    validation_data=(val_data, val_labels),
                    epochs = num_epochs,
                    batch_size = 10,
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


#de-encode to normal labels
model =  build_model()
prediction_NN = model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Bidirectional','Spinning','Wandering'], yticklabels=['Bidirectional','Spinning','Wandering'])

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction_NN, normalize=False)
