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
dfData_val = pd.read_pickle("clusters_validation.txt") 


# #adding 4 ppoints each between 2 frames for every cell to make the no. of frames = 180, i.e. 3 hours
# fr_36_data = dfData.traj #fr_36_data[0] is x and fr_36_data[1] is y
# fl = [] #final array after adding all the 4 new elements between 2 frames. Total 180 xl and 180 yl

# for i in range(len(fr_36_data)):
#     xl = [] # temporary array to hold x values
#     yl = [] # temporary array to hold y values
#     for j in range(len(fr_36_data[i][0]) - 1): # no. of frames
#         xl.append(fr_36_data[i][0][j]) # to take the 1st value of an array
#         yl.append(fr_36_data[i][1][j])
#         for r in range(4) :
#             xl.append(xl[-1] + (fr_36_data[i][0][j+1] - fr_36_data[i][0][j]) / 5) 
#             yl.append(yl[-1] + (fr_36_data[i][1][j+1] - fr_36_data[i][1][j]) / 5)
#     fl.append([xl,yl])

# dfData['training_traj'] = fl # data augmented 




dfData = dfData[dfData.pattern != 'Mix']

#Single cell movies
xb= pd.read_csv("B_CellCx.txt")
xb = xb.Var1.tolist()
yb = pd.read_csv("B_CellCy.txt")
yb = yb.Var1.tolist()

xw= pd.read_csv("W_CellCx.txt")
xw = xw.Var1.tolist()
yw = pd.read_csv("W_CellCy.txt")
yw = yw.Var1.tolist()

xs= pd.read_csv("S_CellCx.txt")
xs = xs.Var1.tolist()
ys = pd.read_csv("S_CellCy.txt")
ys = ys.Var1.tolist()

# Append the single cell values to the validation set dataframe
dfData_val.loc[len(dfData_val)]=[[xb,yb],'Bidirectional','M2',2] 
dfData_val.loc[len(dfData_val)]=[[xw,yw],'Wandering','M1',1] 
dfData_val.loc[len(dfData_val)]=[[xs,ys],'Spinning','M0',9] 



# split the 240 frames data to 180-training(3hrs) and make it 5mins frames each for 36 points.
fr_240_data = dfData_val.traj
train_240f_36 = []   #final array having 36 datapoints in validation dataset
for i in range(len(fr_240_data)):
    xl_36 = []; yl_36 = [];
    for j in range(37):
        xl_36.append(fr_240_data[i][0][j*5])
        yl_36.append(fr_240_data[i][1][j*5])
    train_240f_36.append([xl_36,yl_36])
    

dfData_val['traj_36'] = train_240f_36

# split the 240 frames data to 180-training(3hrs) and 60-testing(last 1 hr)
fr_240_data = dfData_val.traj
training_180_data = []
for i in range(len(fr_240_data)):
    xl_180 = []; xl_60 = [];
    yl_180 = []; yl_60 = [];
    for j in range(len(fr_240_data[i][0]) - 1):
        xl_180 = fr_240_data[i][0][:180]
        yl_180 = fr_240_data[i][1][:180]
    training_180_data.append([xl_180,yl_180])
    
dfData_val['training_traj'] = training_180_data

# Trying to correct the labels manually for the mislabelled cells from validation dataset
dfData_val.pattern[22] = 'Wandering' 
dfData_val.pattern[18] = 'Wandering' 
dfData_val.pattern[16] = 'Bidirectional' 
dfData_val.pattern[13] = 'Bidirectional' 
dfData_val.pattern[5] = 'Bidirectional' 

#to check the trajectories
#plt.plot(dfData.training_traj[84][0],dfData.training_traj[84][1])   #68,74,78,79,80,81 wrong
#plt.plot(dfData.traj[84][0],dfData.traj[84][1])   #68,74,78,79,80,81 wrong


#tuples (x,y)
tup1 = []
for i in range(len(dfData)):
    tupxy = [];
    for k in range(len(dfData.training_traj[0][0])):
        tupxy.append((dfData.training_traj[i][0][k],dfData.training_traj[i][1][k]))
    tup1.append(tupxy)
        
#tuples (x,y)
tup2 = []
for i in range(len(dfData_val)):
    tupxy2 = [];
    for k in range(len(dfData.training_traj[0][0])):
        tupxy2.append((dfData_val.training_traj[i][0][k],dfData_val.training_traj[i][1][k]))
    tup2.append(tupxy2)
    
dfData['tup_traj'] = tup1
dfData_val['tup_traj'] = tup2


df1 = dfData.drop(labels = ['traj','parentImg','patternNo','training_traj'],axis =1)
df2 = dfData_val.drop(labels = ['traj','parentImg','patternNo','training_traj'],axis =1)

df = pd.concat([df1,df2], ignore_index=True)

sns.countplot(x='pattern', data =df)

#Normalize
trajs_list = df['tup_traj'].values.tolist() # convert the dataframe column to list
trajs = np.array(trajs_list) 
#"""Normalize each feature to have a mean of 0 and to have a standard deviation of 1."""
trajs -= trajs.mean()
trajs /= trajs.std()

traj_labels = df['pattern']


plt.plot(trajs[0][0],trajs[0][1])   #68,74,78,79,80,81 wrong
plt.plot(-trajs[34][0],trajs[34][1])




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
x_val = x_train[:150]
partial_x_train = x_train[150:]

y_val = y_train_one_hot[:150]
partial_y_train = y_train_one_hot[150:]


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


feature_extractor.add(Dense(10, activation='relu',kernel_initializer='he_uniform',  input_shape=(180,2)))

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
prediction_NN = cnn_model.predict(x_val)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train[:150], prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)

