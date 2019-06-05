
import keras
import keras.backend as K
from keras.callbacks import  EarlyStopping
from keras.models import Model 
from keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


np.random.seed(1337)  # for reproducibility

# training phase
# load data using matlab
print('loading data ...')

label_number = 3
label = [4, 7, 0]
center = np.zeros([label_number, 2])
sample_number, dimension = train_data.shape
index = np.arange(sample_number)
np.random.shuffle(index)
train_data_shuffle = train_data[index,:]

# dimension reduction using autoencoder
print('dimension reduction by LAE ...')

input_1 = Input(shape = (dimension,))
encode_1 = Dense(2, activation = 'tanh')(input_1)
decode_1 = Dense(dimension, activation = 'linear')(encode_1)
autoencoder_1 = Model(inputs = input_1, outputs = decode_1)
encoder_1 = Model(inputs = input_1, outputs = encode_1)

autoencoder_1.compile(loss='mean_squared_error', optimizer = keras.optimizers.SGD(lr = 0.1))
autoencoder_1.fit(train_data_shuffle, train_data_shuffle, epochs = 1000, batch_size = 100, shuffle = True, 
                  validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])

# plot the 2D features for model 1 during the training phase
print('LAE results ...')
train_feature_1 = encoder_1.predict(train_data)
for i in range(label_number):
    center[i] = np.mean(train_feature_1[i * 480 : (i + 1) * 480, :], axis = 0).reshape(1,2)
    plt.scatter(train_feature_1[i * 480 : (i + 1) * 480, 0], train_feature_1[i * 480 : (i + 1) * 480, 1], 
                label = '{}'.format(label[i]), marker = '*')
    plt.legend(loc = 'best')
plt.title('Clustering by LAE', fontsize = 20, color = 'k')
plt.xlabel('First dimension', fontsize = 20, color = 'k')
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize = 18)
plt.ylabel('Second dimension', fontsize = 20, color = 'k')
plt.yticks(fontsize = 18)
plt.show()

# extract the centers of different categories
train_feature_center = np.append(np.tile(center[0],(480,1)),np.tile(center[1],(480,1)),axis = 0)
for j in range(label_number-2):
    train_feature_center = np.append(train_feature_center,np.tile(center[j+2],(480,1)),axis = 0)
train_x_shuffle = train_x[index,:]
train_feature_center_shuffle = train_feature_center[index,:]

# second model for learning the centers
print('center the results ...')

# a new loss function for better clustering
def loss(x,c):
    return 1.0 * K.mean(K.square(x - c)) + 0.5 * K.maximum(0.,K.sqrt(K.square(x - c)) - 0.01)

input_2 = Input(shape = (52,))
encode_2 = Dense(36, activation = 'tanh')(input_2)
encode_2 = Dense(25, activation = 'tanh')(input_2)
encode_2 = Dense(15, activation = 'tanh')(encode_2)
output_2 = Dense(2, activation = 'linear')(encode_2)
autoencoder_2 = Model(inputs = input_2, outputs = output_2)

autoencoder_2.compile(loss = loss, optimizer = keras.optimizers.SGD(lr = 0.1))
autoencoder_2.fit(train_x_shuffle, train_feature_center_shuffle, epochs=1000, batch_size = 30, shuffle = True, 
                  validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])

# plot the 2D features of model 2 during training phase
print('LAE-NN results ...')
train_feature_2 = autoencoder_2.predict(train_x)
for i in range(label_number):
    plt.scatter(train_feature_2[i * 480 : (i + 1) * 480, 0], train_feature_2[i * 480 : (i + 1) * 480, 1],
                label = '{}'.format(label[i]), marker = '*')
    plt.legend(loc = 'best')
plt.title('Clustering by LAE-NN', fontsize = 20, color = 'k')
plt.xlabel('First dimension', fontsize = 20, color = 'k')
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize = 18)
plt.ylabel('Second dimension', fontsize = 20, color = 'k')
plt.yticks(fontsize = 18)
plt.show()

# using knn algorithm to divide areas of different categories in a 2D plane
print('decision plane ...')

# based on meshing the grids in a plane and decide the category of every grid
y = 0 * np.ones([480,])
y = np.append(y, 1 * np.ones([480,]))
y = np.append(y, 2 * np.ones([480,]))

clf = KNeighborsClassifier(n_neighbors = 10)
clf.fit(train_feature_2, y)
x_min, x_max = train_feature_2[:, 0].min() - 0.05, train_feature_2[:, 0].max() + 0.05
y_min, y_max = train_feature_2[:, 1].min() - 0.005, train_feature_2[:, 1].max() + 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.7)
plt.scatter(train_feature_2[:, 0], train_feature_2[:, 1], c=y, s=10, edgecolor = 'k')
plt.annotate('4', xy = (0.7, -0.97), fontsize = 40, color = 'k')
plt.annotate('7', xy = (-0.9, -0.97), fontsize = 40, color = 'k')
plt.annotate('0', xy = (-0.2, -0.97), fontsize = 40, color = 'k')
plt.title('Decision plane of IDV(0), (4) and (7)', fontsize = 20, color = 'k')
plt.xlabel('First dimension', fontsize = 20, color = 'k')
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize = 18)
plt.ylabel('Second dimension', fontsize = 20, color = 'k')
plt.yticks(fontsize = 18)
plt.show()

# testing phase
print('testing start ...')

# function to calculate the accuracy
def accuracy(pre,true):
    count_false = 0;count_true = 0
    for i in range(960):
        if pre[i] == true[i]:
            count_true = count_true + 1
        else:
            count_false = count_false + 1
    return count_true / 960

test_feature = autoencoder_2.predict(test_x); Z_test = clf.predict(test_feature)

Class4_predict = Z_test[0:960]; Class4_true = np.append(2*np.ones([160,]), 0*np.ones([800,]))
acc4 = accuracy(Class4_predict,Class4_true )

Class7_predict = Z_test[960:1920]; Class7_true = np.append(2*np.ones([160,]), 1*np.ones([800,]))
acc7 = accuracy(Class7_predict,Class7_true )

Class0_predict = Z_test[1920:2880]; Class0_true = np.append(2*np.ones([160,]), 2*np.ones([800,]))
acc0 = accuracy(Class0_predict,Class0_true )

# plot the accuracy
plt.scatter([x for x in range(2880)], Z_test, marker = '*')
plt.vlines(960, -0.2, 2.5, colors = 'r', linestyles = "dashed")
plt.vlines(1920, -0.2, 2.5, colors = 'r', linestyles = "dashed")
plt.annotate('FDR = 100%', xy = (120, 0.1), fontsize = 14, color = 'k')
plt.annotate('FDR = 100%', xy = (1080, 1.1), fontsize = 14, color = 'k')
plt.annotate('FDR = 99.4%', xy = (2040, 2.1), fontsize = 14, color = 'k')
plt.title('Results of testing', fontsize = 20, color = 'k')
plt.xlabel('samples', fontsize = 20, color = 'k')
plt.xticks([960, 1920, 2880], fontsize = 18)
plt.xlim(0.0, 2880)
plt.ylabel('Predicted category', fontsize = 20, color = 'k')
plt.yticks([0, 1, 2], ['IDV(4)', 'IDV(7)', 'IDV(0)'], fontsize = 18)
plt.ylim(-0.2,2.5)
plt.show()











