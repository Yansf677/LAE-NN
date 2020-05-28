
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.backend as K
from keras.models import Model 
from keras.layers import Dense, Input
from keras.callbacks import  EarlyStopping


def load_data(category=[4, 7, 22]):
    
    n_class = np.array(category) - 1
    n = n_class.size
    unit_matrix = np.eye(n)
    
    train_data = np.load('TE_480.npy')
    test_data = np.load('TE_960.npy')
    n_train_sample = train_data.shape[0]
    
    train_label = unit_matrix[0,:]
    train_x = train_data[0,:,0]
    test_x = test_data[0,:,0]
    
    for i in range(n):
        train_label = np.vstack((train_label, np.tile(unit_matrix[i,:], (n_train_sample,1))))
        train_x = np.vstack((train_x, train_data[:, :, n_class[i]]))
        test_x = np.vstack((test_x, test_data[:, :, n_class[i]]))
    
    train_x = np.delete(train_x, 0, axis = 0)
    train_label = np.delete(train_label, 0, axis = 0)
    test_x = np.delete(test_x, 0, axis = 0)
    
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x) 
    
    return np.hstack((train_x, train_label)), train_x, test_x

def loss(x,c):
    
    return 1.0 * K.mean(K.square(x - c)) + 0.5 * K.maximum(0.,K.sqrt(K.square(x - c)) - 0.01)

def get_LAE(x):
    
    LAE_input = Input(shape = (x.shape[1],))
    LAE_hidden = Dense(2, activation = 'tanh')(LAE_input)
    LAE_output = Dense(x.shape[1], activation = 'linear')(LAE_hidden)
    
    LAE = Model(inputs=LAE_input, outputs=LAE_output)
    LAE_encoder = Model(inputs=LAE_input, outputs=LAE_hidden)
    
    LAE.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
    LAE.fit(x, x, epochs = 1000, batch_size = 100, shuffle = True, 
            validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 30)])
    
    return LAE, LAE_encoder
    
def get_NN(x, y):
    
    NN_input = Input(shape = (x.shape[1],))
    NN_hidden_1 = Dense(36, activation = 'tanh')(NN_input)
    NN_hidden_2 = Dense(25, activation = 'tanh')(NN_hidden_1)
    NN_hidden_3 = Dense(15, activation = 'tanh')(NN_hidden_2)
    NN_output = Dense(2, activation = 'linear')(NN_hidden_3)
    NN = Model(inputs=NN_input, outputs=NN_output)
    
    NN.compile(loss=loss, optimizer=keras.optimizers.Adam())
    NN.fit(x, y, epochs=1000, batch_size = 30, shuffle = True, 
           validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])
    
    return NN

def main():
    
    # prepare data
    n_label = 3
    label = [4, 7, 0]
    train_x_label, train_x, test_x = load_data([4, 7, 22])
    
    # get discriminate 2D features
    n_sample = train_x_label.shape[0]
    index = np.arange(n_sample)
    np.random.shuffle(index)
    train_x_label_shuffle = train_x_label[index,:]
    
    LAE, LAE_encoder = get_LAE(train_x_label_shuffle)
    
    center = np.zeros([n_label, 2])
    discrimate_feature = LAE_encoder.predict(train_x_label)
    
    for i in range(n_label):
        center[i] = np.mean(discrimate_feature[i * 480 : (i + 1) * 480, :], axis = 0).reshape(1,2)
        plt.scatter(discrimate_feature[i * 480 : (i + 1) * 480, 0], discrimate_feature[i * 480 : (i + 1) * 480, 1], 
                    label = '{}'.format(label[i]), marker = '*')
        plt.legend(loc = 'best')
    
    plt.title('Discrimate features by LAE', fontsize = 20, color = 'k')
    #plt.xlabel('First dimension',  fontsize = 20, color = 'k')
    #plt.ylabel('Second dimension', fontsize = 20, color = 'k')
    plt.show()
    
    # fit the above 2D features
    discrimate_center = center[0]
    for j in range(n_label):
        discrimate_center = np.vstack((discrimate_center, np.tile(center[j],(480,1))))
    discrimate_center = np.delete(discrimate_center, 0, axis = 0)
    
    NN = get_NN(train_x, discrimate_feature)    
    unsuperivised_feature = NN.predict(train_x)
    
    for i in range(n_label):
        plt.scatter(unsuperivised_feature[i * 480 : (i + 1) * 480, 0], unsuperivised_feature[i * 480 : (i + 1) * 480, 1],
                    label = '{}'.format(label[i]), marker = '*')
        plt.legend(loc = 'best')
        
    plt.title('Features by LAE-NN', fontsize = 20, color = 'k')
    #plt.xlabel('First dimension', fontsize = 20, color = 'k')
    #plt.ylabel('Second dimension', fontsize = 20, color = 'k')
    plt.show()
    
    # using knn algorithm to divide areas of different categories in a 2D plane
    y = np.array([0])
    for k in range(n_label):
        y = np.append(y, k * np.ones([480,]))
    y = np.delete(y, 0, axis=0)

    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(unsuperivised_feature, y)
    
    x_min, x_max = unsuperivised_feature[:, 0].min() - 0.05, unsuperivised_feature[:, 0].max() + 0.05
    y_min, y_max = unsuperivised_feature[:, 1].min() - 0.01, unsuperivised_feature[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha = 0.7)
    plt.scatter(unsuperivised_feature[:, 0], unsuperivised_feature[:, 1], c=y, s=10, edgecolor = 'k')
    plt.show()


if __name__ == '__main__':
    
    np.random.seed(1337)  
    main()
    
    