import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder



class data_batch () :
    def __init__(self, x_train, y_train, size) :
        self.x_train = x_train
        self.y_train = y_train
        self.size = size
        self.index = 0
        self.len = x_train.shape[0]
    def generate_batch (self) :
        if self.index + self.size <= self.len :
            x_train_batch = self.x_train[self.index:self.index + self.size]
            y_train_batch = self.y_train[self.index:self.index + self.size]
            self.index += self.size
            return x_train_batch, y_train_batch
        else :
            x_train_batch_1 = self.x_train[self.index:][:]
            y_train_batch_1 = self.y_train[self.index:][:]
            train_pairs = list(zip(self.x_train, self.y_train))
            print (train_pairs)
            random.shuffle(train_pairs)
            self.x_train, self.y_train = list(zip(*train_pairs))
            self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

            print (self.y_train)
            x_train_batch = np.vstack((x_train_batch_1, self.x_train[0:self.index + self.size - self.len])) if self.index != self.len else self.x_train[0:self.index + self.size - self.len]
            y_train_batch = np.vstack((y_train_batch_1, self.y_train[0:self.index + self.size - self.len])) if self.index != self.len else self.y_train[0:self.index + self.size - self.len]
            self.index = self.index + self.size - self.len
            #print (x_train_batch.shape, y_train_batch)
            return x_train_batch, y_train_batch

a = [i for i in range (0,12)]
a = np.array(a)
a = a.reshape(4,3)
print (a)
b = [1, 2, 3, 4]
b = np.array(b)
b = np.array(OneHotEncoder().fit_transform(b.reshape(-1,1)).todense())
print (b)
batch = data_batch(a, b, 3)
for i in range(15) :
    x, y = batch.generate_batch()
    print (x, y)