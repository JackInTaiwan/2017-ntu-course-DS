import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import os
import random
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def unpickle(file, mode_bytes = True) :
    with open('cifar-10-batches-py/' + file, 'rb') as fo :
        dict = pickle.load(fo, encoding='bytes') if mode_bytes == True else pickle.load(fo)
    return dict

def load_data(index=6) :
    file_names = ['data_batch_' + str(i) for i in range(1,index)]
    x_train, y_train = None, None
    for i, file_name in enumerate(file_names) :
        data_batch = unpickle(file_name)
        if i == 0 :
            x_train, y_train = data_batch[b'data'], data_batch[b'labels']
        else :
            x_train = np.vstack((x_train, data_batch[b'data']))
            y_train = np.concatenate((y_train, data_batch[b'labels']))
    return x_train, y_train

class data_batch (object) :
    def __init__(self, x_train, y_train, size) :
        self.x_train = x_train[:]
        self.y_train = y_train[:]
        self.size = size
        self.index = 0
        self.len = x_train.shape[0]
    def generate_batch (self) :
        if (self.index + self.size) <= self.len :
            x_train_batch_ = self.x_train[self.index:(self.index + self.size)][:]
            y_train_batch_ = self.y_train[self.index:(self.index + self.size)][:]
            self.index += self.size
            return x_train_batch_, y_train_batch_
        else :
            self.index = 0
            #train_pairs = list(zip(self.x_train, self.y_train))
            #random.shuffle(train_pairs)
            #self.x_train, self.y_train = list(zip(*train_pairs))
            #self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
            x_train_batch_ = self.x_train[self.index: (self.index + self.size)][:]
            y_train_batch_ = self.y_train[self.index: (self.index + self.size)][:]
            self.index = self.index + self.size
            return x_train_batch_, y_train_batch_


file_name = 'data_batch_1'
data = unpickle(file_name)
x_train, y_train = np.array(data[b'data'])[:], np.array(data[b'labels'])[:]
y_train_ohe = np.array(OneHotEncoder().fit_transform(y_train.reshape(-1,1)).todense())

#x_test_cross, y_test_cross = x_train[0:100][:], y_train[0:100][:]
#y_test_cross_ohe = OneHotEncoder().fit_transform(y_test_cross.reshape(-1,1)).todense()

file_name_test = 'test_batch'
data_test = unpickle(file_name_test)

x_test, y_test = np.array(data_test[b'data'])[0:1000], np.array(data_test[b'labels'])[0:1000]
y_test_ohe = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).todense()


def weight_var(shape) :
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_var(shape) :
    init = tf.constant(1.0, shape=shape)
    return tf.Variable(init)

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x) :
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




xs = tf.placeholder(tf.float32, [None, 3072])
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs, [-1, 32, 32, 3])

W_conv1 = weight_var([10, 10, 3, 32])
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_var([5, 5, 32, 96])
b_conv2 = bias_var([96])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_var([5, 5, 96, 192])
b_conv3 = bias_var([192])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


W_fc1 = weight_var([4 * 4 * 192, 2 ** 14])  #7*7*64 inputs to 1024 neurons
b_fc1 = bias_var([2 ** 14])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 192])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_var([2 ** 14, 10])  #1024 neurons to 10 neurons
b_fc2 = bias_var([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001#starting with 0.1 or 0.01 causes bad performances
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.95, staircase=True)


cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred_cross = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
acc_cross = tf.reduce_mean(tf.cast(correct_pred_cross, tf.float32))
batch = data_batch(x_train, y_train_ohe, 100)

### Saver
saver = tf.train.Saver()

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    #x_train_batch, y_train_batch = x_train[0:100][:], y_train_ohe[0:100][:]
    #saver.restore(sess, '/home/jack/PycharmProjects/DS/model_20171122_22_32.ckpt')
    #init = tf.initialize_variables([global_step])
    #sess.run(init)
    for i in range(10000) :
        x_train_batch, y_train_batch = batch.generate_batch()
        x_train_batch__, y_train_batch__ = x_train[(i % 100) * 100:(i % 100) * 100 + 100], y_train_ohe[(i % 100) * 100:(i % 100) * 100 + 100]
        x_test_cross, y_test_cross = x_train[0:1000][:], y_train_ohe[0:1000][:]
        train_step.run(feed_dict={xs: x_train_batch, ys: y_train_batch, keep_prob: 0.75})
        if i % 10 == 0 :
            print (i)
        if i % 50 == 0 :
            _, loss, l_r, y_c = sess.run([train_step, cross_entropy, learning_rate, y_conv], feed_dict={xs: x_train_batch, ys: y_train_batch, keep_prob: 1.0})
            print ('l_r', l_r)
            print ('loss:', loss)
            #print ('y_c', y_c[0:20])
            #print ('y_', y_test_cross[0:20])
            train_accuracy = acc.eval(feed_dict={xs: x_test, ys: y_test_ohe, keep_prob: 1.0})
            print ('acc:', train_accuracy)
            train_accuracy_cross = acc_cross.eval(feed_dict={xs: x_test_cross, ys: y_test_cross, keep_prob: 1.0})
            print ('acc for cross', train_accuracy_cross)
        """
        if i % 100 == 0 :
            now = datetime.datetime.now()
            date = "%s%s%s_%s_%s" % (now.year, now.month, now.day, now.hour, now.minute)
            saver.save(sess, '/home/jack/PycharmProjects/DS/model_%s.ckpt' % date)
            print ('Save %s with acc=%s acc_cross=%s' % (date, train_accuracy, train_accuracy_cross))
        """