'''

doctorAI re-implementation code in tensorflow
by seungbin
    
Env

print(tf.__version__)
1.0.1

sys.version
'3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  5 2016, 11:41:13) [MSC v.1900 64 bit (AMD64)]'
    
'''

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

import numpy as np

from class_GRU_mnist import GRU_Cell

import sys
import os
from tqdm import tqdm
import datetime


def process_batch_input(raw_input):
    """
        Function to convert batch input data to use scan ops of tensorflow.
    """     
    # lists of [batch_size, timestep_input_size]
    batch_input = tf.split(raw_input, num_or_size_splits=28, axis=1)
    batch_input = tf.stack(batch_input) # list to dimension [timestep, batch_size, timestep_input_size]
    return batch_input
    
    
#%%
# Dataset Preparation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\data.img\MNIST_data", one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.2)


# Hyper-paramaters
hidden_layer_size = 128
raw_input_size = 28 * 28
timestep = 28
timestep_input_size = 28
target_size_code = 10
target_size_time = 28

batch_size = 64

starter_learning_rate = 0.0005

#%% Graph Definition init

# Initialize Weights for model]
gru = GRU_Cell(timestep_input_size, hidden_layer_size, target_size_code, target_size_time)


#%% graph def shape

raw_input = tf.placeholder(tf.float32, shape=[None, raw_input_size], name='raw_input') #

x_i = process_batch_input(raw_input) # [timestep, batch_size, timestep_input_size]

y_next = tf.placeholder(tf.float32, shape=[None, target_size_code], name='y_next')
y_hat_next = gru.get_outputs_code(x_i)[:-1] # [timestep, batch_size, target_size]

d_next = x_i[1:]
d_hat_next = gru.get_outputs_time(x_i)[:-1]
                           
print("x_i :", x_i.get_shape())
print("y_i_hat :", y_hat_next.get_shape())

#%% graph ends

# Computing the Cross Entropy loss
cross_entropy = tf.reduce_mean(tf.map_fn(lambda i: tf.nn.softmax_cross_entropy_with_logits(
                            labels=y_next, logits=y_hat_next[i]), tf.range(x_i.shape[0]-1), dtype=tf.float32))
se = 0.5 * tf.reduce_mean(tf.square(d_hat_next - d_next))
loss = cross_entropy + se
so_loss = tf.summary.scalar('loss', loss)

# optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)


# Calculatio of correct prediction and accuracy
correct_prediction = tf.equal(tf.argmax(tf.expand_dims(y_next, 0), 2), tf.argmax(y_hat_next, 2))

accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

acc = tf.summary.scalar('accuracy', accuracy)

#%%
#session init
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#tensorboard
filelist = [ f for f in os.listdir("tensorboard/train/")]
for f in filelist:
    os.remove("tensorboard/train/" + f)
    
filelist = [ f for f in os.listdir("tensorboard/test/")]
for f in filelist:
    os.remove("tensorboard/test/" + f)
    
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("tensorboard/train")
test_writer = tf.summary.FileWriter("tensorboard/test")

train_writer.add_graph(sess.graph)

#%% Iterations to do trainning
for epoch in tqdm(range(500)):

    def log_metrics_for_batch(X_batch, Y_batch):
        
        #for train set
        feed_dict = {raw_input: X_train, y_next: y_train, \
                     gru.initial_hidden: np.zeros([X_train.shape[0], hidden_layer_size])}        
        train_accuracy, summ_acc = sess.run([accuracy, acc], feed_dict=feed_dict)
        train_loss, summ_loss = sess.run([loss, so_loss], feed_dict=feed_dict)
        train_writer.add_summary(summ_acc, sess.run(global_step))
        train_writer.add_summary(summ_loss, sess.run(global_step))
        
        #for test set
        feed_dict = {raw_input: X_test, y_next: y_test, \
                     gru.initial_hidden: np.zeros([X_test.shape[0], hidden_layer_size])}
        test_accuracy, summ_acc = sess.run([accuracy, acc], feed_dict=feed_dict)
        test_loss, summ_loss = sess.run([loss, so_loss], feed_dict=feed_dict)
        test_writer.add_summary(summ_acc, sess.run(global_step))  
        test_writer.add_summary(summ_loss, sess.run(global_step))
        
        sys.stdout.flush()
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\r%s iter: [%d] %d Loss(Tr/Te): %s / %s Acc(Tr/Te): %s / %s" % \
              (now, epoch, sess.run(global_step), str(train_loss), str(test_loss),  str(train_accuracy), str(test_accuracy)))
        

    X_train, y_train = shuffle(X_train, y_train)
    
    start = 0
    end = batch_size
    for i in range( int(X_train.shape[0] / batch_size) ):
 
        X_batch = X_train[start:end]
        Y_batch = y_train[start:end]
        start = end
        end = start + batch_size
        
        # train , gradient descent
        feed_dict = {raw_input: X_batch, y_next: Y_batch, \
                     gru.initial_hidden: np.zeros([batch_size, hidden_layer_size])}
        sess.run(train_step, feed_dict=feed_dict)
        
        if (sess.run(global_step) % 100) == 0:
            log_metrics_for_batch(X_batch, Y_batch)

    

train_writer.close()
test_writer.close()

#sess.close()
