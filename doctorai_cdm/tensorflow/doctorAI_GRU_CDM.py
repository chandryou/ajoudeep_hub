'''

RETAIN re-implementation code in tensorflow
CDM version

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
import pandas as pd
from itertools import islice, chain

from class_GRU import GRU_Cell

import sys
import os
from tqdm import tqdm
import datetime

#os.environ['R_USER'] = "windows user name"
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

#%% load rds to pandas
left_col = 7

pandas2ri.activate()
readRDS = robjects.r['readRDS']
df = readRDS('data/encodedcohort.rds')
df = pandas2ri.ri2py(df)

df.ix[:,left_col:] = df.ix[:,left_col:].astype(int)
df['person_id'] = df.person_id.astype(int)

# timestamp in day unit from 1970-01-01 to datetime format
df['visit_start_date'] = df.visit_start_date.astype(int).apply(lambda diff: pd.to_datetime(diff, unit='D'))

df = df.reset_index(drop = True)

#%% 

  
def process_batch_input(raw_input):
    """
        Function to convert batch input data to use scan ops of tensorflow.
    """
    
    # [timestep, raw_input_size]

    x_i = raw_input[:-1] # 1 to n-1, code+day part    
    y_next = raw_input[1:, :-1] # 2 to n, code part
    d_next = raw_input[1:, -1] # 2 to n, day part

    x_i = tf.expand_dims(x_i, axis=1)
    y_next = tf.expand_dims(y_next, axis=1)
    d_next = tf.reshape(d_next, [-1, 1, target_size_time])
    
    return x_i, y_next, d_next
    
#%%
# Dataset Preparation
# labels : y_train = np.random.randint(0, 2, size=df.index.unique().shape)

# Hyper-paramaters
hidden_layer_size = 256
raw_input_size = df.ix[:,left_col:].shape[1] + 1 # dx.... + timediff
timestep_input_size = raw_input_size

target_size_code = timestep_input_size-1
target_size_time = 1

batch_size = 1 # one batch, one person
top_k = 30
starter_learning_rate = 0.00005

#%% Graph Definition init

# Initialize Weights for model]
gru = GRU_Cell(timestep_input_size, hidden_layer_size, target_size_code, target_size_time)


#%% graph def shape

raw_input = tf.placeholder(tf.float32, shape=[None, raw_input_size], name='raw_input') #
x_i, y_next, d_next = process_batch_input(raw_input) # [timestep, batch_size, timestep_input_size]

y_hat_next = gru.get_outputs_code(x_i) # [timestep, batch_size, target_size_code]
d_hat_next = gru.get_outputs_time(x_i)
                           
print("x_i :", x_i.get_shape())

print("y_next :", y_next.get_shape())
print("y_hat_next :", y_hat_next.get_shape())

print("d_next :", d_next.get_shape())
print("d_hat_next :", d_hat_next.get_shape())

#%% graph ends

# Computing the Cross Entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.reshape(y_next, [-1,target_size_code]), 
                    logits=tf.reshape(y_hat_next, [-1, target_size_code])))
se = 0.5 * tf.reduce_mean(tf.square(d_hat_next - d_next))
loss = cross_entropy + se
so_loss = tf.summary.scalar('loss', loss)

# optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)


# Calculatio of correct prediction and accuracy

# for total sequence
# y_hat_next_hot = tf.map_fn(lambda t: tf.reduce_sum(tf.one_hot(t, depth=target_size_code), axis=1), tf.nn.top_k(y_hat_next, 30)[1], 
#                           dtype=tf.float32)
#correct = tf.reduce_sum(y_hat_next_hot * y_next)
#total = tf.reduce_sum(y_next)

y_hat_next_hot = tf.reduce_sum(tf.one_hot(tf.nn.top_k(y_hat_next[-1], 30)[1], depth=target_size_code), axis=1)  # [1, 30]
correct = tf.reduce_sum(y_hat_next_hot * y_next[-1])
total = tf.reduce_sum(y_next[-1])

accuracy = correct / total * 100

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

for e in range(50):
    for idx, group in df.groupby('person_id'):
        if group.values.shape[0] > 1:
            group = pd.concat([ group.ix[:,left_col:], group.datediff ], axis=1)
            feed_dict = {raw_input: group.values, \
                         gru.initial_hidden: np.zeros([batch_size, hidden_layer_size])}

            sess.run(train_step, feed_dict=feed_dict)
            
            test_accuracy, summ_acc = sess.run([accuracy, acc], feed_dict=feed_dict)
            test_loss, summ_loss = sess.run([loss, so_loss], feed_dict=feed_dict)
            test_writer.add_summary(summ_acc, sess.run(global_step))  
            test_writer.add_summary(summ_loss, sess.run(global_step))
            
            n_correct, n_total = sess.run([correct, total], feed_dict=feed_dict)
            
            if (sess.run(global_step) % 50) == 0:
                print("\repoch:", e, "loss:", test_loss, "acc:", test_accuracy, "\thit:", n_correct, "/", n_total)
            
train_writer.close()
test_writer.close()
