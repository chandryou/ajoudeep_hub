
# coding: utf-8

# In[11]:

import numpy as np
import tensorflow as tf


# In[2]:

data = np.genfromtxt('/Users/chan/OneDrive/Study/CVDPrediction/R/minitrain.csv',delimiter=",")


# In[3]:

data3 = np.genfromtxt('/Users/chan/OneDrive/Study/CVDPrediction/R/3DL_test.csv',delimiter=",")


# In[4]:

trX= data[1:,1:-1]
trX = trX.reshape(-1,56,2191,1)
trX = np.float32(trX)
Y=data[1:,-1]
tr_ = np.zeros((len(Y),2))
for i in range(len(Y)):
    if Y[i]==0:
        tr_[i,0]=1
    if Y[i]==1:
        tr_[i,1]=1
        
trY=tr_


# In[5]:

#making test set
teX= data3[1:,1:-1]
teX = teX.reshape(-1,56,2191,1)
teX = np.float32(teX)
Y=data3[1:,-1]


# In[6]:

te_ = np.zeros((len(Y),2))
for i in range(len(Y)):
    if Y[i]==0:
        te_[i,0]=1
    if Y[i]==1:
        te_[i,1]=1
        
teY=te_


# In[8]:

##For sensitivity and specificity, breaking down with the outcome
nonMIset=teX[Y==0]
MIset=teX[Y==1]

outcome_nonMIset = np.zeros((len(nonMIset),2))
outcome_MIset = np.zeros((len(MIset),2))

outcome_nonMIset[:,0]=1
outcome_MIset[:,1]=1

del(Y)


# In[10]:

batch_size = 128
test_size = 128


# In[12]:

sess=tf.InteractiveSession()


# In[13]:

X=tf.placeholder("float",shape=[None,56,2191,1]) #dim = 56 x 2191


# In[14]:

Y=tf.placeholder("float",shape=[None,2])


# In[15]:

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))

w = init_weights([3,12,1,8])
w2 = init_weights([3,3,8,16])
w3 = init_weights([3,3,16,32])
w4 = init_weights([1344,625])
w_o = init_weights([625,2])

###trial
#w0 = init_weights([28*45,2])
#w00 = init_weights([625,2])


# In[16]:

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


# In[18]:

sess.run(tf.global_variables_initializer())


# In[19]:

def model(X,w,w2,w3,w4,w_o,p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X,w ,  ##l1a shape =(?,56,2191,8)
                                 strides=[1,1,7,1],padding='SAME'))
    l1 = tf.nn.max_pool(l1a,ksize=[1,2,35,1], #l1 shape = (?,28,45,8)
                        strides=[1,2,7,1], padding='SAME')
    l1 = tf.nn.dropout(l1,p_keep_conv)
    
    l2a = tf.nn.relu(tf.nn.conv2d(l1,w2, #l2a shape = (?,28,45,16)
                               strides=[1,1,1,1],padding='SAME'))
    l2 = tf.nn.max_pool(l2a,ksize=[1,2,2,1],#l2 shape = (?,14,23,16)
                       strides=[1,2,2,1],padding='SAME')
    l2 = tf.nn.dropout(l2,p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2,w3, #l3a shape = (?,14,23, 32)
                               strides=[1,1,1,1],padding='SAME'))
    l3 = tf.nn.max_pool(l3a,ksize=[1,2,4,1], 
                                  strides=[1,2,4,1],padding='SAME')
    l3 = tf.reshape(l3, [-1,l3.get_shape()[1].value*l3.get_shape()[2].value*l3.get_shape()[3].value]) #reshape to 8960
    l3 = tf.nn.dropout(l3,p_keep_conv)
    
    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    
    pyx = tf.matmul(l4,w_o)
    return pyx


# In[20]:

py_x =model(X,w,w2,w3,w4,w_o,p_keep_conv, p_keep_hidden)


# In[25]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))


# In[26]:

train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
predict_op = tf.argmax(py_x,1)


# In[27]:

correct_prediction = tf.equal(tf.argmax(py_x,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[32]:

for i in range(5001):
    training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))

    for start, end in training_batch:
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
        """if i%100 ==0:
            print("step %d, training cost %g, test accuracy %g"%(i, sess.run([cost],feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})[0],
                                                                sess.run([accuracy],feed_dict={X: teX, Y: teY,
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})[0]))
            print("test specificity %g, sensitivity %g"%(sess.run([accuracy],feed_dict={X: nonMIset,Y: outcome_nonMIset,
                                            p_keep_conv: 1.0, p_keep_hidden: 1.0})[0],
                                                        sess.run([accuracy],feed_dict={X: MIset,Y: outcome_MIset,
                                            p_keep_conv: 1.0, p_keep_hidden: 1.0})[0]))"""


# In[31]:

training_batch

