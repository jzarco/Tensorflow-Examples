import tensorflow as tf
from image_reader_utils import *
######################################
"""CREATE PLACEHOLDER FOR DATA IN"""
######################################

X = tf.placeholder(shape=[None,100,100,3],dtype=tf.float32)
Y = tf.placeholder(shape=[None,10],dtype=tf.float32)

######################################
"""BUILD FILTERS WITH BIASES"""
######################################
"""
Filters should be of shape height, width, and number of channels with defined total number of filters. 
e.g. [4,4,3,32] defines a 4x4 filter for a 3 channel input that outputs 32 channels.
"""
F1 = tf.get_variable('F1',shape=[4,4,3,32],initializer=tf.contrib.layers.xavier_intializer(seed=1))
fb1 = tf.get_variable('fb1',shape=[32],initializer=tf.zeros_initializer())

F2 = tf.get_variable('F2',shape=[4,4,32,64],initializer=tf.contrib.layers.xavier_intializer(seed=1))
fb2 = tf.get_variable('fb2',shape=[64],initializer=tf.zeros_initializer())

F3 = tf.get_variable('F3',shape=[4,4,64,128],initializer=tf.contrib.layers.xavier_intializer(seed=1))
fb2 = tf.get_variable('fb2',shape=[128],initializer=tf.zeros_initializer())


######################################
"""BUILD WEIGHTS WITH BIASES FOR FULLY CONNECTED LAYERS"""
######################################

W1 = tf.get_variable('W1',shape=[1024],initializer=tf.contrib.layers.xavier_intializer(seed=1))
b1 = tf.get_variable('b1',shape=[1024,1],initializer=tf.zeros_initializer())

W2 = tf.get_variable('W2',shape=[512,1024],initializer=tf.contrib.layers.xavier_intializer(seed=1))
b2 = tf.get_variable('b2',shape=[512,1],initializer=tf.zeros_initializer())

W3 = tf.get_variable('W3',shape=[10,512],initializer=tf.contrib.layers.xavier_intializer(seed=1))
b3 = tf.get_variable('b3',shape=[10,1],initializer=tf.zeros_initializer())

######################################
"""STORE FILTERS AND WEIGHTS """
######################################
parameters = {'F1':F1,
              'fb1':fb1,
              'F2':F2,
              'fb2':fb2,
              'F3':F3,
              'fb3':fb3,
              'W1':W1,
              'b1':b1,
              'W2':W2,
              'b2':b2,
              'W3':W3,
              'b3':b3}
######################################
"""CREATE WRAPPERS FOR CONVOLUTIONAL NETWORK OPERATIONS"""
######################################

def conv2d(x,filter,strides=1,padding='SAME',bias):
    y = tf.nn.conv2d(x,filter,strides=strides,padding=padding)
    y = tf.nn.bias_add(y,bias)
    y = tf.nn.relu(y)
    return y

def maxpool2d(x,k=2,padding='SAME'):
    y = tf.nn.max_pool(y,ksize=[1,k,k,1],strides=[1,k,k,1],padding=padding)
    return y

def dropout(x,prob):
    y = tf.nn.dropout(x,prob)
    return y

def dense(x,W,b):
    y = tf.add(tf.matmul(W,x),b)
    y = tf.nn.relu(y)
    return y

def flatten(x):
    y = tf.layers.flatten(x)
    return y

def one_hot_mat(y, n_classes):
    """

    :param y: label vector
    :param n_classes: number of different classes
    :return: one hot tensorflow matrix
    """

    n_c = tf.constant(n_classes,name='n_classes')

    one_hot_mat = tf.one_hot(y,depth=n_c,axis=0)

    sess = tf.Session()

    one_hot = sess.run(one_hot_mat)

    sess.close()

    return one_hot

######################################
"""BUILD CONVNET"""
######################################


def convnet(X,parameters,keep_prob):

    ####FIRST CONVOLUTION LAYER####
    conv1 = conv2d(X,parameters['F1'],strides=1,padding='SAME',parameters['fb1'])
    conv1 = maxpool2d(conv1,k=3)

    ####SECOND CONVOLUTION LAYER####
    conv2 = conv2d(conv1,parameters['F2'],strides=1,padding='SAME',parameters['fb2'])
    conv2 = maxpool2d(conv2,k=3)

    ####THIRD CONVOLUTION LAYER####
    conv3 = conv2d(conv2,parameters['F3'],strides=1,padding='SAME',parameters['fb3'])
    conv3 = maxpool2d(conv3,k=3)

    ####FULLY CONNECTED LAYERS####
    fc1 = flatten(conv3)
    s = tf.shape(fc1)
    parameters['W1'] = tf.get_variable('W1',shape=[s,1024],initializer=tf.contrib.layers.xavier_intializer(seed=1))

    d1= dense(fc1,parameters['W1'],parameters['b1'])
    d1 = dropout(d1,keep_prob)
    d2 = dense(d1,parameters['W2'],parameters['b2'])
    d2 = dropout(d2,keep_prob)
    out = dense(d2,parameters['W3'],parameters['b3'])

    return out

######################################
"""CHECK PREDICTIONS AND DEFINE OPTIMIZATION"""
######################################

logits = convnet(X,parameters,0.8)
predictions = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_train))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

evaluate = tf.equal(tf.argmax(predictions),tf.argmax(Y))

accuracy = tf.reduce_mean(tf.cast(evaluate,tf.float32))

######################################
"""INITIALIZE TENSORS AND BEGIN TRAINING"""
######################################

init = tf.global_variables_initializer()

training_data = "C:\\Users\\Juan.Zarco\\Documents\\GitHub\\Tensorflow-Examples\\Softmax MNIST\\fruits-360\\Training"
testing_data = "C:\\Users\\Juan.Zarco\\Documents\\GitHub\\Tensorflow-Examples\\Softmax MNIST\\fruits-360\\Test"

X_train,Y_train = build_imgdata_from_dirs(training_data)
X_test,Y_test = build_imgdata_from_dirs(testing_data)

n_c = len(np.unique(Y_train))

Y_train = one_hot_mat(Y_train,n_c)

def model(X_train,Y_train,X_test,Y_test,epochs=15,batch_size=5):

    m = X_train.shape[0]

    with tf.Session() as sess:

        sess.run(init)

        shuffler = np.random.permutation(m)
        X_train = X_train[shuffler,:,:,:]


        for epoch in range(epochs):

