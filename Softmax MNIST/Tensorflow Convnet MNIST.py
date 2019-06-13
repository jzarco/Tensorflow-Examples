import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

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


def create_placeholders(n_h,n_w,n_c,n_y):
    """

    :param n_x: dimensions of X data example
    :param n_y: dimensions of Y data example
    :return: tensorflow variable placeholders
    """

    x = tf.placeholder(tf.float32,shape=[None,n_h,n_w,n_c])
    y = tf.placeholder(tf.float32,shape=[None,n_y])

    return x,y

def create_filter(f_height,f_width,n_input,n_filters):

    filter = tf.Variable(tf.random_normal([f_height,f_width,n_input,n_filters]))

    return filter

def create_conv_parameters(f_sizes,bias=True):
    parameters = {}
    if bias:
        for i,size in enumerate(f_sizes):
            parameters['F'+str(i)] = tf.get_variable('F'+str(i),shape=size,initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters['b'+str(i)] = tf.get_variable('b'+str(i),shape=size[3],initializer=tf.zeros_initializer())
    else:
        for i, size in enumerate(f_sizes):
            parameters['F' + str(i)] = tf.get_variable('F' + str(i), shape=size,initializer=tf.contrib.layers.xavier_initializer(seed=1))

    return parameters

def create_dense_parameters(sizes,bias=True):
    parameters = {}
    if bias:
        for i, size in enumerate(sizes):
            parameters['W' + str(i)] = tf.get_variable('W' + str(i), shape=[size[0],size[1]],initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters['b' + str(i)] = tf.get_variable('b' + str(i), shape=[size[0],1], initializer=tf.zeros_initializer())
    else:
        for i, size in enumerate(sizes):
            parameters['W' + str(i)] = tf.get_variable('W' + str(i), shape=size,initializer=tf.contrib.layers.xavier_initializer(seed=1))

    return parameters

###################################################
"""DEFINING CONVOLUTION MODEL STEPS"""
###################################################
def conv2d(x,filter,strides,padding,bias):
    x = tf.nn.conv2d(x,filter,strides=[1,strides,strides,1],padding=padding)
    x = tf.nn.bias_add(x,bias)
    return tf.nn.relu(x)

def maxpool2d(x,k=2,pad='SAME'):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding=pad)

def avgpool2d(x,k=2,pad='SAME'):
    return tf.nn.avg_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding=pad)

def flatten_with_dense(x,W,b):
    x = tf.layers.flatten(x,name="flat_layer")
    s = tf.shape(x)
    layer_size=tf.shape(W)[0]
    W = tf.get_variable("W_flat",shape=[layer_size,s],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b = tf.get_variable("b_flat",shape=[layer_size,1])
    x = tf.add(tf.matmul(W,x),b)
    return x

def dense(x,W,b):
    x = tf.add(tf.matmul(W,x),b)
    return tf.nn.relu(x)

def dropout(x):
    return tf.nn.dropout(x,keep_prob=0.8)

def residual(x):
    pass

valid_steps = ['conv2d','max_pool','avg_pool','flatten','dense','dropout','residual']

def conv_net(X,steps,conv_parameters,dense_parameters):
    """
    :param X: Input tensor
    :param steps: list of steps for the convolutional network to take
    :param conv_parameters: the stored convolutional parameters
    :param dense_parameters:  the stored dense or fully connected network parameters
    :return: output result tensor. e.g. if Softmax with n_classes output would be tensor of size n_classes
    """
    X_in = X
    conv_indexer = 0
    conv_keys = list(conv_parameters.keys())

    dense_indexer = 0
    dense_keys = list(dense_parameters.keys())

    initial = 0

    if steps[0] == 'conv2d':
        initial = conv2d(X, conv_parameters[conv_keys[conv_indexer]], strides=1, padding="SAME",
                         bias=conv_parameters[conv_keys[conv_indexer + 1]])
        conv_indexer += 2
    elif steps[0] == 'flatten':
        initial = flatten(X)
    elif steps[0] == 'residual':
        initial = residual(X)

    for step in steps[1:]:
        if step == 'conv2d':
            initial = conv2d(initial,conv_parameters[conv_keys[conv_indexer]], strides=1, padding="SAME",
                         bias=conv_parameters[conv_keys[conv_indexer + 1]])
            conv_indexer += 2
        elif step == 'max_pool':
            initial = max_pool2d(initial)
        elif step == 'avg_pool':
            initial = avgpool2d(initial)
        elif step == 'flatten':
            initial = flatten_with_dense(initial,dense_parameters[dense_keys[dense_indexer]],dense_parameters[dense_keys[dense_indexer + 1]])
            dense_indexer += 2
        elif step == 'dense':
            initial = dense(initial, dense_parameters[dense_keys[dense_indexer]],
                            dense_parameters[dense_keys[dense_indexer + 1]])
            dense_indexer += 2
        elif step == 'dropout':
            initial = dropout(initial)
        elif step == 'residual':
            initial = residual(initial)

    return initial

def compute_cost(predictions,y,loss_type=None):

    logits = tf.transpose(predictions)
    labels = tf.transpose(y)

    if loss_type == "categorical":
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
    elif loss_type == "binary":
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    else:
        raise


def model(X_train,X_test,Y_train,Y_test,
          filter_sizes,layer_dims,conv_steps,lr=0.0001,
          epochs=1500,batch_size=16,print_cost=True):


    ops.reset_default_graph()
    tf.set_random_seed(1)

    costs = []

    n_h,n_w,m = X_train.shape
    n_y = Y_train.shape[0]

    ####CREATE PLACEHOLDERS TO FEED INTO TENSOR VERTEX####

    X,Y = create_placeholders(n_h,n_w,1,n_y)

    ####CREATE TENSOR PARAMETER VARIABLES TO FEED DATA INTO PARAMETERS####

    conv_parameters = create_conv_parameters(filter_sizes)
    dense_parameters = create_dense_parameters(layer_dims)

    ####FEED DATA TO PROPAGATE THROUGH THE TENSOR GRAPH####

    layer_out = conv_net(X,steps,conv_parameters,dense_parameters)

    ####CREATE COST VERTEX TO CALCULATE COST IN GRAPH####

    cost = compute_cost(layer_out,Y,loss_type="categorical")

    ####CREATE OPTIMIZER VERTEX TO FEED AND OPTIMIZE OFF OF ERRORS FOR LEARNING####

    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    init = tf.global_variables_initializer()

    correct_predictions = tf.equal(tf.argmax(layer_out), tf.argmax(Y))

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:\\Users\\Juan Felipe Zarco\\Documents\\ML Projects\\Tensorflow Practice\\LogDir")

    with tf.Session() as sess:

        sess.run(init)
        ####TRAINING LOOP####
        for epoch in range(epochs):
            epoch = epoch + 1
            epoch_cost = 0
            mb_costs = []

            total_batches = int(m/batch_size)

            ####SHUFFLE TRAINING DATA#####

            shuffler = np.random.permutation(m)

            X_shuffled = X_train[:,shuffler]
            Y_shuffled = Y_train[:,shuffler].reshape((n_y,m))

            if m % batch_size == 0:
                for batch_num in range(total_batches):
                    X_batch = X_shuffled[:,batch_num*batch_size:batch_num*batch_size + batch_size]
                    Y_batch = Y_shuffled[:,batch_num*batch_size:batch_num*batch_size + batch_size]

                    _,mb_cost = sess.run([opt,cost],feed_dict={X:X_batch,Y:Y_batch})
                    mb_costs.append(mb_cost)

                    epoch_cost += mb_cost/total_batches
                plt.plot(np.squeeze(mb_costs))
                plt.title("Minibatch Costs")
                plt.show()
            else:
                for batch_num in range(total_batches):
                    try:
                        X_batch = X_shuffled[:,batch_num*batch_size:batch_num*batch_size + batch_size]
                        Y_batch = Y_shuffled[:,batch_num*batch_size:batch_num*batch_size + batch_size]
                    except:
                        try:
                            X_batch = X_shuffled[:, batch_num * batch_size:batch_num * batch_size + batch_size]
                            Y_batch = Y_shuffled[:, batch_num * batch_size:batch_num * batch_size + batch_size]
                        except:
                            print("Error slicing minibatches")

                    _,mb_cost = sess.run([opt,cost],feed_dict={X:X_batch,Y:Y_batch})

                    epoch_cost += mb_cost/total_batches

            summary = sess.run(merged, feed_dict={X: X_test, Y: Y_test})
            writer.add_summary(summary,epoch)

            if print_cost == True:
                print("Total Cost at Epoch {epoch}: {cost}".format(epoch=epoch,cost=epoch_cost))
            costs.append(epoch_cost)

        ####AFTER TRAINING SAVE PARAMETERS AND BUILD ACCURACY EVALUATOR####

        parameters = sess.run(parameters)

        writer.add_graph(sess.graph)

        print("Train Accuracy: ",accuracy.eval({X:X_train,Y:Y_train}))
        print("Test Accuracy: ", accuracy.eval({X:X_test,Y:Y_test}))

        return parameters,costs


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train,y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train/255.0, x_test/255.0
    x_train = x_train.T
    x_test = x_test.T

    y_train = one_hot_mat(y_train,10)
    y_test = one_hot_mat(y_test,10)


    filter_sizes = [32,64,128]
    steps = ['conv2d','max_pool','conv2d','max_pool','conv2d','max_pool','flatten','dense','dropout']
    layer_dims = [(1028,None),(512,1028),(256,128),(10,128)]

    #x,y = create_placeholders(28,28,3,10)
    #print(x)
    #print(y)
    #f = tf.get_variable('F0',shape=[10,10,3,32],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    #b = tf.get_variable('b0', shape=32, initializer=tf.zeros_initializer())

    #W = tf.Variable(tf.random_normal([5, 5, 3, 32]))
    #b = tf.Variable(tf.random_normal([32]))

    #x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #print(x)
    #x = tf.nn.bias_add(x, b)
    #print(x)

    #conv = conv2d(x,f,strides=1,padding="SAME",bias=b)
    #print(conv)

    params,costs = model(x_train,x_test,y_train,y_test,filter_sizes,layer_dims,steps,epochs=15,batch_size=150)

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Cost per iteration")
    plt.show()
