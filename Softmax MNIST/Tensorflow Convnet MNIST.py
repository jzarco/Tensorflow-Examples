import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

x_dim = None

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

    :param n_h: pixel height dimension
    :param n_w: pixel width dimension
    :param n_c: RGB channel dimension
    :param n_y: num classes/dimension for y
    :return: tensorflow placeholder variables
    """

    x = tf.placeholder(tf.float32,shape=[None,n_h,n_w,n_c])
    y = tf.placeholder(tf.float32,shape=[None,n_y])

    return x,y

def create_filter(f_height,f_width,n_input,n_filters):

    filter = tf.Variable(tf.random_normal([f_height,f_width,n_input,n_filters]))

    return filter

def conv2d(x,,filter,strides,padding,bias):
    x = tf.nn.conv2d(x,filter,strides=[1,strides,strides,1],padding=padding)
    x = tf.nn.bias_add(x,bias)

def forward_propagate(X,parameters):
    """

    :param X: feature matrix to propagate through
    :param parameters: tensorflow parameter variable dictionary
    :return: forward propagate output WITHOUT passing activation function on last layer.
    """
    ####First try block assumes there is bias parameters. Error will incur if there is none and will try without bias.
    try:
        temp = list(parameters.keys())
        param_list = [(temp[i],temp[i+1]) for i in range(0,len(temp),2)] #Creating pairwise zip object for each weight and bias
        holder = tf.add(tf.matmul(parameters[param_list[0][0]],X),parameters[param_list[0][1]])
        holder = tf.nn.relu(holder)
        for W,b in param_list[1:-1]:
            holder = tf.add(tf.matmul(parameters[W],holder),parameters[b])
            holder = tf.nn.relu(holder)

        holder = tf.add(tf.matmul(parameters[param_list[-1][0]],holder),parameters[param_list[-1][1]])
        return holder
    except:
        try:
            param_list = list(parameters.keys())
            holder = tf.matmul(parameters[param_list[0]],X)
            holder = tf.nn.relu(holder)
            for W in param_list[1:-1]:
                holder = tf.matmul(parameters[W],holder)
                holder = tf.nn.relu(holder)

            holder = tf.add(tf.matmul(parameters[param_list[-1][0]], holder), parameters[param_list[-1][1]])
            return holder
        except:
            "Failed operation running 'forward_propagate' function"


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
          layer_dims=[],lr=0.0001,
          epochs=1500,batch_size=16,print_cost=True):


    ops.reset_default_graph()
    tf.set_random_seed(123)

    costs = []

    n_x,m = X_train.shape
    n_y = Y_train.shape[0]

    ####CREATE PLACEHOLDERS TO FEED INTO TENSOR VERTEX####

    X,Y = create_placeholders(n_x,n_y)

    ####CREATE TENSOR PARAMETER VARIABLES TO FEED DATA INTO PARAMETERS####

    parameters = create_parameters(parameter_dimensions=layer_dims)

    ####FEED DATA TO PROPAGATE THROUGH THE TENSOR GRAPH####

    layer_out = forward_propagate(X,parameters)

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
                #plt.plot(np.squeeze(mb_costs))
                #plt.title("Minibatch Costs")
                #plt.show()
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
    x_train = x_train.reshape((60000,28*28)).T
    x_test = x_test.reshape((10000,28*28)).T

    y_train = one_hot_mat(y_train,10)
    y_test = one_hot_mat(y_test,10)

    params,costs = model(x_train,x_test,y_train,y_test,[(256,784),(128,256),(10,128)],epochs=15,batch_size=100)

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')
    plt.xlabel('iterations per 5')
    plt.title("Cost per iteration")
    plt.show()
