import scipy.io
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

class configurations:
    width = 500
    height = 333
    channels = 3

def load_vgg_model(path):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """

    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, configurations.height, configurations.width, configurations.channels)),
                                 dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph


def generate_noise_image(content_image, noise_ratio=0.6):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20,
                                    (1, configurations.height, configurations.width, configurations.channels)).astype(
        'float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input of VGG16
    image = image - np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    return image


def save_image(path, image):
    # Un-normalize the image so that it looks good
    image = image + np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


#########################################################################
"""DEFINE TENSOR GRAPH FUNCTIONS"""
#########################################################################

def compute_content_cost(i_C,i_G):
    i_C = tf.convert_to_tensor(i_C,tf.float32)
    i_G = tf.convert_to_tensor(i_G,tf.float32)
    m, n_h, n_w, n_c = i_C.get_shape().as_list()
    #m, n_h, n_w, n_c = i_C.shape

    iC_unrolled = tf.reshape(i_C,[n_h*n_w,n_c])
    iG_unrolled = tf.reshape(i_G,[n_h*n_w,n_c])

    J_content = tf.reduce_sum(tf.squared_difference(iC_unrolled,iG_unrolled))/(4*n_h*n_w*n_c)

    return J_content


def gram_matrix(A):

    gram = tf.matmul(A,tf.transpose(A))

    return gram

def style_cost(i_S,i_G):

    m, n_h, n_w, n_c = i_G.get_shape().as_list()
    #m, n_h, n_w, n_c = i_G.shape

    iS = tf.transpose(tf.reshape(i_S,[n_h*n_w,n_c]))
    iG = tf.transpose(tf.reshape(i_G,[n_h*n_w,n_c]))

    gram_style = gram_matrix(iS)
    gram_generated = gram_matrix(iG)

    J_style = tf.convert_to_tensor(tf.reduce_sum(tf.squared_difference(gram_style,gram_generated))/(4*(n_h*n_h*n_w*n_w*n_c*n_c)),tf.float32)

    return J_style


def style_cost_layers(model,sess,style_weights):

    J_style = 0

    for layer_name, coeff in style_weights:

        out = model[layer_name]

        img_style = sess.run(out)

        img_gen = out

        J_style_layer = style_cost(img_style,img_gen)

        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content,J_style,alpha=10,beta=40):

    J = alpha*J_content + beta*J_style

    return J


def nst_nn(model,input_image,style_image,style_weights,num_iterations=300):
    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    c_image = scipy.misc.imread(input_image)
    c_image = reshape_and_normalize_image(c_image)

    s_image = scipy.misc.imread(style_image)
    s_image = reshape_and_normalize_image(s_image)

    gen_image = generate_noise_image(c_image)
    imshow(gen_image[0])

    model = load_vgg_model(model)
    print("Model Loaded\n")

    sess.run(model['input'].assign(c_image))
    out = tf.convert_to_tensor(model['conv4_2'],tf.float32)

    i_C = tf.convert_to_tensor(sess.run(out),tf.float32)
    i_G = tf.convert_to_tensor(out,tf.float32)

    J_content = compute_content_cost(i_C,i_G)

    sess.run(model['input'].assign(s_image))

    def style_cost_layers(model, style_weights):

        J_style = 0

        for layer_name, coeff in style_weights:
            out = model[layer_name]

            img_style = tf.convert_to_tensor(sess.run(out),tf.float32)

            img_gen = tf.convert_to_tensor(out,tf.float32)

            J_style_layer = style_cost(img_style, img_gen)

            J_style += coeff * J_style_layer

        return J_style

    J_style = style_cost_layers(model,style_weights)

    J = total_cost(J_content,J_style)

    opt = tf.train.AdamOptimizer(2.0)
    train_step = opt.minimize(J)

    init = tf.global_variables_initializer()
    sess.run(init)

    generated_image = sess.run(model['input'].assign(gen_image))

    for i in range(num_iterations):

        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if  i%10 == 0:
            Jt, Jc, Js = sess.run([J,J_content,J_style])
            print("Iteration {}: ".format(i))
            print("total cost = {}".format(Jt))
            print("content cost = {}".format(Jc))
            print("style cost = {}".format(Js))

            save_image("output/generated_image.png",generated_image)

    save_image("output/generated_image.jpg", generated_image)

    return generated_image

