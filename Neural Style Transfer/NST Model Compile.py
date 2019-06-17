import tensorflow as tf
from neural_style_utils import *
import scipy
import random

DEFAULT_STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def rand_weights(n_layers):
    pass


STYLE_LAYERS = [
    ('conv1_1', random.uniform(0,1/11)),
    ('conv1_2',random.uniform(0,1/11)),
    ('conv2_1', random.uniform(0,1/11)),
    ('conv2_2',random.uniform(0,1/11)),
    ('conv3_1', random.uniform(0,1/11)),
    ('conv3_4',random.uniform(0,1/11)),
    ('conv4_1', random.uniform(0,1/11)),
    ('conv4_4',random.uniform(0,1/11)),
    ('conv5_1', random.uniform(0,1/11)),
    ('conv5_4',random.uniform(0,1/11))]

c_img = "D:\\Image Data\\Artistic Images\\bridge.jpg"
s_img = "D:\\Image Data\\Artistic Images\\california-impressionism-2.jpg"

model_path = "D:\\Models\\NST VGG Model\\imagenet-vgg-verydeep-19.mat"

g_image = nst_nn(model_path, c_img, s_img, STYLE_LAYERS)

#imshow("output/generated_image.jpg")