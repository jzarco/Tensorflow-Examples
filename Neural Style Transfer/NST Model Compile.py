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

#model = load_vgg_model("D:\\Models\\NST VGG Model\\imagenet-vgg-verydeep-19.mat")

#c_image = scipy.misc.imread(c_img)
#c_image = reshape_and_normalize_image(c_image)

#print("Content shape: ",c_image.shape)

#s_image = scipy.misc.imread(s_img)
#s_image = reshape_and_normalize_image(s_image)

#print("Style shape: ", s_image.shape)

#gen_image = generate_noise_image(c_image)
#imshow(gen_image[0])

g_image = nst_nn(model_path, c_img, s_img, STYLE_LAYERS)

#imshow("output/generated_image.jpg")