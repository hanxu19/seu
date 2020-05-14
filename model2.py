from __future__ import print_function
# model Reuse model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Use other GPU

import numpy as np
import tensorflow.keras.backend as K
import time
import tensorflow as tf
import xmltodict
import glob
import os
import sys
import math
import keras.losses
import shutil  

from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.layers import merge, Input, concatenate
from keras.layers import Dense, Activation, Flatten, Reshape, MaxPooling1D
from keras.layers import MaxPooling2D, ZeroPadding2D, AveragePooling2D, Conv2D, MaxPool2D, Cropping2D, TimeDistributed
from keras.layers import SimpleRNN
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from skimage.io import imread, imshow
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
from keras.layers.core import Layer


if (sys.platform == 'linux'):
    imgpath = r'/home/hx/hanxu/cu/*.jpg'
    annotations_paths = r'/home/hx/hanxu/cu/*.xml'
    modelsavepath = r'/home/hx/hanxu/cu.h5'
else:
    imgpath =  r'C:/Users/gngn39\Desktop/vi/Image/foggy/*.jpg'
    annotations_paths =  r'C:/Users/gngn39\Desktop/vi/Image/foggy/*.xml'
    modelsavepath = r'C:/Users/gngn39/Desktop/vi/cu.h5'

TIME_STEPS = 64
INPUT_SIZE = 64
# img = Image.open(imgpath)
IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3

# process data
images = []

image_paths = sorted(glob.glob(imgpath))
for imagefile in image_paths:
    #print (imagefile)
    image = Image.open(imagefile).resize((227, 227))
    image = np.asarray(image) / 255.0
    images.append(image)

fog_image_paths = sorted(glob.glob(r'/home/hx/hanxu/foggy/*.jpg'))
for fogimagefile in fog_image_paths:
    #print (fogimagefile)
    image = Image.open(fogimagefile).resize((227, 227))
    image = np.asarray(image) / 255.0
    images.append(image)

bboxes = []

annotations_file = sorted(glob.glob(annotations_paths))
for xmlfile in annotations_file:
    #print (xmlfile)
    x = xmltodict.parse(open(xmlfile, 'rb'))
    bndbox = x['annotation']['object']['bndbox']
    bndbox = np.array([float(bndbox['ymin']), float(
        bndbox['xmin']), float(bndbox['ymax']), float(bndbox['xmax'])])

    bndbox2 = [None] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]
    bndbox2 = np.array(bndbox2) / 227
    bboxes.append(bndbox2)

fog_xml_file = sorted(glob.glob(r'/home/hx/hanxu/foggy/*.xml'))
for fog_xmlfile in fog_xml_file:
    #print (fog_xmlfile)
    x = xmltodict.parse(open(fog_xmlfile, 'rb'))
    bndbox = x['bndbox']
    bndbox = np.array([ float(bndbox[ 'top' ]) , float(bndbox[ 'left' ]) , float(bndbox[ 'bottom' ]) , float(bndbox[ 'right' ]) ])
    bndbox2 = [None] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]
    bndbox2 = np.array(bndbox2)
    bboxes.append(bndbox2)

boxes = np.array(bboxes)

print ("boxes.shape:" , boxes.shape)

np.set_printoptions(threshold=sys.maxsize)
print ("boxes:" ,boxes)

Y = np.array(boxes)
X = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# layer

class croprt(Layer):

    def __init__(self, **kwargs):
        super(croprt, self).__init__(**kwargs)

    def call(self, inputs):
        image_meta = inputs[0]
        boxes = inputs[1]

        # top = tf.slice(boxes, [0,0], [-1, 1])
        # left = tf.slice(boxes, [0,1], [-1, 1])
        # bottom = tf.slice(boxes, [0,2], [-1, 1])
        # right = tf.slice(boxes, [0,3], [-1, 1])
        # box = tf.concat([top, left, bottom, right], axis=1)
        # boxes2 = tf.stop_gradient(box)

        # print (image_meta)
        # print (boxes)
        # img0 = tf.slice(image_meta,[0,0,0,0],[1,-1,-1,-1])
        # box0 = tf.slice(boxes,[0,0],[1,-1])
        # img1 = tf.slice(image_meta,[1,0,0,0],[1,-1,-1,-1])
        # box1 = tf.slice(boxes,[1,0],[1,-1])
        # print (img0)
        # print (box0)
        # print (img1)
        # print (box1)
        # croped_img0 = tf.image.crop_and_resize(img0, box0, [0], (227, 227))
        # croped_img1 = tf.image.crop_and_resize(img1, box1, [0], (227, 227))

        # print (croped_img0)
        # print (croped_img1)
        # img = tf.concat([croped_img0, croped_img1], axis=0)
        # print (img)
        # image_meta = tf.stop_gradient(image_meta)
        # boxes = tf.stop_gradient(boxes)
        pooled = []
        pooled.append(tf.image.crop_and_resize(image_meta, boxes, [0], (227, 227)))
        # img = tf.image.crop_and_resize(image_meta, boxes, [0], (227, 227))
        pooled = tf.concat(pooled, axis=0)
        print (pooled)
        return pooled

    def compute_output_shape(self, input_shape):
        
        return (None, 227, 227, 3)


def gs_block_layer1(img, layerNumber):

    # if first RNN layer, don't crop
    # inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3)) #channel 3
    c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid',
                name='conv1' + '_rnn_' + str(layerNumber))(img)
    c2 = BatchNormalization(name='conv1bn' + '_rnn_' + str(layerNumber))(c1)
    c3 = MaxPool2D((3, 3), strides=2, padding='valid')(c2)

    c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv2' + '_rnn_' + str(layerNumber))(c3)
    c5 = BatchNormalization(name='conv2bn' + '_rnn_' + str(layerNumber))(c4)
    c6 = MaxPool2D((3, 3), strides=2, padding='valid')(c5)

    c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv3' + '_rnn_' + str(layerNumber))(c6)
    c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv4' + '_rnn_' + str(layerNumber))(c7)
    c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv5' + '_rnn_' + str(layerNumber))(c8)
    c10 = MaxPool2D((3, 3), strides=2, padding='valid')(c9)

    c11 = Flatten()(c10)
    c12 = Dense(4096, activation='relu', name='fc4096' + '_rnn_' +
                str(layerNumber))(c11)  # 2048 in paper, fc 1
    c13 = Dropout(0.5)(c12)
    c14 = Lambda(lambda c13: K.expand_dims(c13, axis=1))(c13)

    return c14


def gs_block(img, rt, layerNumber):
    # #if first RNN layer, don't crop
    # #top left bottom right
    # top = Lambda(lambda x: tf.slice(x, [0, 0], [-1, 1]))(rt)
    # left = Lambda(lambda x: tf.slice(x, [0, 1], [-1, 1]))(rt)
    # bottom = Lambda(lambda x: tf.slice(x, [0, 2], [-1, 1]))(rt)
    # right = Lambda(lambda x: tf.slice(x, [0, 3], [-1, 1]))(rt)
    # #boxes = K.concatenate([y1, x1, y2, x2], axis=1)
    # boxes = Lambda(lambda a: tf.concat(a, axis=1))([top, left, bottom, right])

    # #cropped = tf.image.crop_and_resize(img,[[0.5,0.6,0.9,0.8]],box_ind=[0],crop_size=(100,100))
    # cropped = Lambda(tf.image.crop_and_resize,output_shape=(227, 227, 3),arguments={'boxes':boxes,\
    #     'box_ind':[0],'crop_size':(227,227)})(img) # box_ind number = batch number
    cropped = croprt()([input, rt])

    # inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3)) #channel 3
    c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid',
                name='conv1' + '_rnn_' + str(layerNumber))(cropped)
    c2 = BatchNormalization(name='conv1bn' + '_rnn_' + str(layerNumber))(c1)
    c3 = MaxPool2D((3, 3), strides=2, padding='valid')(c2)

    c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv2' + '_rnn_' + str(layerNumber))(c3)
    c5 = BatchNormalization(name='conv2bn' + '_rnn_' + str(layerNumber))(c4)
    c6 = MaxPool2D((3, 3), strides=2, padding='valid')(c5)

    c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv3' + '_rnn_' + str(layerNumber))(c6)
    c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv4' + '_rnn_' + str(layerNumber))(c7)
    c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                name='conv5' + '_rnn_' + str(layerNumber))(c8)
    c10 = MaxPool2D((3, 3), strides=2, padding='valid')(c9)

    c11 = Flatten(name = "Flatten"+ str(layerNumber) )(c10)
    c12 = Dense(4096, activation='relu', name='fc4096' + '_rnn_' +
                str(layerNumber))(c11)  # 2048 in paper, fc 1
    c13 = Dropout(0.5)(c12)
    c14 = Lambda(lambda c13: K.expand_dims(c13, axis=1))(c13)
    #alexnet = Model(inputs , c13)
    #alexnet.load_weights('C:\Users\gngn39\Desktop/vi\model-2.h5' , by_name = True)
    #features = alexnet.predict(image_data)

    #print (features.shape)
    #np.savetxt('C:/Users/gngn39/Desktop/vi/features2.txt', features ,fmt='%s')
    # return features
    return c14


def gr_block(ht):
    dense128 = Dense(128)(ht)
    rt = Dense(4)(dense128)
    #output = Flatten()(output)
    return rt

# inputs_gr = Input((256,))
# dense128 = Dense(128)(inputs_gr)
# rt = Dense(4)(dense128)
# gr_block = Model(inputs_gr,rt,name='gr_block')


input = Input(name='TOTAL_input', shape=(227, 227, 3))
# rnn
vt1 = gs_block_layer1(input, 1)
ht1, state_1 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block1')(vt1)
# F1 = merge([state_1, c1_r1_Flatten], mode='sum') #F1

rt2 = gr_block(ht1)
vt2 = gs_block(input, rt2, 2)
ht2, state_2 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block2')(vt2, initial_state=state_1)

rt3 = gr_block(ht2)
vt3 = gs_block(input, rt3, 3)
ht3, state_3 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block3')(vt3, initial_state=state_2)

rt4 = gr_block(ht3)
vt4 = gs_block(input, rt4, 4)
ht4, state_4 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block4')(vt4, initial_state=state_3)
# F2 = merge([state_4, c2_r4_Flatten], mode='sum') #F2

rt5 = gr_block(ht4)
vt5 = gs_block(input, rt5, 5)
ht5, state_5 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block5')(vt5, initial_state=state_4)

rt6 = gr_block(ht5)
vt6 = gs_block(input, rt6, 6)
ht6, state_6 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block6')(vt6, initial_state=state_5)

rt7 = gr_block(ht6)
vt7 = gs_block(input, rt7, 7)
ht7, state_7 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block7')(vt7, initial_state=state_6)
# F3 = merge([state_7, c3_r7_Flatten], mode='sum') #F3

rt8 = gr_block(ht7)
vt8 = gs_block(input, rt8, 8)
ht8, state_8 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block8')(vt8, initial_state=state_7)

rt9 = gr_block(ht8)
vt9 = gs_block(input, rt9, 9)
ht9, state_9 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                         batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block9')(vt9, initial_state=state_8)

rt10 = gr_block(ht9)
vt10 = gs_block(input, rt10, 10)
ht10, state_10 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block10')(vt10, initial_state=state_9)
# F4 = merge([state_10, c4_r10_Flatten], mode='sum') #F4

rt11 = gr_block(ht10)
vt11 = gs_block(input, rt11, 11)
ht11, state_11 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block11')(vt11, initial_state=state_10)

rt12 = gr_block(ht11)
vt12 = gs_block(input, rt12, 12)
ht12, state_12 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block12')(vt12, initial_state=state_11)

rt13 = gr_block(ht12)
vt13 = gs_block(input, rt13, 13)
ht13, state_13 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block13')(vt13, initial_state=state_12)
# F5 = merge([state_13, c5_r13_Flatten], mode='sum') #F5

rt14 = gr_block(ht13)
vt14 = gs_block(input, rt14, 14)
ht14, state_14 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block14')(vt14, initial_state=state_13)

rt15 = gr_block(ht14)
vt15 = gs_block(input, rt15, 15)
ht15, state_15 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block15')(vt15, initial_state=state_14)

rt16 = gr_block(ht15)
vt16 = gs_block(input, rt16, 16)
ht16, state_16 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block16')(vt16, initial_state=state_15)
# F6 = merge([state_16, c6_r16_Flatten], mode='sum') #F6

rt17 = gr_block(ht16)
vt17 = gs_block(input, rt17, 17)
ht17, state_17 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block17')(vt17, initial_state=state_16)

rt18 = gr_block(ht17)
vt18 = gs_block(input, rt18, 18)
ht18, state_18 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=True,
                           batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block18')(vt18, initial_state=state_17)

rt19 = gr_block(ht18)
vt19 = gs_block(input, rt19, 19)
ht19 = SimpleRNN(units=256, activation='relu', return_sequences=False, return_state=False,
                 batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), name='gh_block19')(vt19, initial_state=state_18)
# F7 = merge([ht19, c7_r19_Flatten], mode='sum') #F7
# rnn_outputs = F7
# dense64 = Dense(64)(ht2)
# dense16 = Dense(16)(dense64)
dense128 = Dense(128)(ht19)
dense_output = Dense(4, name='dense_output')(dense128)
#outputs = concatenate([cnn_outputs , rnn_outputs])


# def calculate_iou(target_boxes, pred_boxes):
#     yA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])  # yA
#     xA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])  # xA
#     yB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])  # yB
#     xB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])  # xB
#     interArea = K.maximum(0.0, yB - yA) * K.maximum(0.0, xB - xA)
#     boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * \
#         (target_boxes[..., 3] - target_boxes[..., 1])
#     boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * \
#         (pred_boxes[..., 3] - pred_boxes[..., 1])
#     iou = interArea / (boxAArea + boxBArea - interArea)
#     return iou

def calculate_iou(boxes1, boxes2):
    '''
    计算giou = iou - (C-AUB)/C
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    '''
    boxes1_x0y0x1y1 = boxes1
    boxes2_x0y0x1y1 = boxes2
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    return giou


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    # return (1 - iou)
    return mse + (1 - iou)

def mse(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    return mse
def predict0 (y_true,y_pred):
    return y_pred[0][0]
def predict1 (y_true,y_pred):
    return y_pred[0][1]
def predict2 (y_true,y_pred):
    return y_pred[0][2]
def predict3 (y_true,y_pred):
    return y_pred[0][3]

def true0 (y_true,y_pred):
    return y_true[0][0]
def true1 (y_true,y_pred):
    return y_true[0][1]
def true2 (y_true,y_pred):
    return y_true[0][2]
def true3 (y_true,y_pred):
    return y_true[0][3]

model = Model(inputs=[input], outputs=[rt2,rt3,rt4,rt5,rt6,rt7,rt8,rt9,rt10,rt11,rt12,rt13,rt14,rt15,rt16,rt17,rt18,rt19,dense_output])

#metrics= [calculate_iou,mse,predict0,predict1,predict2,predict3,true0,true1,true2,true3]
#loss=[custom_loss_rt2, custom_loss_rt3, custom_loss_rt4, custom_loss_rt5, custom_loss_rt6, custom_loss_rt7, custom_loss_rt8, custom_loss_rt9, custom_loss_rt10, custom_loss_rt11, custom_loss_rt12, custom_loss_rt13, custom_loss_rt14, custom_loss_rt15, custom_loss_rt16, custom_loss_rt17, custom_loss_rt18, custom_loss_rt19,custom_loss]
model.compile(optimizer=Adam( lr=0.0001 ), 

loss=[custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss,custom_loss], 

loss_weights=[1/19,2/19,3/19,4/19,5/19,6/19,7/19,8/19,9/19,10/19,11/19,12/19,13/19,14/19,15/19,16/19,17/19,18/19,19/19],

metrics= [calculate_iou,mse,predict0,predict1,predict2,predict3,true0,true1,true2,true3] )
#model.summary()

print ("y_train :" , y_train.shape )
train_number = y_train.shape[0]
print ("y_test :" , y_test.shape )
test_number = y_test.shape[0]

# train or predict
Next = 'train'
if (Next == 'train'):
    
    ran = 3

    y_train_2 = np.ones([train_number,4]) 
    y_train_2[ : , 0:2] = y_train[ : , 0:2] * 1/19
    y_train_2[ : , 2:4] = y_train[ : , 2:4] + (y_train_2[ : , 2:4] - y_train[ : , 2:4])*18/19
    print ("y_train_2[" + str(ran) + "]:" , y_train_2[ran])

    y_train_3 = np.ones([train_number,4]) 
    y_train_3[ : , 0:2] = y_train[ : , 0:2] * 2/19
    y_train_3[ : , 2:4] = y_train[ : , 2:4] + (y_train_3[ : , 2:4] - y_train[ : , 2:4])*17/19
    print ("y_train_3[" + str(ran) + "]:" , y_train_3[ran])

    y_train_4 = np.ones([train_number,4]) 
    y_train_4[ : , 0:2] = y_train[ : , 0:2] * 3/19
    y_train_4[ : , 2:4] = y_train[ : , 2:4] + (y_train_4[ : , 2:4] - y_train[ : , 2:4])*16/19
    print ("y_train_4[" + str(ran) + "]:" , y_train_4[ran])

    y_train_5 = np.ones([train_number,4]) 
    y_train_5[ : , 0:2] = y_train[ : , 0:2] * 4/19
    y_train_5[ : , 2:4] = y_train[ : , 2:4] + (y_train_5[ : , 2:4] - y_train[ : , 2:4])*15/19
    print ("y_train_5[" + str(ran) + "]:" , y_train_5[ran])

    y_train_6 = np.ones([train_number,4]) 
    y_train_6[ : , 0:2] = y_train[ : , 0:2] * 5/19
    y_train_6[ : , 2:4] = y_train[ : , 2:4] + (y_train_6[ : , 2:4] - y_train[ : , 2:4])*14/19
    print ("y_train_6[" + str(ran) + "]:" , y_train_6[ran])

    y_train_7 = np.ones([train_number,4]) 
    y_train_7[ : , 0:2] = y_train[ : , 0:2] * 6/19
    y_train_7[ : , 2:4] = y_train[ : , 2:4] + (y_train_7[ : , 2:4] - y_train[ : , 2:4])*13/19
    print ("y_train_7[" + str(ran) + "]:" , y_train_7[ran])

    y_train_8 = np.ones([train_number,4]) 
    y_train_8[ : , 0:2] = y_train[ : , 0:2] * 7/19
    y_train_8[ : , 2:4] = y_train[ : , 2:4] + (y_train_8[ : , 2:4] - y_train[ : , 2:4])*12/19
    print ("y_train_8[" + str(ran) + "]:" , y_train_8[ran])

    y_train_9 = np.ones([train_number,4]) 
    y_train_9[ : , 0:2] = y_train[ : , 0:2] * 8/19
    y_train_9[ : , 2:4] = y_train[ : , 2:4] + (y_train_9[ : , 2:4] - y_train[ : , 2:4])*11/19
    print ("y_train_9[" + str(ran) + "]:" , y_train_9[ran])

    y_train_10 = np.ones([train_number,4]) 
    y_train_10[ : , 0:2] = y_train[ : , 0:2] * 9/19
    y_train_10[ : , 2:4] = y_train[ : , 2:4] + (y_train_10[ : , 2:4] - y_train[ : , 2:4])*10/19
    print ("y_train_10[" + str(ran) + "]:" , y_train_10[ran])

    y_train_11 = np.ones([train_number,4]) 
    y_train_11[ : , 0:2] = y_train[ : , 0:2] * 10/19
    y_train_11[ : , 2:4] = y_train[ : , 2:4] + (y_train_11[ : , 2:4] - y_train[ : , 2:4])*9/19
    print ("y_train_11[" + str(ran) + "]:" , y_train_11[ran])

    y_train_12 = np.ones([train_number,4]) 
    y_train_12[ : , 0:2] = y_train[ : , 0:2] * 11/19
    y_train_12[ : , 2:4] = y_train[ : , 2:4] + (y_train_12[ : , 2:4] - y_train[ : , 2:4])*8/19
    print ("y_train_12[" + str(ran) + "]:" , y_train_12[ran])

    y_train_13 = np.ones([train_number,4]) 
    y_train_13[ : , 0:2] = y_train[ : , 0:2] * 12/19
    y_train_13[ : , 2:4] = y_train[ : , 2:4] + (y_train_13[ : , 2:4] - y_train[ : , 2:4])*7/19
    print ("y_train_13[" + str(ran) + "]:" , y_train_13[ran])

    y_train_14 = np.ones([train_number,4]) 
    y_train_14[ : , 0:2] = y_train[ : , 0:2] * 13/19
    y_train_14[ : , 2:4] = y_train[ : , 2:4] + (y_train_14[ : , 2:4] - y_train[ : , 2:4])*6/19
    print ("y_train_14[" + str(ran) + "]:" , y_train_14[ran])

    y_train_15 = np.ones([train_number,4]) 
    y_train_15[ : , 0:2] = y_train[ : , 0:2] * 14/19
    y_train_15[ : , 2:4] = y_train[ : , 2:4] + (y_train_15[ : , 2:4] - y_train[ : , 2:4])*5/19
    print ("y_train_15[" + str(ran) + "]:" , y_train_15[ran])

    y_train_16 = np.ones([train_number,4]) 
    y_train_16[ : , 0:2] = y_train[ : , 0:2] * 15/19
    y_train_16[ : , 2:4] = y_train[ : , 2:4] + (y_train_16[ : , 2:4] - y_train[ : , 2:4])*4/19
    print ("y_train_16[" + str(ran) + "]:" , y_train_16[ran])

    y_train_17 = np.ones([train_number,4]) 
    y_train_17[ : , 0:2] = y_train[ : , 0:2] * 16/19
    y_train_17[ : , 2:4] = y_train[ : , 2:4] + (y_train_17[ : , 2:4] - y_train[ : , 2:4])*3/19
    print ("y_train_17[" + str(ran) + "]:" , y_train_17[ran])

    y_train_18 = np.ones([train_number,4]) 
    y_train_18[ : , 0:2] = y_train[ : , 0:2] * 17/19
    y_train_18[ : , 2:4] = y_train[ : , 2:4] + (y_train_18[ : , 2:4] - y_train[ : , 2:4])*2/19
    print ("y_train_18[" + str(ran) + "]:" , y_train_18[ran])

    y_train_19 = np.ones([train_number,4]) 
    y_train_19[ : , 0:2] = y_train[ : , 0:2] * 18/19
    y_train_19[ : , 2:4] = y_train[ : , 2:4] + (y_train_19[ : , 2:4] - y_train[ : , 2:4])*1/19
    print ("y_train_19[" + str(ran) + "]:" , y_train_19[ran])

    print ("y_train[" + str(ran) + "]:" , y_train[ran])
    

    y_test_2 = np.ones([test_number,4])
    y_test_2[ : , 0:2] = y_test[ : , 0:2] * 1/19
    y_test_2[ : , 2:4] = y_test[ : , 2:4] + (y_test_2[ : , 2:4] - y_test[ : , 2:4])*18/19
    print ("y_test_2[" + str(ran) + "]:" , y_test_2[ran])

    y_test_3 = np.ones([test_number,4])
    y_test_3[ : , 0:2] = y_test[ : , 0:2] * 2/19
    y_test_3[ : , 2:4] = y_test[ : , 2:4] + (y_test_3[ : , 2:4] - y_test[ : , 2:4])*17/19
    print ("y_test_3[" + str(ran) + "]:" , y_test_3[ran])

    y_test_4 = np.ones([test_number,4])
    y_test_4[ : , 0:2] = y_test[ : , 0:2] * 3/19
    y_test_4[ : , 2:4] = y_test[ : , 2:4] + (y_test_4[ : , 2:4] - y_test[ : , 2:4])*16/19
    print ("y_test_4[" + str(ran) + "]:" , y_test_4[ran])

    y_test_5 = np.ones([test_number,4])
    y_test_5[ : , 0:2] = y_test[ : , 0:2] * 4/19
    y_test_5[ : , 2:4] = y_test[ : , 2:4] + (y_test_5[ : , 2:4] - y_test[ : , 2:4])*15/19
    print ("y_test_5[" + str(ran) + "]:" , y_test_5[ran])

    y_test_6 = np.ones([test_number,4])
    y_test_6[ : , 0:2] = y_test[ : , 0:2] * 5/19
    y_test_6[ : , 2:4] = y_test[ : , 2:4] + (y_test_6[ : , 2:4] - y_test[ : , 2:4])*14/19
    print ("y_test_6[" + str(ran) + "]:" , y_test_6[ran])

    y_test_7 = np.ones([test_number,4])
    y_test_7[ : , 0:2] = y_test[ : , 0:2] * 6/19
    y_test_7[ : , 2:4] = y_test[ : , 2:4] + (y_test_7[ : , 2:4] - y_test[ : , 2:4])*13/19
    print ("y_test_7[" + str(ran) + "]:" , y_test_7[ran])

    y_test_8 = np.ones([test_number,4])
    y_test_8[ : , 0:2] = y_test[ : , 0:2] * 7/19
    y_test_8[ : , 2:4] = y_test[ : , 2:4] + (y_test_8[ : , 2:4] - y_test[ : , 2:4])*12/19
    print ("y_test_8[" + str(ran) + "]:" , y_test_8[ran])

    y_test_9 = np.ones([test_number,4])
    y_test_9[ : , 0:2] = y_test[ : , 0:2] * 8/19
    y_test_9[ : , 2:4] = y_test[ : , 2:4] + (y_test_9[ : , 2:4] - y_test[ : , 2:4])*11/19
    print ("y_test_9[" + str(ran) + "]:" , y_test_9[ran])

    y_test_10 = np.ones([test_number,4])
    y_test_10[ : , 0:2] = y_test[ : , 0:2] * 9/19
    y_test_10[ : , 2:4] = y_test[ : , 2:4] + (y_test_10[ : , 2:4] - y_test[ : , 2:4])*10/19
    print ("y_test_10[" + str(ran) + "]:" , y_test_10[ran])

    y_test_11 = np.ones([test_number,4])
    y_test_11[ : , 0:2] = y_test[ : , 0:2] * 10/19
    y_test_11[ : , 2:4] = y_test[ : , 2:4] + (y_test_11[ : , 2:4] - y_test[ : , 2:4])*9/19
    print ("y_test_11[" + str(ran) + "]:" , y_test_11[ran])

    y_test_12 = np.ones([test_number,4])
    y_test_12[ : , 0:2] = y_test[ : , 0:2] * 11/19
    y_test_12[ : , 2:4] = y_test[ : , 2:4] + (y_test_12[ : , 2:4] - y_test[ : , 2:4])*8/19
    print ("y_test_12[" + str(ran) + "]:" , y_test_12[ran])

    y_test_13 = np.ones([test_number,4])
    y_test_13[ : , 0:2] = y_test[ : , 0:2] * 12/19
    y_test_13[ : , 2:4] = y_test[ : , 2:4] + (y_test_13[ : , 2:4] - y_test[ : , 2:4])*7/19
    print ("y_test_13[" + str(ran) + "]:" , y_test_13[ran])

    y_test_14 = np.ones([test_number,4])
    y_test_14[ : , 0:2] = y_test[ : , 0:2] * 13/19
    y_test_14[ : , 2:4] = y_test[ : , 2:4] + (y_test_14[ : , 2:4] - y_test[ : , 2:4])*6/19
    print ("y_test_14[" + str(ran) + "]:" , y_test_14[ran])

    y_test_15 = np.ones([test_number,4])
    y_test_15[ : , 0:2] = y_test[ : , 0:2] * 14/19
    y_test_15[ : , 2:4] = y_test[ : , 2:4] + (y_test_15[ : , 2:4] - y_test[ : , 2:4])*5/19
    print ("y_test_15[" + str(ran) + "]:" , y_test_15[ran])

    y_test_16 = np.ones([test_number,4])
    y_test_16[ : , 0:2] = y_test[ : , 0:2] * 15/19
    y_test_16[ : , 2:4] = y_test[ : , 2:4] + (y_test_16[ : , 2:4] - y_test[ : , 2:4])*4/19
    print ("y_test_16[" + str(ran) + "]:" , y_test_16[ran])

    y_test_17 = np.ones([test_number,4])
    y_test_17[ : , 0:2] = y_test[ : , 0:2] * 16/19
    y_test_17[ : , 2:4] = y_test[ : , 2:4] + (y_test_17[ : , 2:4] - y_test[ : , 2:4])*3/19
    print ("y_test_17[" + str(ran) + "]:" , y_test_17[ran])

    y_test_18 = np.ones([test_number,4])
    y_test_18[ : , 0:2] = y_test[ : , 0:2] * 17/19
    y_test_18[ : , 2:4] = y_test[ : , 2:4] + (y_test_18[ : , 2:4] - y_test[ : , 2:4])*2/19
    print ("y_test_18[" + str(ran) + "]:" , y_test_18[ran])

    y_test_19 = np.ones([test_number,4])
    y_test_19[ : , 0:2] = y_test[ : , 0:2] * 18/19
    y_test_19[ : , 2:4] = y_test[ : , 2:4] + (y_test_19[ : , 2:4] - y_test[ : , 2:4])*1/19
    print ("y_test_19[" + str(ran) + "]:" , y_test_19[ran])

    print ("y_test[" + str(ran) + "]:" , y_test[ran])

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(
        modelsavepath, verbose=1, save_best_only=True, save_weights_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/home/hx/hanxu/logs/{}'.format("3RNN")) # tensorboard --logdir=logs/

    model.fit(x = x_train, y = [y_train_2,y_train_3,y_train_4,y_train_5,y_train_6,y_train_7,y_train_8,y_train_9,y_train_10,y_train_11,y_train_12,y_train_13,y_train_14,y_train_15,y_train_16,y_train_17,y_train_18,y_train_19,y_train], batch_size=1, validation_data=(
        x_test, [y_test_2,y_test_3,y_test_4,y_test_5,y_test_6,y_test_7,y_test_8,y_test_9,y_test_10,y_test_11,y_test_12,y_test_13,y_test_14,y_test_15,y_test_16,y_test_17,y_test_18,y_test_19,y_test]), epochs=100, callbacks=[checkpointer]) 
        # tensorboard,earlystopper,checkpointer

elif (Next == 'predict'):
    shutil.rmtree('test')
    os.mkdir('test')
    model.load_weights(modelsavepath, by_name=True)

    test_image_paths = sorted(glob.glob(imgpath))
    test_annotations_paths = sorted(glob.glob(annotations_paths))

    for imagefile, xmlfile in zip(test_image_paths, test_annotations_paths):
        #img = Image.open(imagefile).resize((227, 227))
        img = load_img(imagefile, target_size=(227, 227))
        imgarr = img_to_array(img)
        x = np.asarray(imgarr) / 255.0
        x = np.expand_dims(x, axis=0)
        # x[:, :, :, 0] -= 103.939
        # x[:, :, :, 1] -= 116.779
        # x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        # x = x[:, :, :, ::-1]
        preds = model.predict(x)
        # print ('preds:', preds)

        x = xmltodict.parse(open(xmlfile, 'rb'))
        bndbox = x['annotation']['object']['bndbox']
        bndbox = np.array([float(bndbox['ymin']), float(
            bndbox['xmin']), float(bndbox['ymax']), float(bndbox['xmax'])])
        bndbox2 = [None] * 4
        bndbox2[0] = bndbox[0]
        bndbox2[1] = bndbox[1]
        bndbox2[2] = bndbox[2]
        bndbox2[3] = bndbox[3]
        bndbox2 = np.array(bndbox2)
        #print ('True box: ', bndbox2)
        
        predictBox = preds[18][0] * 227
        print ('cu Predict box: ', predictBox)
        source_img = Image.fromarray(imgarr.astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(source_img)
        draw.rectangle(
            (predictBox[1], predictBox[0], predictBox[3], predictBox[2]), outline=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) )
        draw.rectangle(
            (bndbox[1], bndbox[0], bndbox[3],bndbox[2]), outline="yellow")

        # print (str(imagefile)+' '+str(xmlfile))

        source_img.save('./test/{}.png'.format(str(imagefile).replace(str('/home/hx/hanxu/cu/'),'').replace('.jpg','')), 'png')

    test_image_paths = sorted(glob.glob(r'/home/hx/hanxu/foggy/*.jpg'))
    test_annotations_paths = sorted(glob.glob(r'/home/hx/hanxu/foggy/*.xml'))


    for imagefile, xmlfile in zip(test_image_paths, test_annotations_paths):
        #img = Image.open(imagefile).resize((227, 227))
        img = load_img(imagefile, target_size=(227, 227))
        imgarr = img_to_array(img)
        x = np.asarray(imgarr) / 255.0
        x = np.expand_dims(x, axis=0)
        # x[:, :, :, 0] -= 103.939
        # x[:, :, :, 1] -= 116.779
        # x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        # x = x[:, :, :, ::-1]
        preds = model.predict(x)
        # print ('preds:', preds)

        x = xmltodict.parse(open(xmlfile, 'rb'))
        bndbox = x['bndbox']
        bndbox = np.array([ float(bndbox[ 'top' ]) , float(bndbox[ 'left' ]) , float(bndbox[ 'bottom' ]) , float(bndbox[ 'right' ]) ])
        bndbox2 = [None] * 4
        bndbox2[0] = bndbox[0]
        bndbox2[1] = bndbox[1]
        bndbox2[2] = bndbox[2]
        bndbox2[3] = bndbox[3]
        bndbox2 = np.array(bndbox2)
        #print ('True box: ', bndbox2)

        predictBox = preds[18][0] * 227
        bndbox = bndbox * 227
        print ('foggy Predict box: ', predictBox)
        source_img = Image.fromarray(imgarr.astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(source_img)
        draw.rectangle(
            (predictBox[1], predictBox[0], predictBox[3], predictBox[2]), outline=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) )
        draw.rectangle(
            (bndbox[1], bndbox[0], bndbox[3],bndbox[2]), outline="yellow")

        # print (str(imagefile)+' '+str(xmlfile))
        source_img.save('./test/{}.png'.format(str(imagefile).replace(str('/home/hx/hanxu/foggy/'),'').replace('.jpg','')), 'png')


    test_image_paths = sorted(glob.glob(r'/home/hx/hanxu/rain/*.jpg'))
    #test_annotations_paths = sorted(glob.glob(r'/home/hx/hanxu/foggy/*.xml'))

    for imagefile in test_image_paths:
        #img = Image.open(imagefile).resize((227, 227))
        img = load_img(imagefile, target_size=(227, 227))
        imgarr = img_to_array(img)
        x = np.asarray(imgarr) / 255.0
        x = np.expand_dims(x, axis=0)
        # x[:, :, :, 0] -= 103.939
        # x[:, :, :, 1] -= 116.779
        # x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        # x = x[:, :, :, ::-1]
        preds = model.predict(x)
        # print ('preds:', preds)

        predictBox = preds[18][0] * 227
        
        print ('rain Predict box: ', predictBox)
        source_img = Image.fromarray(imgarr.astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(source_img)
        draw.rectangle(
            (predictBox[1], predictBox[0], predictBox[3], predictBox[2]), outline=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) )
        

        # print (str(imagefile)+' '+str(xmlfile))
        source_img.save('./raintest/{}.png'.format(str(imagefile).replace(str('/home/hx/hanxu/rain/'),'').replace('.jpg','')), 'png')
    print ( "###FINISH###" )