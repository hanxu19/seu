from __future__ import print_function
#model Reuse model
import numpy as np
import keras.backend as K
import time
import tensorflow as tf
import xmltodict
import glob
import os
import sys

from keras.utils import plot_model
from keras.layers import merge, Input ,concatenate
from keras.layers import Dense, Activation, Flatten, Reshape, MaxPooling1D
from keras.layers import MaxPooling2D, ZeroPadding2D, AveragePooling2D, Conv2D, MaxPool2D, Cropping2D, TimeDistributed
from keras.layers import SimpleRNN
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
from skimage.io import imread,imshow
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image , ImageDraw
from keras.layers.core import Layer

if (sys.platform == 'linux'):
    imgpath = r'/home/smf/hanxu/foggy/*.jpg'
    annotations_paths = r'/home/smf/hanxu/foggy/*.xml'
    modelsavepath = r'/home/smf/hanxu/foggy.h5'
else:
    imgpath =  r'C:/Users/gngn39\Desktop/vi/Image/foggy/*.jpg' 
    annotations_paths =  r'C:/Users/gngn39\Desktop/vi/Image/foggy/*.xml' 
    modelsavepath = r'C:/Users/gngn39/Desktop/vi/foggy.h5'
    

TIME_STEPS = 64
INPUT_SIZE = 64
# img = Image.open(imgpath)
IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3

#process data
images = []
image_paths = glob.glob(imgpath)
for imagefile in image_paths:
    image = Image.open( imagefile ).resize( ( 227 , 227 ))
    image = np.asarray( image ) / 255.0
    images.append( image )

bboxes = []
classes_raw = []
annotations_paths = glob.glob(annotations_paths)
for xmlfile in annotations_paths:
    x = xmltodict.parse( open( xmlfile , 'rb' ) )
    bndbox = x[ 'bndbox' ]
    bndbox = np.array([ float(bndbox[ 'top' ]) , float(bndbox[ 'left' ]) , float(bndbox[ 'bottom' ]) , float(bndbox[ 'right' ]) ])
    bndbox2 = [ None ] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]

    bndbox2 = np.array( bndbox2 ) 

    bboxes.append( bndbox2 )

boxes = np.array( bboxes ) 

Y = np.array( boxes )
X = np.array( images )

x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.1 )


#layer
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
        img = tf.image.crop_and_resize(image_meta,boxes,[0],(227,227))
        #print (img)
        return img

    def compute_output_shape(self, input_shape):
        batch_size = 1
        return (batch_size,227,227,3)


def gs_block_layer1(img,layerNumber):
    
    #if first RNN layer, don't crop 
    # inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3)) #channel 3
    c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid', \
        name='conv1'+ '_rnn_' + str(layerNumber) )(img)
    c2 = BatchNormalization(name='conv1bn'+ '_rnn_' + str(layerNumber))(c1)
    c3 = MaxPool2D((3, 3), strides=2,padding='valid')(c2)
    
    c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform', \
        name='conv2'+ '_rnn_' + str(layerNumber))(c3)
    c5 = BatchNormalization(name='conv2bn'+ '_rnn_' + str(layerNumber))(c4)
    c6 = MaxPool2D((3, 3), strides=2,padding='valid')(c5)
    
    c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv3'+ '_rnn_' + str(layerNumber))(c6)
    c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv4'+ '_rnn_' + str(layerNumber))(c7)
    c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv5'+ '_rnn_' + str(layerNumber))(c8)
    c10 = MaxPool2D((3, 3), strides=2,padding='valid')(c9)

    c11 = Flatten()(c10)
    c12 = Dense(4096, activation='relu',name='fc4096'+ '_rnn_' + str(layerNumber))(c11)  # 2048 in paper, fc 1
    c13 = Dropout(0.5)(c12)
    c14 = Lambda(lambda c13: K.expand_dims(c13,axis=1))(c13)
  
    return c14


def gs_block(img,rt,layerNumber):
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
    cropped = croprt()([input,rt])

    # inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3)) #channel 3
    c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid', \
        name='conv1'+ '_rnn_' + str(layerNumber) )(cropped)
    c2 = BatchNormalization(name='conv1bn'+ '_rnn_' + str(layerNumber))(c1)
    c3 = MaxPool2D((3, 3), strides=2,padding='valid')(c2)
    
    c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform', \
        name='conv2'+ '_rnn_' + str(layerNumber))(c3)
    c5 = BatchNormalization(name='conv2bn'+ '_rnn_' + str(layerNumber))(c4)
    c6 = MaxPool2D((3, 3), strides=2,padding='valid')(c5)
    
    c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv3'+ '_rnn_' + str(layerNumber))(c6)
    c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv4'+ '_rnn_' + str(layerNumber))(c7)
    c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform',\
        name='conv5'+ '_rnn_' + str(layerNumber))(c8)
    c10 = MaxPool2D((3, 3), strides=2,padding='valid')(c9)

    c11 = Flatten()(c10)
    c12 = Dense(4096, activation='relu',name='fc4096'+ '_rnn_' + str(layerNumber))(c11)  # 2048 in paper, fc 1
    c13 = Dropout(0.5)(c12)
    c14 = Lambda(lambda c13: K.expand_dims(c13,axis=1))(c13)
    #alexnet = Model(inputs , c13)
    #alexnet.load_weights('C:\Users\gngn39\Desktop/vi\model-2.h5' , by_name = True)
    #features = alexnet.predict(image_data)
    
    #print (features.shape)
    #np.savetxt('C:/Users/gngn39/Desktop/vi/features2.txt', features ,fmt='%s')
    #return features
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
        
input= Input(name='TOTAL_input', shape=(227,227,3))
#rnn
vt1 = gs_block_layer1(input , 1)
ht1, state_1 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block1')(vt1)
#F1 = merge([state_1, c1_r1_Flatten], mode='sum') #F1

rt2 = gr_block(ht1)
vt2 = gs_block(input , rt2 , 2)
ht2, state_2 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block2')(vt2 , initial_state=state_1)

rt3 = gr_block(ht2)
vt3 = gs_block(input , rt3 , 3)
ht3, state_3 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block3')(vt3 , initial_state=state_2)

rt4 = gr_block(ht3)
vt4 = gs_block(input , rt4 , 4)
ht4, state_4 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block4')(vt4 , initial_state=state_3)
#F2 = merge([state_4, c2_r4_Flatten], mode='sum') #F2

rt5 = gr_block(ht4)
vt5 = gs_block(input , rt5 , 5)
ht5, state_5 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block5')(vt5 , initial_state=state_4)

rt6 = gr_block(ht5)
vt6 = gs_block(input , rt6 , 6)
ht6, state_6 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block6')(vt6 , initial_state=state_5)

rt7 = gr_block(ht6)
vt7 = gs_block(input , rt7 , 7)
ht7, state_7 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block7')(vt7 , initial_state=state_6)
#F3 = merge([state_7, c3_r7_Flatten], mode='sum') #F3 

rt8 = gr_block(ht7)
vt8 = gs_block(input , rt8 , 8)
ht8, state_8 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block8')(vt8 , initial_state=state_7)

rt9 = gr_block(ht8)
vt9 = gs_block(input , rt9 , 9)
ht9, state_9 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block9')(vt9 , initial_state=state_8)

rt10 = gr_block(ht9)
vt10 = gs_block(input , rt10 , 10)
ht10, state_10 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block10')(vt10 , initial_state=state_9)
#F4 = merge([state_10, c4_r10_Flatten], mode='sum') #F4

rt11 = gr_block(ht10)
vt11 = gs_block(input , rt11 , 11)
ht11, state_11 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block11')(vt11 , initial_state=state_10)

rt12 = gr_block(ht11)
vt12 = gs_block(input , rt12 , 12)
ht12, state_12 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block12')(vt12 , initial_state=state_11)

rt13 = gr_block(ht12)
vt13 = gs_block(input , rt13 , 13)
ht13, state_13 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block13')(vt13 , initial_state=state_12)
#F5 = merge([state_13, c5_r13_Flatten], mode='sum') #F5

rt14 = gr_block(ht13)
vt14 = gs_block(input , rt14 , 14)
ht14, state_14 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block14')(vt14 , initial_state=state_13)

rt15 = gr_block(ht14)
vt15 = gs_block(input , rt15 , 15)
ht15, state_15 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block15')(vt15 , initial_state=state_14)

rt16 = gr_block(ht15)
vt16 = gs_block(input , rt16 , 16)
ht16, state_16 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block16')(vt16 , initial_state=state_15)
#F6 = merge([state_16, c6_r16_Flatten], mode='sum') #F6

rt17 = gr_block(ht16)
vt17 = gs_block(input , rt17 , 17)
ht17, state_17 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block17')(vt17 , initial_state=state_16)

rt18 = gr_block(ht17)
vt18 = gs_block(input , rt18 , 18)
ht18, state_18 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=True ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block18')(vt18 , initial_state=state_17)

rt19 = gr_block(ht18)
vt19 = gs_block(input , rt19 , 19)
ht19 = SimpleRNN(units=256 , activation='relu' , return_sequences=False , return_state=False ,\
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE), name='gh_block19')(vt19 , initial_state=state_18)
#F7 = merge([ht19, c7_r19_Flatten], mode='sum') #F7
#rnn_outputs = F7
dense_output = Dense(4,name='dense_output')(ht19)
#outputs = concatenate([cnn_outputs , rnn_outputs])

model = Model(inputs=[input], outputs=[dense_output])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train or predict
Next = 'train'
if (Next == 'train'):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(modelsavepath, verbose=1, save_best_only=True,save_weights_only=True)
    model.fit(x_train,y_train , batch_size = 1 ,validation_data=( x_test , y_test ), epochs = 10, callbacks=[earlystopper, checkpointer])
else :
    model.load_weights(modelsavepath , by_name = True)
    img = load_img(testimgpath, target_size=(227, 227))
    x = img_to_array(img)
    x = np.asarray( x ) / 255.0
    x = np.expand_dims(x, axis=0)
    # x[:, :, :, 0] -= 103.939
    # x[:, :, :, 1] -= 116.779
    # x[:, :, :, 2] -= 123.68
    # # 'RGB'->'BGR'
    # x = x[:, :, :, ::-1]
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print (preds)