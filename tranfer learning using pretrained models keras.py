# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing
from random import shuffle, uniform, seed
import augmentations as aug
import utils
import params
import scipy

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate, GlobalMaxPooling2D, Dropout, Lambda
from keras.preprocessing import image
from keras.models import Model
from keras.backend import tf as ktf
from keras.optimizers import Adam, SGD, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def data_generator(data=None, meta_data=None, labels=None, batch_size=64, augment={}, opt_shuffle=True):
    
    indices = [i for i in range(len(labels))]
    
    while True:
        
        if opt_shuffle:
            shuffle(indices)
        
        x_data = np.copy(data)
        x_meta_data = np.copy(meta_data)
        x_labels = np.copy(labels)
        
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            sel_indices = indices[start:end]
            
            #select data
            data_batch = x_data[sel_indices]
            xm_batch = x_meta_data[sel_indices]
            y_batch = x_labels[sel_indices]
            x_batch = []
            
            for x in data_batch:
                #x = scipy.misc.imresize(x, (299, 299, 3))
                #augment                               
                if augment.get('Rotate', False):
                    x = aug.Rotate(x, u=0.1, v=np.random.random())
                    x = aug.Rotate90(x, u=0.1, v=np.random.random())

                if augment.get('Shift', False):
                    x = aug.Shift(x, u=0.05, v=np.random.random())

                if augment.get('Zoom', False):
                    x = aug.Zoom(x, u=0.05, v=np.random.random())

                if augment.get('Flip', False):
                    x = aug.HorizontalFlip(x, u=0.5, v=np.random.random())
                    x = aug.VerticalFlip(x, u=0.5, v=np.random.random())

                x_batch.append(x)
                
            x_batch = np.array(x_batch, np.float32)
            
            yield [x_batch, xm_batch], y_batch
        
def resize_image(X_data, new_shape):

    X_data_resized = [scipy.misc.imresize(img, new_shape) for img in X_data] 
    
    return np.array(X_data_resized)

def get_model(img_shape, name=''):
    if not name:
        print("enter model name.")
        raise NameError
    if name == 'resnet50':
        basemodel = ResNet50(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'vgg16':
        basemodel = VGG16(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'vgg19':
        basemodel = VGG19(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'xception':
        basemodel = Xception(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'inceptionv3':
        basemodel = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'inceptionv2':
        basemodel = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_shape, classes=1)
    elif name == 'mobilenet':
        basemodel = MobileNet(weights='imagenet', alpha=0.5, include_top=False, input_shape=img_shape, classes=1)
    
    return basemodel


np.random.seed(1017)
target = 'is_iceberg'

#Load data
train, train_bands = utils.read_jason(file='train.json')
test, test_bands = utils.read_jason(file='test.json')

#target
train_y = train[target].values
split_indices = train_y.copy()

#data set
train_X = utils.rescale(train_bands)
train_meta = train['inc_angle'].values
test_X_dup = utils.rescale(test_bands)
test_meta = test['inc_angle'].values

opt_augments = {'Flip': False, 'Rotate': False, 'Shift': False, 'Zoom': False}
opt_augments['Flip'] = True
opt_augments['Rotate'] = True
opt_augments['Shift'] = True
opt_augments['Zoom'] = True    
print(opt_augments)
epochs = params.epochs
batch_size = params.batch_size

## Extract Xception bottleneck features
new_shape = (299, 299, 3)
train_bands_299 = resize_image(train_bands, new_shape)
test_bands_299 = resize_image(test_bands, new_shape)   

model_name = 'xception'
xception = get_model(img_shape=(299, 299, 3), name=model_name)

xception_train = xception.predict(train_bands_299, batch_size=batch_size, verbose=1)
xception_test = xception.predict(test_bands_299, batch_size=batch_size, verbose=1)

## Extract Inception_v3 bottleneck features
model_name = 'inceptionv3'
Inceptionv3 = get_model(img_shape=(299, 299, 3), name=model_name)

Inceptionv3_train = Inceptionv3.predict(train_bands_299, batch_size=batch_size, verbose=1)
Inceptionv3_test = Inceptionv3.predict(test_bands_299, batch_size=batch_size, verbose=1)




## Extract vgg16 bottleneck features

model_name = 'xception'
xception = get_model(img_shape=(299, 299, 3), name=model_name)

xception_train = xception.predict(train_bands_299, batch_size=batch_size, verbose=1)
xception_test = xception.predict(test_bands_299, batch_size=batch_size, verbose=1)































