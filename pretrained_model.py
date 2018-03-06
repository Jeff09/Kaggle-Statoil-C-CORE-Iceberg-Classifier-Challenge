# -*- coding: utf-8 -*-

import os
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from random import shuffle, uniform, seed
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
#
from keras.optimizers import Adam, SGD, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#
import augmentations as aug
import utils
import params
import scipy
import models

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
###############################################################################

def resize_image(X_data, new_shape):

    X_data_resized = [scipy.misc.imresize(img, new_shape) for img in X_data] 
    
    return np.array(X_data_resized)

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
                    
    x = basemodel.output
    x = GlobalMaxPooling2D()(x)
    
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    
    merge_one = concatenate([x, angle_layer])
    
    merge_one = Dropout(0.7)(merge_one)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)
    
    model = Model(input=[basemodel.input, input_2], output=predictions)
    
    opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True) #Adam(lr=5e-4) # 
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model

#Resize image
def resize_image2(images):
    inp = Input(shape=(None, None, 3))
    try:
        out = Lambda(lambda img: ktf.image.resize_images(img, (224, 224)))(inp)
    except :
        # if you have older version of tensorflow
        out = Lambda(lambda img: ktf.image.resize_images(img, 224, 224))(inp)
    
    model = Model(input=inp, output=out)
    out = model.predict(images)
    
    return out

###############################################################################
if __name__ == '__main__':
    
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
    
    new_shape = (224, 224, 3)
    train_X = resize_image(train_X, new_shape)
    test_X_dup = resize_image(test_X_dup, new_shape)

    #train, validataion split
    test_ratio = 0.2
    nr_runs = 5
    split_seed = 25
    kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
    
    #training, evaluation, test and make submission
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*train_y
    for r, (train_index, valid_index) in enumerate(kf.split(train, split_indices)):
        
        print("fold {}".format(r))

        tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        y1, y2 = train_y[train_index], train_y[valid_index]
        x1, x2 = train_X[train_index], train_X[valid_index]
        xm1, xm2 = train_meta[train_index], train_meta[valid_index]
                

        print('splitted: {0}, {1}'.format(x1.shape, x2.shape), flush=True)
        print('splitted: {0}, {1}'.format(y1.shape, y2.shape), flush=True)
        ################################
        
        weights_file = "%s_aug_model_weights.hdf5"%r
        model_name = 'resnet50'
        model = get_model(img_shape=(224, 224, 3), name=model_name)
        
        #optim = SGD(lr=0.005, momentum=0.0, decay=0.002, nesterov=True)
        #optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002)
        
        #model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
        #call backs
        earlystop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, min_delta=1e-4, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
        model_chk = ModelCheckpoint(monitor='val_loss', filepath=weights_file, save_best_only=True, save_weights_only=True, mode='min')
        tb = TensorBoard(log_dir='/logs/', histogram_freq=1, write_grads=True, write_images=True)
        callbacks = [earlystop, reduce_lr_loss, model_chk]
        
        ##########
        model.fit_generator(generator=data_generator(x1, xm1, y1, batch_size=batch_size, augment=opt_augments),
                            steps_per_epoch= np.ceil(8.0 * float(len(y1)) / float(batch_size)),
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=data_generator(x2, xm2, y2, batch_size=batch_size),
                            validation_steps=np.ceil(8.0 * float(len(y2)) / float(batch_size)))
        
        #Getting the Best Model
        model.load_weights(filepath=weights_file)
        #Getting Training Score
        score = model.evaluate([x1, xm1], y1, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = model.evaluate([x2, xm2], y2, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
                
        #Getting validation Score.
        pred_valid=model.predict([x2, xm2])
        y_valid_pred_log[valid_index] = pred_valid.reshape(pred_valid.shape[0])
        
        #Getting Test Scores
        temp_test=model.predict([test_X_dup, test_meta], batch_size=batch_size, verbose=1)
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores
        temp_train=model.predict([train_X, train_meta])
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

    y_test_pred_log=y_test_pred_log/nr_runs
    y_train_pred_log=y_train_pred_log/nr_runs

    print('\n Train Log Loss Validation= ',log_loss(train_y, y_train_pred_log))
    print(' Test Log Loss Validation= ',log_loss(train_y, y_valid_pred_log))
        
    ids = test['id'].values

    file = 'subm_{}_{}.csv'.format(tmp, model_name)
    subm = pd.DataFrame({'id': ids, target: y_test_pred_log})
    subm.to_csv('submit/{}'.format(file), index=False, float_format='%.6f')