# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import   Model, Sequential
import keras.backend as K
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop
import h5py
import sys
import cv2
from data_utils_extras import *

import scipy
import scipy.stats

try:
    from data_utils import *
except:
    print("data_utils.py not found. Training will not work.")

def round_down(num, divisor=32):
    return num - (num%divisor)

def tf_like_imread(f_name, mode=0):
#    print (f_name)
    im= cv2.imread(f_name,cv2.IMREAD_UNCHANGED)              
    if im is None:       
        raise  ValueError('No image at '+f_name)
        return -1               
    else:    
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.float32(im)
        im = get_luminance(im)[:,:,np.newaxis]
        if mode == 1:
            [w,h,c] = im.shape
            im = im[0:round_down(w), 0:round_down(h), :]
        im[im>4000] = 4000
        im[im<0] = 0
        return im
    
class augmented_input_layer(Layer):     
    # Gives 2 filter maps one mean and one std deviation. 
    def build(self, input_shape):
        # Starting with custom scaling values for similar feature
        self.W_meanL =  K.variable( 1/255.0 ) 
        self.W_lum =  K.variable( 1/255.0 )
        self.W_var = K.variable( 1/255.0 )
        self.W_nim = K.variable(1/4.0 )
        self.B_mean = K.variable(1e-6)
        self.B_lum = K.variable(1e-6)
        self.B_nim = K.variable(1e-6)
        self.B_var = K.variable(1e-6)
        self.trainable_weights =[self.W_meanL,self.W_lum, 
                                self.B_mean,self.B_lum,
                                 self.B_nim,self.B_var,
                                 self.W_var,self.W_nim ]
        
    def compute_output_shape(self, input_shape): 
        shape = list(input_shape)        
        shape[3] *= 4;           
        return tuple(shape)
    def _tf_fspecial_gauss(self,size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)
    def call(self, x, mask=None):           
        image = x
        G_kernel = self._tf_fspecial_gauss(7, 1) # window shape [size, size]            
            
        meanL = tf.nn.conv2d(image, G_kernel, strides=[1,1,1,1], padding='SAME')
        meanL2 = tf.nn.conv2d(K.square(image),G_kernel, strides=[1,1,1,1], padding='SAME')        
 
        varL= K.sqrt( K.abs( meanL2 - K.square(meanL) ) )            
        nim = K.abs(x - meanL)/(varL+0.01) 
        
        return concatenate( 
                   [   ( x*self.W_lum   + self.B_lum), 
                    ( meanL*self.W_meanL + self.B_mean) ,
                            varL*self.W_var   + self.B_var, 
                             nim*self.W_nim   + self.B_nim],
                    axis=3)

class mixing_function(Layer):
    def build(self, input_shape):  
        self.a = K.variable(0.3) 
        self.trainable_weights = [self.a]
        
    def compute_output_shape(self, input_shape): 
        shape = list(input_shape)        
        shape[1] = 1;       
        return tuple([input_shape[0][0],1])

    def call(self, x, mask=None): 
        err =self.a*( x[0] )
        jnd = x[1] 
        qscore =  1 - K.exp(- err/jnd)
#        qscore =  K.tanh( K.abs(err/jnd))
        return   qscore

class clip(Layer):   
    def compute_output_shape(self, input_shape):                   
        return input_shape

    def call(self, x, mask=None):         
        jnd = K.clip( x,0.01,100.0) 
        return  jnd
    
class model_deep_vis():
    def __init__(self, img_rows=32, img_cols = 32, channel = 1, load_weights=0):
        self.img_shape = (img_rows, img_cols, channel)
        self.make_model()
        if load_weights:
            self.load_weights()
                
    def load_weights(self):        
        self.P_net_model.load_weights('weights/iqa.h5')
                
    def predict_quality(self,f1,f2,draw=1):
        try:
            im_dis = tf_like_imread(f1,1)
        except ValueError as e:
            raise ValueError(e)                  
        
        if f2 is not None:
            try:
                im_ref = tf_like_imread(f2,1)
            except ValueError as e:
                raise ValueError(e)
        else:
            im_ref = im_dis
        
        [wd,ht,ch] = im_dis.shape
        ref_patches,dis_patches,pdmos, _ =  extractpatchwithLabel(im_ref, im_dis ,
                                            self.img_shape[0],self.img_shape[1],
                                            subsample=1,score=0)
        delta = self.err( dis_patches, ref_patches)
        p_dis_map = self.HDR_PCNN.predict([dis_patches,delta])
        algo_score = np.mean(p_dis_map)

        per_resist = ( self.P_net_model.predict(dis_patches) )
        if draw:
            try:
                k = delta.reshape( (wd-32)//32,-1 )
                per_resist = per_resist.reshape( (wd)//32,-1 )
                p_dis_map = p_dis_map.reshape( (wd)//32,-1 )
            except:
                try:
                    k = delta.reshape( (wd-32)//32,-1 )
                    per_resist = per_resist.reshape( (wd-32)//32,-1 )
                    p_dis_map = p_dis_map.reshape( (wd-32)//32,-1 )                
                except :
                    print('')
                print('')
            imshow( k, title="Delta" )
            imshow(per_resist, title="Perceptual Resistance" )
            imshow(p_dis_map, title="Perceptual distortion" )
        try:
            p_dis_map = p_dis_map.reshape( (wd-32)//32,-1 ) 
        except:
            print("Error in reshaping. Distortion map returned as array.")
        return [algo_score, p_dis_map]
    
    def get_thresh(self,im_dis,draw=1):
        im_dis = np.float32(im_dis)
        [wd,ht] = im_dis.shape
        ch = 1
        self.load_weights()
        ref_patches,dis_patches,pdmos, _ =  extractpatchwithLabel(im_dis, im_dis ,
                                            self.img_shape[0],self.img_shape[1],
                                            subsample=8,score=0)
        dis_patches = dis_patches.reshape(-1,self.img_shape[0],self.img_shape[1],1)
        per_resist = ( self.P_net_model.predict(dis_patches) )
        if draw:
            wd = np.int( np.sqrt(len(per_resist)) )

            try:
                per_resist = per_resist.reshape( (wd),-1 )
            except:
                try:
                    per_resist = per_resist.reshape( (wd-32),-1 )
                except :
                    print('')
                print('')
        return  per_resist
    
    
    def test(self, draw=1, num=0):
        machine = []
        human = []
        try:
            [dis,ref,mos] = self.db.get_test_sets(num)
        except:
            self.db = db_CSIQ('D:\Documents\PhD\LDR datasets\CSIQ\\',1, 1,self.img_shape[0],self.img_shape[1])
            [dis,ref,mos] = self.db.get_test_sets(num)
            
        count = 0
        print( "Testing on %d images."%len(dis))
        self.load_weights()
        
        for [f1,f2,val] in zip(dis,ref,mos):
            print(".", end='')
            if draw:
                im_dis = tf_like_imread(f1,1)
                im_ref = tf_like_imread(f2,1)
                [wd,ht,ch] = im_dis.shape
                ref_patches,dis_patches,pdmos, _ =  extractpatchwithLabel(im_ref, im_dis ,
                                                self.img_shape[0],self.img_shape[1],
                                                subsample=1,score=0)
                delta = np.mean( np.abs(ref_patches-dis_patches), axis=(1,2,3) )
                im_dis = im_dis[:,:,0]
                im_ref = im_ref[:,:,0]
                delta_actual = np.abs(im_ref-im_dis)
                
                imshow(delta_actual, title="Actual Error" )
                im_dis = cv2.imread(f1)
                im_dis = cv2.cvtColor(im_dis, cv2.COLOR_BGR2RGB)
                plt.imshow(im_dis)
                plt.title("Distorted Image Luminance" )
                plt.show()
            
            mos_train = val
            [algo_score,map] = self.predict_quality(f1,f2,draw=draw)
            machine.append(algo_score)
            human.append(mos_train)
            
        srcc = scipy.stats.spearmanr(np.array(machine), np.array(human))[0]
        print(machine[0:5],human[0:5])
        print ("\nSRCC: ", srcc)
        return srcc
                
    def get_nets(self):
        return [self.HDR_PCNN, self.P_net_model, self.E_net_model]
    
    def make_model(self):
        ############################ P net #########################################
        P_net = Sequential()
        P_net.add(augmented_input_layer(input_shape=(self.img_shape)))
        P_net.add(Conv2D(32, (5,5),input_shape=(self.img_shape)))
        P_net.add(Activation('relu'))
        P_net.add(Flatten())
        P_net.add(Dropout(0.5))
        P_net.add(Dense(100))
        P_net.add(Activation ('relu'))
        P_net.add(Dropout(0.5))
        P_net.add(Dense(1,activation='linear'))
        P_net.add(clip())
#        P_net.summary()
        ##############################################################################
        input_layer = Input(shape=(self.img_shape))	
        delta_hat     = Input(shape=(1,))
        perceptual_resistance = P_net(input_layer)
        perceptual_distortion = mixing_function()( [delta_hat, perceptual_resistance] )
        sgd = keras.optimizers.Adam()
        self.P_net_model = Model( input_layer, perceptual_resistance) 
        self.P_net_model.compile(loss='mae', optimizer=sgd)
        #######################  MODEL COMPILES  #####################################	        
        self.HDR_PCNN    = Model( [input_layer,delta_hat] , perceptual_distortion)
        self.HDR_PCNN.compile(loss='mae', optimizer=sgd)
        return 
    
    def err(self,X_train_dis,X_train_ref):
        er = np.mean(np.abs(X_train_dis - X_train_ref), axis = (1,2,3))
        con = er
        return con
    
    def train(self, n_epochs , batch_size ):        
        self.db = db_TID2013("D:\Documents\PhD\LDR datasets\\tid2013\\",batch_size, 1,self.img_shape[0],self.img_shape[1])
        N = self.db.get_count()
        n_batch =  np.max([1,N//batch_size])
        print( 'Training on %d samples with %d batches.'%(N,n_batch))               
        test_SRCC = 0
        for eph in range(n_epochs):
            for j in range(n_batch):
                [X_train_dis,X_train_ref,Y_train] = self.db.get_next_batch()
                print(X_train_dis.shape)
                delta = self.err( X_train_dis, X_train_ref)
                er2 = 0
                er2 = self.HDR_PCNN.fit ([X_train_dis,delta],Y_train,batch_size=32,
                                         epochs = 1 )
            self.P_net_model.save_weights('weights/iqa.h5',overwrite=True) 
            print('')
            if test_SRCC:
                    self.test(draw=0)
            
            