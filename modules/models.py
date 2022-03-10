#%matplotlib widget
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
import tensorflow as tf
import tensorflow_probability as tfp
assert '0.12' in tfp.__version__, tfp.__version__
assert '2.4' in tf.__version__, tf.__version__
import mdn
from tensorflow.keras.regularizers import l2
#general 
import numpy as np
import pandas as pd
import math
import os
import keras.backend as K
import tensorflow as tf
# RNN, deep leraning ect.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tf.keras.layers.MultiHeadAttention 
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional,concatenate,LeakyReLU
import datetime
import tensorflow_addons as tfa
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import Flatten,Concatenate,Attention,Add,BatchNormalization,MultiHeadAttention
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, SimpleRNN, GRU, TimeDistributed, ConvLSTM2D, RNN,Conv1D,Average
from tensorflow.keras.layers import RepeatVector, Input, Flatten
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import TimeDistributed, Attention
#plotting
#import gdal
#import osr
#import geopandas as gpd
#import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import datetime, os

from tensorflow.keras.layers import Dense,LSTM, Dropout, Bidirectional, SimpleRNN, GRU, TimeDistributed, Input, RNN,RepeatVector,Masking,TimeDistributed,BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage, misc
import tensorflow as tf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER




from keras.callbacks import LambdaCallback   

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import random
import os


def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(seed=3)



from keras_self_attention import SeqSelfAttention
import os


class thesis_model:
    def __init__(self,
                 inputsize = (20,10),
                 outputsize = (10),
                 model_name='Thesis_model'):
        self.model_name = model_name
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.model = None
        
        
    def build_model(self):
        
        
        encoder_inputs = Input(shape=self.inputsize,name='encoder_input')
        x = Bidirectional(LSTM(356, kernel_initializer = tf.keras.initializers.GlorotNormal(),return_sequences=True))(encoder_inputs)
        x = BatchNormalization(momentum=0.99, 
                                           scale=True, 
                                           center=True,
                                           trainable=False)(x)
        x = Dropout(0.4)(x)

        x = Bidirectional(LSTM(256, kernel_initializer = tf.keras.initializers.GlorotNormal(),return_sequences=True))(x)
        x = BatchNormalization(momentum=0.99, 
                                           scale=True, 
                                           center=True,
                                           trainable=False)(x)

        x = Dropout(0.4)(x)
        x = Bidirectional(LSTM(256,kernel_initializer = tf.keras.initializers.GlorotNormal(), return_sequences=True))(x)
        x = BatchNormalization(momentum=0.99, 
                                           scale=True, 
                                           center=True,
                                           trainable=False)(x)
        x = Dropout(0.4)(x)


        z = Bidirectional(LSTM(128,kernel_initializer = tf.keras.initializers.GlorotNormal(), return_sequences=True))(x)
        z = BatchNormalization(momentum=0.99, 
                                           scale=True, 
                                           center=True,
                                           trainable=False)(z)
        z = Dropout(0.4)(z)
        z = Bidirectional(LSTM(256,kernel_initializer = tf.keras.initializers.GlorotNormal(), return_sequences=True))(z)
        z = BatchNormalization(momentum=0.99, 
                                           scale=True, 
                                           center=True,
                                           trainable=False)(z)
        z = Dropout(0.4)(z)


        attn_out = tf.keras.layers.Attention()([x, z])
        attn_out = Dropout(0.4)(attn_out)
        attn_out = Flatten()(attn_out)
        pred = Dense(self.outputsize[-1])(attn_out)
        Name = f"{datetime.datetime.now().strftime('%Y%m%d-%H')}_{self.model_name}"


        self.model = Model(inputs=encoder_inputs, outputs=pred,name=Name)
        
        return self.model
        

class EncoderDecoderAttention:
    '''
    '''
    def __init__(self, 
                 encoder_decoder_name:str = 'Encoder_Decoder',
                 encoder_hidden_layers:list = [256,256],
                 decoder_hidden_layers:list = [256,256],
                 kernel_initializer:str ='glorot_uniform' , 
                 kernel_regularizer:float = 0.0001,
                 bias_regularizer:float = 0.0001,
                 recurrent_regularizer:float = 0.0001,
                 dynamic_dropout:float = 0.2,
                 combiend_dropout:float = 0.2,
                 static_dropout:float = 0.2,
                 decoder_dropout:float = 0.2,
                 BN_momentum:float = 0.99,
                 static_FN_depth:list = [50,50],
                 combied_FN_depth:list = [64,64],
                 decoder_FC_depth:list = [64,64],
                 FN_activation:str = 'relu',
                 meta_shape:int = 0,
                 using_attention:bool = True,
                 verbose:int = 0):
        '''
        
        '''
        self.encoder_decoder_name = encoder_decoder_name
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer =kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.dynamic_dropout = dynamic_dropout
        self.BN_momentum = BN_momentum
        self.static_FN_depth = static_FN_depth
        self.decoder_FC_depth = decoder_FC_depth
        self.combied_FN_depth = combied_FN_depth
        self.combiend_dropout = combiend_dropout
        self.decoder_dropout = decoder_dropout
        self.static_dropout = static_dropout
        self.FN_activation = FN_activation
        self.meta_shape = None
        self.decoder_inputs = None
        self.encoder_dynamic_inputs = None
        self.encoder_static_inputs = None
        self.using_attention= using_attention
        self.verbose = verbose
        self.model = None
        self.input_shape = None
        self.encoder_model = None
        self.decoder_model = None
        
        try:
            keras.backend.clear_session()
        except:
            pass
     
    def build_autoencoder_attention(self,input_shape=(20,10),
                                    meta_shape=0):
        '''
        '''
        self.input_shape = input_shape
        #build BLSTM Encoder part for dynamic inputs
        if self.verbose>0:
            print('Encoder| building')

        
        self.input_shape = input_shape
        encoder_dynamic_inputs = Input(shape=(input_shape[0],input_shape[1]),name='encoder_dynamic_inputs')
        if meta_shape>0:
            self.meta_shape = meta_shape
            encoder_static_inputs = Input(shape=(meta_shape,),name=f'encoder_static_inputs')
        ###########################################################
        #################### ENCODER DYNAMIC ######################
        ###########################################################
        encoder1 = Bidirectional(LSTM(self.encoder_hidden_layers[0],
                               kernel_initializer = self.kernel_initializer, 
                               kernel_regularizer=l2(self.recurrent_regularizer), 
                               recurrent_regularizer=l2(self.kernel_regularizer),
                               bias_regularizer=l2(self.bias_regularizer),
                               return_sequences=True,
                               name=f'encoder_blsmt_1'))(encoder_dynamic_inputs)
        encoder1 = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_1')(encoder1)
        encoder1 = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_1')(encoder1)
        encoder2 = Bidirectional(LSTM(self.encoder_hidden_layers[1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_2'))(encoder1)
        encoder2 = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'encoder_blsmt_BN_2')(encoder2)
        encoder2 = Dropout(self.dynamic_dropout,
                            name=f'encoder_blsmt_dropout_2')(encoder2)
        encoder3 = Bidirectional(LSTM(self.encoder_hidden_layers[1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                   name=f'encoder_blsmt_3'))(encoder2)
        encoder3 = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_3')(encoder3)
        encoder3 = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_3')(encoder3)
        
        ###########################################################
        #################### ENCODER STATIC  ######################
        ###########################################################    
        if meta_shape>0:
            encoder_meta1 = Dense(self.static_FN_depth[0],
                           activation=self.FN_activation,
                           name='encoder_static_FN_1')(encoder_static_inputs)
            encoder_meta1 = Dropout(self.static_dropout,name='encoder_static_FN_dropout_0')(encoder_meta1)
            encoder_meta2 = Dense(self.encoder_hidden_layers[1]*2,
                           activation=self.FN_activation,
                           name=f'encoder_static_FN_2')(encoder_meta1)
            encoder_meta2 = Dropout(self.static_dropout,name=f'encoder_static_FN_dropout_2')(encoder_meta2)
        
        ###########################################################
        ################### ENCODER Attention #####################
        ###########################################################  
        
        layer = MultiHeadAttention(num_heads=4, 
                                   key_dim=6)
        attn_output_tensor, attn_weights = layer(encoder2,
                                                  encoder3,
                                                  return_attention_scores=True)
        
        
        
        ###########################################################
        #################### ENCODER COMBINE ######################
        ###########################################################      
        merged_encoder = concatenate([encoder3,
                                      attn_output_tensor],name='encoder_combined_data') # (samples, 101)
        merged_encoder1 = Dense(self.combied_FN_depth[0],
                               activation=self.FN_activation,
                               name=f'encoder_combined_FC_1')(merged_encoder)
        merged_encoder1 = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput_1')(merged_encoder1)
        
        merged_encoder2 = Dense(self.combied_FN_depth[1],
                               activation=self.FN_activation,
                               name=f'encoder_combined_FC_2')(merged_encoder1)
        merged_encoder2 = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput_2')(merged_encoder2)
        
        merged_encoder3 = Flatten()(merged_encoder2) 
        if meta_shape>0:
            merged_encoder3 = concatenate([merged_encoder3,encoder_meta2])
        merged_encoder3 = Dense(self.combied_FN_depth[-1],
                               activation=self.FN_activation,
                               name=f'encoder_combined_FC_3')(merged_encoder3)

        ###########################################################
        #################### ENCODER BUILD ######################
        if meta_shape>0:
            self.encoder_model =  Model(inputs= [encoder_dynamic_inputs,encoder_static_inputs],outputs = [merged_encoder3],name='Encoder_model')
        else:
            self.encoder_model =  Model(inputs= [encoder_dynamic_inputs],outputs = [merged_encoder3],name='Encoder_model')
        
        if self.verbose>0:
            print('Encoder| built')
        
        ###########################################################
        #################### DECODER  ######################
        ########################################################### 

        decoder_inputs = Input(shape=(merged_encoder3.shape[1:]),name='decoder_input') 
        
        decoder_inputs_rep = RepeatVector(20)(decoder_inputs)
        
        if self.verbose>0:
            print('Decoder| building')
            

        decoder_1 = Bidirectional(LSTM(self.decoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'decoder_blsmt_1'))(decoder_inputs_rep)
        decoder_1 = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'decoder_blsmt_BN_1')(decoder_1)
        decoder_1 = Dropout(self.decoder_dropout,
                        name=f'decoder_blsmt_dropout_1')(decoder_1)
        
        decoder_2 = Bidirectional(LSTM(self.decoder_hidden_layers[1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'decoder_blsmt_2'))(decoder_1)
        decoder_2 = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'decoder_blsmt_BN_2')(decoder_2)
        decoder_2 = Dropout(self.decoder_dropout,
                            name=f'decoder_blsmt_dropout_2')(decoder_2)
            

        pred = Flatten()(decoder_2)
        pred = (Dense(self.input_shape[1],name='Prediction'))(pred)
        if self.verbose>0:
            print('Decoder| built')
        ###########################################################
        #################### DECODER BUILD  ######################
        ########################################################### 
        
        self.decoder_model = Model(inputs = [decoder_inputs],outputs=[pred],name='Decoder_model')
        if meta_shape>0:
            encoder_outputs = self.encoder_model([encoder_dynamic_inputs,encoder_static_inputs])
        else:
            encoder_outputs = self.encoder_model([encoder_dynamic_inputs])
            
        decoder_outputs = self.decoder_model(encoder_outputs)
        if self.verbose>0:
            print('EncoderDecoder| building')
        Name = f"{datetime.datetime.now().strftime('%Y%m%d-%H')}_{self.encoder_decoder_name}"
        if meta_shape>0:
            self.model = Model(inputs=[encoder_dynamic_inputs,encoder_static_inputs],outputs = [decoder_outputs],name=Name)  
        else:
            self.model = Model(inputs=[encoder_dynamic_inputs],outputs = [decoder_outputs],name=Name)  
        
        if self.verbose>0:
            print('EncoderDecoder| built')

        if self.verbose>1:
            trainable_count = count_params(self.model.trainable_weights)
            print(f'Autoencoder| trainable parameters: {trainable_count}')

        
        return self.model
    
    
    
    
class article_model:
    '''
    
    '''
    
    def __init__(self,
                 name ='AIS_forcast',
                 inputsize=(60,5),
                 output_size = (1,5),
                 encoder_stack_hidden_neurons=[356,256],
                 global_dropout = 0.4,
                 blstm_kernel_initializer = None,
                 attention_type='soft',
                 self_attention_heads=1,
                 FC_neurons = 56,
                 FC_activation = 'relu',
                 init_lr = 0.0001,
                 optimizer = 'adam',
                 verbose=1,
                 GPU=3):
        '''
        
        '''
        
        self.name = name
        self.inputsize=inputsize
        self.outputsize = output_size
        self.global_dropout = global_dropout
        self.blstm_kernel_initializer = blstm_kernel_initializer
        self.encoder_stack_hidden_neurons = encoder_stack_hidden_neurons
        self.attention_type = attention_type
        self.model = None
        self.self_attention_heads = self_attention_heads
        self.FC_activation = FC_activation
        self.FC_neurons = FC_neurons
        self.kernel_initializer = None
        self.optimizer = optimizer
        self.GPU = str(GPU)
        self.strategy = None
        self.callbacks = None
        self.init_lr = init_lr
        self.training_history = None
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None
        self.verbose = verbose
        self.mdn_mixers = None
        
        self.haversine_regularization=0.0000001
        
        
    def prepare_HPC(self):
        '''
            Just preparing the HPC and GPU yo.
            make sure GPUS are available, e.g. by 
            import os 
            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
            #os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU
        '''
        
        
        
        #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy().scope()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        #print(gpus)
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
        except RuntimeError as e:
            print(e)
           
        self.strategy = tf.device(f"/GPU:{self.GPU}")
        
    def check_req(self):
        '''
            Quick check to see if parms er ready
        '''
        
        # Attention type
        if self.attention_type.lower() =='self':
            if (self.self_attention_heads is None) or (self.self_attention_heads<1):
                print('Number of heads must be specified when using Multi-headed Self-attention. Default of 1 is picked.')
                self.self_attention_heads = 1
        
        if self.attention_type.lower() =='soft':
            if (self.self_attention_heads is not None):  
                if (self.self_attention_heads>0):
                    print(f'Self_attention_heads is set to {self.self_attention_heads} by user. Soft-attention is used, and no heads will be used by the model.')
                    self.self_attention_heads = None
                    
                    
        # Activation functions            
        allowed_activations = ['relu','sigmoid','softmax','tanh','selu','elu','exponential']
        
        self.lr_multipliers = {'blstm_1': 0.5}
        if self.FC_activation.lower() in allowed_activations:
            self.FC_activation = self.FC_activation.lower()
            pass
        else:
            raise ValueError(f'{self.FC_activation} not allowed as activation function')
        
        if self.FC_activation.lower() =='selu':
            self.kernel_initializer = 'lecun_normal'
        else:
            self.kernel_initializer = 'GlorotNormal'
        
        
        #Optimizers
        allowed_optimizers = ['adam','nadam','wadam','cosine']
        
        if self.optimizer.lower() in allowed_optimizers:
            if self.optimizer.lower() == 'adam':
                self.optimizer = 'Adam'
            
            if self.optimizer.lower() =='nadam':
                self.optimizer = 'Nadam'
            if self.optimizer.lower() =='cosine':
                os.environ["TF_KERAS"] = '1'
        else:
            raise ValueError(f'{self.optimizer} not allowed as optimizer. Allowed optimizers are {allowed_optimizers}')
            
        try:
            keras.backend.clear_session()
        except:
            pass

    def load_data(self,
                  X_train,
                  Y_train,
                  X_val,
                  Y_val,
                  X_test,
                  Y_test):
        '''
        
        '''
        try:
            self.X_train = np.load(X_train)
            self.Y_train = np.load(Y_train)
        except:
            self.X_train = X_train
            self.Y_train = Y_train
        else:
            print('error. No training data')
        try:
            self.X_val = np.load(X_val)
            self.Y_val = np.load(Y_val)
        except:
            self.X_val = X_val
            self.Y_val = Y_val
        else:
            print('error. No validation data')
            
        try:
            self.X_test = np.load(X_test)
            self.Y_test = np.load(Y_test)
        except:
            self.X_test = X_test
            self.Y_test = Y_test
        else:
            print('Warning. No testing data')

    def article_blstm_mdn(self):
        encoder_inputs = Input(shape=(self.inputsize),name='blstm_mdn_input')
        N_mixers = 11
        Dropout_val = 0.3
        x = Bidirectional(LSTM(456, 
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.00001),
                            activity_regularizer=tf.keras.regularizers.l2(0.000001),
                            return_sequences=True))(encoder_inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = Dropout(Dropout_val)(x)
        x = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(x)
        x = Bidirectional(LSTM(456, 
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.001),
                            activity_regularizer=tf.keras.regularizers.l2(0.001),
                            return_sequences=True))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = Dropout(Dropout_val)(x)
        x = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(x)


        y = Bidirectional(LSTM(456,
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.001),
                            activity_regularizer=tf.keras.regularizers.l2(0.001), 
                            return_sequences=False))(x)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(y)
        y = Dropout(Dropout_val)(y)
        y = Bidirectional(LSTM(256,
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.001),
                            activity_regularizer=tf.keras.regularizers.l2(0.001), 
                            return_sequences=False))(x)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(y)
        y = Dropout(Dropout_val)(y)


        z = Bidirectional(LSTM(456,
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                            activity_regularizer=tf.keras.regularizers.l2(0.0001), 
                            return_sequences=True))(x)
        z = tf.keras.layers.LeakyReLU(alpha=0.1)(z)
        z = Dropout(Dropout_val)(z)
        z = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(z)
        z = Bidirectional(LSTM(256,
                            kernel_initializer = tf.keras.initializers.GlorotNormal(),
                            kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                            activity_regularizer=tf.keras.regularizers.l2(0.0001),
                            return_sequences=False))(z)
        z = tf.keras.layers.LeakyReLU(alpha=0.1)(z)
        z = Dropout(Dropout_val)(z)
        z = BatchNormalization(momentum=0.99, 
                                        scale=True, 
                                        center=True,
                                        trainable=False)(z)
        pred = Average()([z,y])

        pred = mdn.MDN(self.input_size[1], N_mixers)(pred)
        name=f"BLSTM_MDN_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=encoder_inputs, outputs=pred,name=name)





    def model_ablstm(self):
        '''
            Defining a model
        '''
        
        
        inputs = tf.keras.Input(shape=self.inputsize,
                                name='encoder_input')
        #print(type(inputs),inputs.shape)
        x = Masking(mask_value=-999,input_shape=self.inputsize)(inputs)
        #print(type(x),x.shape)
        count = 0
        #Encoder
        for hidden_neurons in self.encoder_stack_hidden_neurons[0:-1]:
            x = Bidirectional(LSTM(hidden_neurons,
                                   kernel_initializer = tf.keras.initializers.GlorotNormal(),
                                   name=f'encoder_blstm_{count}',
                                   return_sequences=True))(x)
            x = Dropout(self.global_dropout)(x)
            count+=1       
        #print(x.shape)
        
        if self.attention_type.lower() == 'self':
            y = Bidirectional(LSTM(self.encoder_stack_hidden_neurons[-1],
                               kernel_initializer = tf.keras.initializers.GlorotNormal(),
                               name=f'encoder_blstm_last',
                               return_sequences=True))(x)
            y = Dropout(self.global_dropout)(y)
            layer = MultiHeadAttention(num_heads=self.self_attention_heads, key_dim=6)
            target = Input(shape=self.inputsize)
            source = Input(shape=self.inputsize)
            attn_output_tensor1, attn_weights = layer(inputs, 
                                                     inputs,
                                                     return_attention_scores=True)
            attn_output_tensor2, attn_weights = layer(inputs, 
                                                     inputs,
                                                     return_attention_scores=True)
            attn_output_tensor = Average()([attn_output_tensor1, attn_output_tensor2])
            
        elif self.attention_type.lower() == 'soft':
            y = Bidirectional(LSTM(self.encoder_stack_hidden_neurons[-2],
                               kernel_initializer = tf.keras.initializers.GlorotNormal(),
                               name=f'encoder_blstm_last',
                               return_sequences=True))(x)
            y = Dropout(self.global_dropout)(y)
            #atte = Dense(56)(y)
            #atte = Dense(self.inputsize[1])(y)
            #print(y.shape,inputs.shape)
            #print(y.shape,x.shape)
            attn_output_tensor = tf.keras.layers.Attention()([x,y])
            #attn_out = tf.keras.layers.Attention()([x, z])

            #attn_out = Flatten()(attn_out)
            print(attn_output_tensor.shape)
            #print(attn_output_tensor.shape)
            #layer = MultiHeadAttention(num_heads=2,key_dim=2)
            #target = Input(shape=self.inputsize)
            #source = Input(shape=self.inputsize)
            #print(source.shape,attn.shape)
            
            #attn_output_tensor, attn_weights = layer(attn, 
            #                                         source,
            #                                         return_attention_scores=True)
        #print(attn_output_tensor.shape,y.shape)
        out = Concatenate()([y, 
                             attn_output_tensor])
        # FC LAYERS
        if (self.outputsize[0])<2:
            out = Flatten()(out)
            if len(self.FC_neurons)>0:
                for FC_layer in self.FC_neurons:
                    out = Dense(FC_layer,
                               kernel_initializer = self.kernel_initializer,
                               activation=self.FC_activation)(out)
            pred = Dense(self.inputsize[1],
                         kernel_initializer = self.kernel_initializer,
                         activation=self.FC_activation)(out)
            
        if (self.outputsize[0])>1:
            if len(self.FC_neurons)>0:
                for FC_layer in self.FC_neurons:
                    out = Dense(FC_layer,
                               kernel_initializer = self.kernel_initializer,
                               activation=self.FC_activation)(out)
            pred = Dense(self.inputsize[1],
                         kernel_initializer = self.kernel_initializer,
                         activation=self.FC_activation)(out)
            
        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=inputs, 
                           outputs=pred,
                           name=self.name)
        self.lr_multipliers = {'encoder_blstm_1': 0.5}
        
        
        
    def model_ablstm_sim(self):
        '''
            Defining a model
        '''
        
        
        inputs = tf.keras.Input(shape=self.inputsize,
                                name='encoder_input')
        x = Masking(mask_value=-999,input_shape=self.inputsize)(inputs)
        count = 0
        for hidden_neurons in self.encoder_stack_hidden_neurons:
            x = Bidirectional(LSTM(hidden_neurons,
                                   kernel_initializer = tf.keras.initializers.GlorotNormal(),
                                   name=f'encoder_blstm_{count}',
                                   return_sequences=True))(x)
            x = Dropout(self.global_dropout)(x)
            count+=1     
            
        y = Bidirectional(LSTM(self.encoder_stack_hidden_neurons[-1],
                               kernel_initializer = tf.keras.initializers.GlorotNormal(),
                               name=f'encoder_blstm_last',
                               return_sequences=True))(x)
        y = Dropout(self.global_dropout)(y)    
        
        attn_output_tensor = tf.keras.layers.Attention()([x,y])
        print(y.shape,attn_output_tensor.shape)

        out = Concatenate()([y,
                             attn_output_tensor])
        print(out.shape)
        # FC LAYERS
        if len(self.outputsize)<2:
            out = Flatten()(out)
            for FC_layer in self.FC_neurons:
                out = Dense(FC_layer,
                           kernel_initializer = self.kernel_initializer,
                           activation=self.FC_activation)(out)
            pred = Dense(self.inputsize[1],
                         kernel_initializer = self.kernel_initializer,
                         activation=self.FC_activation)(out)
            
        if len(self.outputsize)>1:
            for FC_layer in self.FC_neurons:
                out = Dense(FC_layer,
                           kernel_initializer = self.kernel_initializer,
                           activation=self.FC_activation)(out)
            print(out.shape)
            pred = Dense(self.inputsize[1],
                         kernel_initializer = self.kernel_initializer,
                         activation=self.FC_activation)(out)
            print(pred.shape)
            
        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=inputs, 
                           outputs=pred,
                           name=self.name)
        self.lr_multipliers = {'encoder_blstm_1': 0.5}
        
    def model_thesis(self):
        model = thesis_model(inputsize = self.inputsize,
                                    outputsize = self.outputsize,
                                    model_name='Thesis_model')
        self.model = model.build_model()
        self.lr_multipliers = {'bidirectional_1': 0.5}
        
    def model_encoderdecoder(self,
                             verbose=1,
                             meta_shape=0):
        '''
        
                             encoder_decoder_name = 'Encoder_Decoder',
                             encoder_hidden_layers = [256,256],
                             decoder_hidden_layers = [256,256],
                             kernel_initializer ='glorot_uniform' , 
                             kernel_regularizer = 0.0001,
                             bias_regularizer = 0.0001,
                             recurrent_regularizer = 0.0001,
                             dynamic_dropout = 0.2,
                             combiend_dropout = 0.2,
                             static_dropout = 0.2,
                             decoder_dropout = 0.2,
                             BN_momentum = 0.99,
                             static_FN_depth = [50,50],
                             combied_FN_depth = [64,64],
                             decoder_FC_depth = [64,64],
                             FN_activation= 'relu',
                             
        '''
        modellen = EncoderDecoderAttention(
                                           verbose=verbose)
        '''
        encoder_decoder_name,
                                           encoder_hidden_layers,
                                           decoder_hidden_layers,
                                           kernel_initializer , 
                                           kernel_regularizer,
                                           bias_regularizer,
                                           recurrent_regularizer,
                                           dynamic_dropout,
                                           combiend_dropout,
                                           static_dropout,
                                           decoder_dropout,
                                           BN_momentum,
                                           static_FN_depth,
                                           combied_FN_depth,
                                           decoder_FC_depth,
                                           FN_activation,
        '''
        #modellen.enocoder_decoder_inputs()
        self.model = modellen.build_autoencoder_attention(input_shape = self.inputsize,meta_shape = meta_shape)

        self.lr_multipliers = {'bidirectional_1': 0.5}
        #mod.layers[2].bidirectional_1
        
    def model_blstm(self):
        
        '''
            Defining a blstm model.
            Eveything is static, as to ensure this same model is used for every comparison.
        '''
        
        
        inputs = tf.keras.Input(shape=self.inputsize,
                                name='blstm_input')
        #LSTM 1
        x = Bidirectional(LSTM(256,
                 kernel_initializer = 'lecun_normal',
                 name=f'blstm_1',
                 return_sequences=True))(inputs)
        #x = BatchNormalization(momentum=0.99, 
        #                           scale=True, 
        #                           center=True,
        #                           trainable=False)(x)
        x = Dropout(self.global_dropout)(x)
        #LSTM 2
        x = Bidirectional(LSTM(256,
                 kernel_initializer = 'lecun_normal',
                 name=f'blstm_2',
                 return_sequences=True))(x)
        #x = BatchNormalization(momentum=0.99, 
        #                           scale=True, 
        #                           center=True,
        #                           trainable=False)(x)
        x = Dropout(self.global_dropout)(x)
        
        y = Bidirectional(LSTM(156,
                 kernel_initializer = 'lecun_normal',
                 name=f'blstm_2',
                 return_sequences=True))(x)
        #y = BatchNormalization(momentum=0.99, 
        #                           scale=True, 
        #                           center=True,
        #                           trainable=False)(y)
        y = Dropout(self.global_dropout)(y)
        y = Bidirectional(LSTM(256,
                 kernel_initializer = 'lecun_normal',
                 name=f'blstm_2',
                 return_sequences=True))(x)
        #y = BatchNormalization(momentum=0.99, 
        #                           scale=True, 
        #                           center=True,
        #                           trainable=False)(y)
        y = Dropout(self.global_dropout)(y)
        
        attn_out = tf.keras.layers.Attention()([x, y])
        attn_out = Dropout(0.4)(attn_out)
        attn_out = Flatten()(attn_out)
        y_out = Flatten()(y)
        out = Concatenate()([attn_out,y_out])
        pred = Dense(self.inputsize[1], 
                     kernel_initializer='lecun_normal',
                     activation='selu')(out)
        #FC 
        #FC = Dense(56,
        #           kernel_initializer = 'GlorotNormal',
        #           activation='relu')(out)
        # 
        #out = Flatten()(y)
        ##Prediction
        #pred = Dense(self.inputsize[1],
        #             kernel_initializer = 'GlorotNormal',
        #             activation='relu')(out)

        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=inputs, 
                           outputs=pred,
                           name=self.name)
        self.lr_multipliers = {'blstm_1': 0.5}
        
        
        
    def model_rnn(self):
        '''
            Defining a vanialle RNN model.
            Eveything is static, as to ensure this same model is used for every comparison.
        '''
        
        
        inputs = tf.keras.Input(shape=self.inputsize,
                                name='lstminput')
        #LSTM 1
        x = RNN(128,
                 kernel_initializer = 'GlorotNormal',
                 name=f'rnn_1',
                 return_sequences=True)(x)
        x = Dropout(self.global_dropout)(x)
        #LSTM 2
        x = RNN(56,
                 kernel_initializer = 'GlorotNormal',
                 name=f'rnn_2',
                 return_sequences=False)(x)
        x = Dropout(self.global_dropout)(x)
        #FC 
        FC = Dense(56,
                   kernel_initializer = 'GlorotNormal',
                   activation='relu')(out)
        
        out = Flatten()(y)
        #Prediction
        pred = Dense(self.inputsize[1],
                     kernel_initializer = 'GlorotNormal',
                     activation='relu')(out)

        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=inputs, 
                           outputs=pred,
                           name=self.name)
        self.lr_multipliers = {'rnn_1': 0.5}
        
        
    def model_cnn(self,conv_layers = [56,56,56,56]):
        
        self.conv_layers = conv_layers 
        inputs = Input(shape=(self.inputsize))
        model2D = Conv1DTranspose(filters=self.inputsize[1], kernel_size=128)(inputs)
        #inputs = Reshape((20,10,1))(inputs)
        
        for convl in conv_layers:
            model2D = Conv1D(convl, kernel_size = 3)(model2D)
            model2D = LeakyReLU()(model2D)
            model2D = MaxPooling1D(1)(model2D)
            model2D = Dropout(0.05)(model2D)

        out = Flatten()(model2D)
        #out = Dense(56)(out)
        out = Dropout(0.01)(out)
        try:
            out = Dense(self.outputsize[0]*self.outputsize[1])(out)
            out = Reshape((self.outputsize))(out)
        except:
            out = Dense(self.inputsize[1])(out)
        #out = Reshape((1,self.inputsize[1]))(out)
        
        
        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        
        
        self.model = Model(inputs=inputs,outputs=out,name=self.name)
        self.lr_multipliers = {'cnn_1': 0.5}
    
        
        
    def model_compiler(self):
        '''
            Compiling a model
        '''
        if self.optimizer=='Nadam':
            self.optimizer = keras.optimizers.Nadam(lr=self.init_lr,clipnorm=1, clipvalue=1.0)
        elif self.optimizer=='Adam':
            self.optimizer = keras.optimizers.Adam(lr=self.init_lr,amsgrad=True,clipnorm=1, clipvalue=1.0)
        elif self.optimizer.lower() == 'cosine':
            self.optimizer = tf.optimizers.Adam(amsgrad=True,clipnorm=1, clipvalue=1.0)
            lr_metric = get_lr_metric(self.optimizer)
        
        layer_names = []
        for layer in self.model.layers:
            layer_names.append(layer.name)
        
        if next((True for layer in layer_names if 'mdn' in layer), False):
            print('mdn in model')
            try:
                with self.strategy:
                    #self.model.compile(loss=[tf.keras.losses.MeanAbsoluteError()],
                    #                   optimizer=self.optimizer,
                    #                   metrics=[tf.keras.metrics.MeanAbsoluteError()])
                    #loss=[Haversine_loss(6356.752)]
                    #self.optimizer = tf.optimizers.Adam(amsgrad=True,clipnorm=1, clipvalue=1.0)
                    lr_metric = get_lr_metric(self.optimizer)
                    self.model.compile(loss=mdn.get_mixture_loss_func(self.inputsize[1],self.mdn_mixers), 
                                       optimizer=self.optimizer,
                                        metrics=[rmse_mdn(), 
                                                 rmse_mdn_coord(), 
                                                 rmse_mdn_lat(), 
                                                 rmse_mdn_lon(),
                                                 lr_metric()])
            except:
                try:
                    #self.optimizer = tf.optimizers.Adam(amsgrad=True,clipnorm=1, clipvalue=1.0)
                    lr_metric = get_lr_metric(self.optimizer)
                    
                    self.model.compile(loss=mdn.get_mixture_loss_func(self.inputsize[1],self.mdn_mixers), 
                                       optimizer=self.optimizer,
                                       metrics=[lr_metric])
                
                except:
                    #self.optimizer = tf.optimizers.Adam(amsgrad=True,clipnorm=1, clipvalue=1.0)
                    lr_metric = get_lr_metric(self.optimizer)
                    
                    self.model.compile(loss=mdn.get_mixture_loss_func(self.inputsize[1],self.mdn_mixers), 
                                       optimizer=self.optimizer)
                    
                    
        else:
            try:
                with self.strategy:
                    #self.model.compile(loss=[tf.keras.losses.MeanAbsoluteError()],
                    #                   optimizer=self.optimizer,
                    #                   metrics=[tf.keras.metrics.MeanAbsoluteError()])
                    #loss=[Haversine_loss(6356.752)]
                    self.model.compile(loss=[tf.keras.losses.MeanAbsoluteError()],
                                   optimizer=self.optimizer,
                                   metrics=[Haversine(6356.752),
                                            rmse_coordinates(),
                                            diff_sog(),
                                            diff_cog(),
                                            diff_dist(),
                                            diff_lat(),
                                            diff_long(),
                                            tf.keras.metrics.RootMeanSquaredError(),
                                            lr_metric])
            except:
                try:
                    self.model.compile(loss=[Haversine_loss(6356.752)],
                                   optimizer=self.optimizer,
                                   metrics=[Haversine(6356.752),
                                            rmse_coordinates(),
                                            diff_sog(),
                                            diff_cog(),
                                            diff_dist(),
                                            tf.keras.metrics.MeanSquaredLogarithmicError(),
                                            tf.keras.metrics.MeanSquaredError(),
                                            tf.keras.metrics.SquaredHinge(),
                                            tf.keras.metrics.RootMeanSquaredError(),
                                            lr_metric])
                
                except:
                    print('there is an error yo')
                    #self.model.compile(loss=[tf.keras.losses.MeanAbsoluteError()],
                    #               optimizer=self.optimizer,
                    #               metrics=[tf.keras.metrics.MeanAbsoluteError()])
            

    def get_callbacks(self,
                      steps_per_epoch=10,
                      wd_norm=0.004,
                      eta_min=0.000002,
                      eta_max=1,
                      eta_decay=0.7,
                      cycle_length=10,
                      cycle_mult_factor=2.5):
        '''
        '''
        os.makedirs(f"{self.model.name}", exist_ok=True) 
        #name = f"{self.model.name}_par_{self.model.count_params()}"
        best_model_file = f"{self.model.name}/best_model_{self.model.name}.h5"

        early_stop = EarlyStopping(monitor='val_loss', patience=50)
    
        best_model = ModelCheckpoint(best_model_file, 
                                     monitor='val_loss', 
                                     mode='auto',
                                     verbose=0, 
                                     save_best_only=True)
        #this is not really used, since Cosine Annealing is used. Still, this ia added to disaply Learning rate. Easiest way yo.
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        #                              factor=0.1,
        #                              patience=100, 
        #                              min_lr=1e-12)
        

          
        cbk = CustomModelCheckpoint()       
        time_callback = TimeHistory()
        cb_wrwd = WRWDScheduler(steps_per_epoch=steps_per_epoch, 
                                lr=self.init_lr,
                                wd_norm=wd_norm,
                                eta_min=eta_min,
                                eta_max=eta_max,
                                eta_decay=eta_decay,
                                cycle_length=cycle_length,
                                cycle_mult_factor=cycle_mult_factor)
        #change_parms_cb = change_parms(self.model)
        
        try:
            self.callbacks=[best_model,
                            early_stop,
                            cb_wrwd,
                            cbk]
        except:
            self.callbacks=[best_model,
                            early_stop,
                            cb_wrwd,
                            cbk]
        
    
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None
             
    def model_external(self,
                       model):
        '''
        
        '''
        self.model = model
        
        
    def model_external_mdn(self,
                           model,
                           mdn_mixers = 3):
        '''
        
        '''
        self.mdn_mixers = mdn_mixers
        self.model = model
        
        
    def model_fit(self,
                  batch_size:int=200,
                  epochs:int=300,
                  verbose:int=1,
                  continue_training:int=None):
        
        if continue_training== None:
            training_history = self.model.fit(self.X_train, 
                                              self.Y_train,
                                              validation_data  = (self.X_val,
                                                                  self.Y_val),
                                              epochs=epochs,
                                              verbose=verbose,
                                              shuffle = True,
                                              batch_size=batch_size,
                                              callbacks = self.callbacks)
        else:
            training_history = self.model.fit(self.X_train, 
                                              self.Y_train,
                                              validation_data  = (self.X_val,
                                                                  self.Y_val),
                                              epochs=epochs,
                                              verbose=verbose,
                                              shuffle = True,
                                              batch_size=batch_size,
                                              initial_epoch = continue_training,
                                              callbacks = self.callbacks)
        
        return training_history
        
        
    def get_statistics(self,statistics):
        '''
        
        '''
        return None

    
    
    
#########################################################
#########################################################

    def model_blstm_mdn(self):
        '''
            Defining a model
        '''
        
        
        inputs = tf.keras.Input(shape=self.inputsize,
                                name='encoder_input')
        #print(type(inputs),inputs.shape)
        x = Masking(mask_value=-999,input_shape=self.inputsize)(inputs)
        #print(type(x),x.shape)
        count = 0
        #Encoder
        for hidden_neurons in self.encoder_stack_hidden_neurons[0:-1]:
            x = Bidirectional(LSTM(hidden_neurons,
                                   kernel_initializer = tf.keras.initializers.GlorotNormal(),
                                   name=f'encoder_blstm_{count}',
                                   return_sequences=True))(x)
            x = Dropout(self.global_dropout)(x)
            count+=1       
        #print(x.shape)
        
        if self.attention_type.lower() == 'self':
            y = Bidirectional(LSTM(self.encoder_stack_hidden_neurons[-1],
                               kernel_initializer = tf.keras.initializers.GlorotNormal(),
                               name=f'encoder_blstm_last',
                               return_sequences=True))(x)
            y = Dropout(self.global_dropout)(y)
            #layer = MultiHeadAttention(num_heads=self.self_attention_heads, key_dim=2)
            #target = Input(shape=self.inputsize)
            #ource = Input(shape=self.inputsize)
            #attn_output_tensor1, attn_weights = layer(inputs, 
            #                                         inputs,
            #                                         return_attention_scores=True)
            #attn_output_tensor2, attn_weights = layer(inputs, 
            #                                         inputs,
            #                                         return_attention_scores=True)
            #attn_output_tensor = Average()([attn_output_tensor1, attn_output_tensor2])
            
        elif self.attention_type.lower() == 'soft':
            y = Bidirectional(LSTM(self.encoder_stack_hidden_neurons[-1],
                               kernel_initializer = tf.keras.initializers.GlorotNormal(),
                               name=f'encoder_blstm_last',
                               return_sequences=True))(x)
            y = Dropout(self.global_dropout)(y)
            #atte = Dense(56)(y)
            #atte = Dense(self.inputsize[1])(y)
            #print(y.shape,inputs.shape)
            #print(y.shape,x.shape)
            #attn_output_tensor = tf.keras.layers.Attention()([y, x])
            #print(attn_output_tensor.shape)
            #print(attn_output_tensor.shape)
            #layer = MultiHeadAttention(num_heads=2,key_dim=2)
            #target = Input(shape=self.inputsize)
            #source = Input(shape=self.inputsize)
            #print(source.shape,attn.shape)
            
            #attn_output_tensor, attn_weights = layer(attn, 
            #                                         source,
            #                                         return_attention_scores=True)
        #print(attn_output_tensor.shape,y.shape)
        #out = Concatenate()([y, 
        #                     attn_output_tensor])
        # FC LAYERS
        out = Flatten()(out)
        
        
        
        
        for FC_layer in self.FC_neurons:
            out = Dense(FC_layer,
                       kernel_initializer = self.kernel_initializer,
                       activation=self.FC_activation)(out)
        pred = Dense(self.inputsize[1],
                     kernel_initializer = self.kernel_initializer,
                     activation=self.FC_activation)(out)
        self.name=f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        self.model = Model(inputs=inputs, 
                           outputs=pred,
                           name=self.name)
        self.lr_multipliers = {'encoder_blstm_1': 0.5}



def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value

def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    The eager mode in tensorflow 2.0 makes is extremely easy to write 
    functions like these. It feels a lot more pythonic to me.
    """
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)


# Mean
#mu = Dense((n_feat * k), activation=None, name='mean_layer')(layer2)
# variance (should be greater than 0 so we exponentiate it)
#var_layer = Dense(k, activation=None, name='dense_var_layer')(layer2)
#var = Lambda(lambda x: exp(x), output_shape=(k,), name='variance_layer')(var_layer)
# mixing coefficient should sum to 1.0
#pi = Dense(k, activation='softmax', name='pi_layer')(layer2)
'''
def calc_pdf(y, mu, var):
    """Calculate component density"""
    y = tf.reshape(y , (161,19,1))
    mu =  tf.reshape(mu ,(161,19,19)) #batch, input features, output features
    value = tf.subtract(y, mu)**2
    return value


def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    The eager mode in tensorflow 2.0 makes is extremely easy to write 
    functions like these. It feels a lot more pythonic to me.
    """
    pi = tf.reshape(pi , (161,300,1))
    var =  tf.reshape(var ,(161,300,26))
    
    
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)
'''





#########################################################
#########################################################

            
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
        
        
        
class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(LossCallback, self).__init__()
        model.beta_x = tf.Variable(1.0, trainable=False, name='weight1', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.beta_x, tf.constant(0.5) * epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['beta_x'] = tf.keras.backend.get_value(self.model.beta_x)

    
        
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_count = 0
            self.learning_rates = []
            
            
            
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch>0:
                if self.model.history.history['val_loss'][-1] == min(self.model.history.history['val_loss']):
                    self.model.save_weights(f'{self.model.name}/weights_lowest_loss_{self.model.name}.h5', overwrite=True)
                    pd.DataFrame(self.model.history.history).to_pickle(f"{self.model.name}/history_lowest_loss_epoch_{self.model.name}.pkl")
                    
                lr = K.get_value(self.model.optimizer.lr)
                self.learning_rates.append(lr)
                self.model.save(f'{self.model.name}/model_{self.model.name}_newest_epoch.h5', overwrite=True)
                self.model.save_weights(f'{self.model.name}/weights_{self.model.name}_newest_epoch.h5', overwrite=True)
                pd.DataFrame(self.model.history.history).to_pickle(f"{self.model.name}/history_newest_epoch_{self.model.name}.pkl")
                
                
############## COSTUME METRICS #######################   

def rmse_mdn():
    @tf.autograph.experimental.do_not_convert
    def get_rmse_mdn(y_true, y_pred):
        N_mixes = 3
        N_features = 7
        print(y_pred.shape,y_true.shape)
        print('to tensor')
        y_true = K.ones_like(y_true)
        y_pred = K.ones_like(y_pred)
        print('sample from output')
        y_pred = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, N_features, N_mixes, temp=1.0)
        print('to tensor againg')
        y_pred = K.ones_like(y_pred)
        #y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        #y_true = K.squeeze(y_true, axis=-2)
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    return get_rmse_mdn

def rmse_mdn_coord():
    @tf.autograph.experimental.do_not_convert
    def get_rmse_mdn_coord(y_true, y_pred):
        N_mixes = 3
        N_features = 7
        y_pred = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, N_features, N_mixes, temp=1.0)
        
        y_true = y_true[:,0:2]
        y_pred = y_pred[:,0:2]
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    return get_rmse_mdn_coord

def rmse_mdn_lat():
    @tf.autograph.experimental.do_not_convert
    def get_rmse_mdn_lat(y_true, y_pred):
        N_mixes = 3
        N_features = 7
        y_pred = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, N_features, N_mixes, temp=1.0)
        
        y_true = y_true[:,0:1]
        y_pred = y_pred[:,0:1]
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    return get_rmse_mdn_lat

def rmse_mdn_lon():
    @tf.autograph.experimental.do_not_convert
    def get_rmse_mdn_lon(y_true, y_pred):
        N_mixes = 3
        N_features = 7
        y_pred = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, N_features, N_mixes, temp=1.0)
        
        y_true = y_true[:,1:2]
        y_pred = y_pred[:,1:2]
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    return get_rmse_mdn_lon

def mdn_var():
    @tf.autograph.experimental.do_not_convert
    def get_mdn_var(y_true, y_pred):
        return y_pred[1]
    return get_mdn_var



@tf.autograph.experimental.do_not_convert        
def rmse_coordinates():
    @tf.autograph.experimental.do_not_convert
    def RMSE_distance(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,0:2]
        y_pred = y_pred[:,0:2]
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    return RMSE_distance
    
@tf.autograph.experimental.do_not_convert    
def diff_sog():  
    @tf.autograph.experimental.do_not_convert
    def sog_rmse(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,2:3]
        y_pred = y_pred[:,2:3]
        return K.square((K.mean((y_pred - y_true))))
    return sog_rmse

@tf.autograph.experimental.do_not_convert
def diff_long():        
    @tf.autograph.experimental.do_not_convert
    def long_rmse(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,0:1]
        y_pred = y_pred[:,0:1]
        return K.square((K.mean((y_pred - y_true))))
    return long_rmse

@tf.autograph.experimental.do_not_convert
def diff_lat():        
    @tf.autograph.experimental.do_not_convert
    def lat_rmae(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,1:2]
        y_pred = y_pred[:,1:2]
        return K.square((K.mean((y_pred - y_true))))
    return lat_rmae   

@tf.autograph.experimental.do_not_convert
def diff_cog():        
    @tf.autograph.experimental.do_not_convert
    def cog_rmse(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,3:4]
        y_pred = y_pred[:,3:4]
        return K.square((K.mean((y_pred - y_true))))
    return cog_rmse 

def diff_dist():        
    @tf.autograph.experimental.do_not_convert
    def distance_rmse(y_true, y_pred):
        #y_true = K.squeeze(y_true, axis=-2)
        y_true = y_true[:,4:5]
        y_pred = y_pred[:,4:5]
        return K.square((K.mean((y_pred - y_true))))
    return distance_rmse

@tf.autograph.experimental.do_not_convert
def Haversine(R):
    # (where R is the radius of the Earth)
    @tf.autograph.experimental.do_not_convert
    def Haversine_distance(y_true,y_pred):  
        # y[Longitude,Latitude] change it accordingly
        #R = 6356.752 #POLAR radius in m.. Equatorial is 6378
        #y_true = K.squeeze(y_true, axis=-2)
        dlon=y_true[:,0:1]-y_pred[:,0:1]
        dlat=y_true[:,1:2]-y_pred[:,1:2]
        a=K.square(K.sin(dlat/2))+K.cos(y_true[1])*K.cos(y_pred[1])*K.square(K.sin(dlon)/2)
        c=2*(tf.atan2(K.sqrt(a),K.sqrt(1-a)))
        return K.abs(c*R)*1000
    return Haversine_distance



@tf.autograph.experimental.do_not_convert    
def Haversine_loss(R):
    # (where R is the radius of the Earth)
    @tf.autograph.experimental.do_not_convert
    def loss_function(y_true,y_pred):  
        # y[Longitude,Latitude] change it accordingly
        #R = 6356.752 #POLAR radius in m.. Equatorial is 6378
        #print(y_true.shape,y_pred.shape)
        #y_true = K.squeeze(y_true, axis=-2)
        dlon=y_true[:,0:1]-y_pred[:,0:1]
        dlat=y_true[:,1:2]-y_pred[:,1:2]
        dsog=y_true[:,2:3]-y_pred[:,2:3]
        dcog=y_true[:,3:4]-y_pred[:,3:4]
        #ddist=y_true[:,4:]-y_pred[:,4:]
        a=K.square(K.sin(dlat/2))+K.cos(y_true[1])*K.cos(y_pred[1])*K.square(K.sin(dlon)/2)
        c=2*(tf.atan2(K.sqrt(a),K.sqrt(1-a)))
        haversine = K.abs(c*R)
        mae = tf.keras.losses.MeanAbsoluteError()
        
        
        mae_loss = mae(y_true, y_pred)
        rmse_loss = K.square(mae_loss)
        
        error = y_true[:,0:2]-y_pred[:,0:2]        #square of the error
        #error = y_true-y_pred
        sqr_error = K.square(error)    #mean of the square of the error
        mean_sqr_error = K.mean(sqr_error)    #square root of the mean of the square of the error
        sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)    #return the error
        
        #haversine_loss = K.square(haversine)
        #print(haversin.shape,dlon.shape,dlat.shape,dsog.shape,dcog.shape)
        
        #haversine_combiend_loss = K.sqrt(K.mean(K.square(haversin)+K.square(dsog)+K.square(dcog)))
        
        #OLD
        #haversine_combiend_loss = K.sqrt(K.mean(K.square(haversin)**2+K.square(dsog)**2+0.25*K.square(dcog)**2+0.25*K.square(ddist)**2))
        #haversine_combiend_loss = K.sqrt(0.25*K.mean(K.square(haversin)**2+0.25*K.square(dsog)**2+0.25*K.square(dcog)**2+0.25*K.square(ddist)**2))
        #return 0.0001* haversine + rmse_loss
        #0.0001*haversine+ rmse_loss
        #+0.001*rmse_loss+0.0000001*haversine
        return sqrt_mean_sqr_error+0.001*rmse_loss
    
    
    
    return loss_function 


        
from tensorflow.keras.optimizers import schedules

def get_lr_metric(optimizer):
    @tf.autograph.experimental.do_not_convert  
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
       
class WRWDScheduler(keras.callbacks.Callback):
    """
    """
    
    @tf.autograph.experimental.do_not_convert
    def __init__(self,
                 steps_per_epoch,
                 lr,
                 wd_norm=0.004,
                 eta_min=0.000002,
                 eta_max=1,
                 eta_decay=0.7,
                 cycle_length=10,
                 cycle_mult_factor=2.5):
        """Constructor for warmup learning rate scheduler
        """

        super(WRWDScheduler, self).__init__()
        self.lr = lr
        self.wd_norm = wd_norm

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

        self.wd = wd_norm / (steps_per_epoch*cycle_length)**0.5

        self.history = defaultdict(list)
        self.batch_count = 0
        self.learning_rates = []
        
        self.batch_count = 0
        self.learning_rates = []

        
        
        
        
        
    
    @tf.autograph.experimental.do_not_convert 
    def cal_eta(self):
        '''Calculate eta'''
        fraction_to_restart = self.steps_since_restart / (self.steps_per_epoch * self.cycle_length)
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(fraction_to_restart * np.pi))
        return eta
    
    @tf.autograph.experimental.do_not_convert 
    def on_train_batch_begin(self, batch, logs={}):
        '''update learning rate and weight decay'''
        eta = self.cal_eta()
        #self.model.optimizer._learning_rate = eta * self.lr
        lr = eta * self.lr
        
        K.set_value(self.model.optimizer.lr, lr)
        #self.model.optimizer._weight_decay = eta * self.wd
    @tf.autograph.experimental.do_not_convert
    def on_train_batch_end(self, batch, logs={}):
        '''Record previous batch statistics'''
        logs = logs or {}
        
        #self.history['wd'].append(self.model.optimizer.optimizer._weight_decay)
        for k, v in logs.items():
            self.history[k].append(v)

        self.steps_since_restart += 1
    @tf.autograph.experimental.do_not_convert
    def on_epoch_end(self, epoch, logs={}):
        
        '''Check for end of current cycle, apply restarts when necessary'''
        def on_epoch_end(self, epoch, logs=None):
            print(K.eval(self.model.optimizer.lr))
            
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
        #self.model.history.history.append(lr)
        
        
        self.history['lr'].append(lr)
        self.learning_rates.append(lr)
        
        
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            self.wd = self.wd_norm / (self.steps_per_epoch*self.cycle_length)**0.5    
            


class use_model():
    '''
    
    '''
    def __init__(self,
                 model,
                 true_targets,
                 true_samples,
                 samples_description,
                 model_features,
                 statistics_training
                ):
        
        self.model = model
        self.targets_testing = true_targets
        self.samples_testing = true_samples
        
        self.corrected_target = None # sample ID corrected normalized
        self.corrected_samples = None
        
        #self.sample_ID = None
        self.predicted_samples = None
        self.samples_description = samples_description
        self.statistics_training = statistics_training
        self.model_features = model_features
        
        self.future_steps = None
        self.ID = None
        self.samples_to_be_predicted = None
        self.trajectory_start = None
        
        
        self.weights_l = None
        self.weights_heatmap_l = None
        self.layer_name_l = None
        self.model_type = None
        layer_names=[layer.name for layer in self.model.layers]
        
        self.input_shape = self.model.input.shape
        if 'mdn' in layer_names:
            self.model_type = 'mdn'
            self.n_mixers = int(self.model.layers[-1].output_shape[-1]/(4*2+1))
            print(f'n mixers: {self.n_mixers}')
        

        
    def predict_future(self):
        '''
    
        '''
        input_sample = self.samples_to_be_predicted.squeeze()
        lookback = input_sample.shape[0]
        features = input_sample.shape[1]
        current_input = input_sample
        if self.future_steps>0:
            for k in range(self.future_steps):
                
                this_input = current_input[-lookback:,:]
                #print(this_input.shape)
                this_input_pred = np.reshape(this_input,(1,lookback,features))
                #print(this_input_pred.shape)
                #print(this_input_pred.shape)
                this_prediction = self.model.predict(this_input_pred)  
                if self.model_type =='mdn':
                    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, 
                                                    this_prediction, 
                                                    self.input_shape[-1], 
                                                    self.n_mixers, temp=1.0)

                
                
                #print(this_prediction.shape)
                #print(current_input.shape,this_prediction.shape)
                current_input = np.vstack((current_input,this_prediction.squeeze()))
            predicted_track =   current_input 
        else:
            this_input_pred = np.reshape(input_sample,(1,-1))
            predicted_track =   self.model.predict(this_input_pred)  
            
        
        self.predicted_samples = predicted_track

    
    
    
    def predict_test_data(self,
                          ID_number,
                          future_steps=100,
                          trajectory_start=20,norm_which='minmax'):
        '''
        
        '''
        self.future_steps = future_steps
        self.ID = ID_number
        self.trajectory_start = trajectory_start
        
        Ids_idx = [x for x, z in enumerate(self.samples_description.values) if z == 'Ids'] 
        
        
        samples_id = self.samples_testing[self.samples_testing[:,:,Ids_idx[0]]==self.ID]
        targets_id = self.targets_testing[self.targets_testing[:,:,Ids_idx[0]]==self.ID]
        #print(samples_id.shape)
        #samples_id = test.samples_training[test.samples_training[:,:,4]==ID_number]
        #targets_id = test.samples_training[test.samples_training[:,:,4]==ID_number]
        samples_id_start = samples_id[self.trajectory_start-20:self.trajectory_start,:]
        print(targets_id.shape)
        #print(targets_id.shape)
        #print(samples_id_start.shape)
        #print(samples_id_start.shape)
        true_sample_features = []
        true_target_features = []
        #print(samples_id_start.shape)
        i = 0
        for par in self.samples_description:
            #k = test.samples_description[par]
            #ind = list(filter(lambda i: list(test.samples_description==par)[i], range(len(list(test.samples_description==par)))))
            if norm_which.lower() =='minmax':
                #print(targets_id[:,i].shape)
                #print(par)
                #print(self.statistics_training[par]['min'],self.statistics_training[par]['max'])
                true_sample_features.append(np.array(samples_id_start[:,i]*(self.statistics_training[par]['max']-self.statistics_training[par]['min'])+self.statistics_training[par]['min']))
                true_target_features.append(np.array(targets_id[:,i]*(self.statistics_training[par]['max']-self.statistics_training[par]['min'])+self.statistics_training[par]['min']))
                #print((targets_id[:,i]*(self.statistics_training[par]['max']-self.statistics_training[par]['min'])+self.statistics_training[par]['min'])).shape
            elif norm_which.lower() =='zscore':
                true_sample_features.append((samples_id_start[:,i]*self.statistics_training[par]['std']+self.statistics_training[par]['mean']))
                true_target_features.append((targets_id[:,i]*self.statistics_training[par]['std']+self.statistics_training[par]['mean']))
            i=i+1
        #print(len(true_target_features))
        print(np.array(true_target_features).shape)
        self.corrected_target =np.array(true_target_features)
        #print(self.corrected_target.shape)
        self.corrected_samples = np.array(true_sample_features)

        li = (self.samples_testing[:,:,4]==self.ID)[:,0]
        
        te = [i for i, x in enumerate(li) if x]
        
        sample_id = self.samples_testing.squeeze()[li[0]:li[0]+1,:,self.model_features]
        self.samples_to_be_predicted = sample_id
        #pred= usres = [x for x, z in enumerate(lst) if z == 10] e_model.predict_future(model,sample_id,future_steps=future_steps)
        use_model.predict_future(self)


        pred_sample_features = []
        i = 0
        for par in self.model_features:
            k = self.samples_description[par]
            #ind = list(filter(lambda i: list(test.samples_description==par)[i], range(len(list(test.samples_description==par)))))
            pred_sample_features.append((self.predicted_samples[:,i]*(self.statistics_training[k]['max']-self.statistics_training[k]['min'])+self.statistics_training[k]['min']))
            i=i+1
            
        self.predicted_samples = np.array(pred_sample_features)
    
    
        #return true_sample_features, true_target_features, pred_sample_features
    
    
    
    
    def plot_prediction(self,save=False):
        '''
    
        '''
        import os
        try:
            os.environ["CARTOPY_USER_BACKGROUNDS"] = "../Visualization"
        except:
            try:
                os.environ["CARTOPY_USER_BACKGROUNDS"] = "Visualization"
            except:
                pass
        
                
            
        #import seaborn as sns
        #seaborn.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
        projections = [ccrs.PlateCarree(),
                       ccrs.Robinson(),
                       ccrs.Mercator(),
                       ccrs.Orthographic(),
                       ccrs.InterruptedGoodeHomolosine()
                      ]



        cmap_reversed = cm.get_cmap('magma_r')

        fig = plt.subplots(figsize=(22,12))
        ax = plt.axes(projection=projections[1])
        #ax.set_extent([pred[:,0].min()-6,pred[:,0].max()+3,pred[:,1].min()-5,pred[:,1].max()+2], crs=ccrs.PlateCarree())
        #ax.set_extent([ pred_sample_features[0,:].min()-1, pred_sample_features[0,:].max()+1,pred_sample_features[1,:].min()-1, pred_sample_features[1,:].max()+1])
        try:
            ax.set_extent([ self.corrected_target[0,:].min()-1, self.corrected_target[0,:].max()+1,self.corrected_target[1,:].min()-1, self.corrected_target[1,:].max()+1])
        except:
            ax.set_extent([ self.predicted_samples[0,:].min()-1, self.predicted_samples[0,:].max()+1,self.predicted_samples[1,:].min()-1, self.predicted_samples[1,:].max()+1])
            
        
        #ax.stock_img()


        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.background_img(name='BM', resolution='high')
        ax.add_feature(cfeature.LAND)
        #ax.add_feature(cfeature.COASTLINE)
        ax.coastlines()
        

        #plt.scatter(samples[:20,0],samples[:20,1],s=10,c='black',transform=ccrs.PlateCarree(),label='Input track')
        #plt.scatter(samples[20:,0],samples[20:,1],s=10,c='green',alpha=0.7,cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='Future track')
        #plt.scatter(samples[0,0],samples[0,1],color='red',s=35,cmap=cmap_reversed,transform=ccrs.PlateCarree())
        cmap=cm.get_cmap("hsv",360)
        #plt.scatter(pred_long[:,0],pred_lat[:,0],s=10,color='darkorange',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='Predicted track')

        plt.scatter(self.predicted_samples[0,20:],self.predicted_samples[1,20:],s=10,color='darkorange',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='Predicted track')
        #plt.scatter(pred_sample_features[0,:20],pred_sample_features[1,:20],s=10,color='darkorange',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='input')
        plt.scatter(self.corrected_target[0,:],self.corrected_target[1,:],s=10,color='green',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='True track')
        plt.scatter(self.corrected_samples[0,:20],self.corrected_samples[1,:20],s=10,color='black',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='Input track')
        #plt.scatter(self.corrected_samples[0,:],self.corrected_samples[1,:],s=10,color='red',cmap=cmap,vmin=0, vmax=360,transform=ccrs.PlateCarree(),label='Input track')
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_left = False
        #gl.xlines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'gray'}
        gl.xlabel_style = {'color': 'black'} #, 'weight': 'bold'
        gl.ylabel_style = {'size': 20, 'color': 'gray'}
        gl.ylabel_style = {'color': 'black'} #, 'weight': 'bold'

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 25}
        gl.ylabel_style = {'size': 25}
        plt.legend(loc="lower left", fontsize=25,markerscale=3)
        
        

        
        if save==True:
            try:
                plt.savefig(f"{predict_model.model.name}/Trajectory_prediction_ID{ID}.png",
                            bbox_inches ="tight",
                            pad_inches = 1,
                            transparent = True,
                            orientation ='landscape',dpi=720)
            except:
                try:
                    plt.savefig(f"../AIS_prediction/{predict_model.model.name}/Trajectory_prediction_ID{ID}.png",
                            bbox_inches ="tight",
                            pad_inches = 1,
                            transparent = True,
                            orientation ='landscape',dpi=720)
                except:
                    pass
            
        
    

    def activation_grad(model,
                        samples_to_be_predicted,
                        layer_name='bidirectional_1'):
        #print(samples_to_be_predicted.shape)
        #seq = seq[np.newaxis,:,:]
        grad_model = Model([model.inputs], 
                           [model.get_layer(f'{layer_name}').output, 
                            model.output])    # Obtain the predicted value and the intermediate filters
        with tf.GradientTape() as tape:
            seq_outputs, predictions = grad_model(samples_to_be_predicted)    # Extract filters and gradients
        seq_outputs.shape
        output = seq_outputs[0]
        #print(predictions)
        grads = tape.gradient(predictions, seq_outputs)[0]    # Average gradients spatially
        #print(grads)
        weights = tf.reduce_mean(grads, axis=0)    # Get a ponderaated map of filters according to grad importance
        cam = np.ones(output.shape[0], dtype=np.float32)
        for index, w in enumerate(weights):
            cam += w * output[:, index]    
        time = int(samples_to_be_predicted.shape[1]/output.shape[0])
        cam = ndimage.zoom(cam.numpy(), time, order=1)
        heatmap = (cam - cam.min())/(cam.max() - cam.min())
        return weights, heatmap
    
    def get_activation_grad(self):
        layer_names=[layer.name for layer in self.model.layers]
        weights_l = []
        weights_heatmap_l = []
        layer_name_l = []
        for layer_name in layer_names:
            if layer_name.startswith( 'drop' ) ==False and layer_name.startswith( 'batch' ) ==False:
                try:
                    weights, heatmap = use_model.activation_grad(self.model,
                                                         self.samples_to_be_predicted,
                                                         layer_name=layer_name)
                    #print(self.samples_to_be_predicted.shape)
                    #vars()[layer_name+'weights'] = weights
                    #vars()[layer_name+'weights_heatmap'] = heatmap
                    weights_l.append(weights)
                    weights_heatmap_l.append(heatmap)
                    layer_name_l.append(layer_name)
                except:
                    print(f'cant do {layer_name}')


        self.weights_heatmap_l = weights_heatmap_l
        self.weights_l = weights_l
        self.layer_name_l = layer_name_l
        
    def plot_activation_grads(self):
        '''
        
        '''
        name_idx = 0
        for heatmap in self.weights_heatmap_l:
            X = np.linspace(1,self.samples_to_be_predicted.shape[1],self.samples_to_be_predicted.shape[1])
            fig, ax = plt.subplots(figsize=(12, 8))
            #print(X.shape)
            #rint(heatmap.shape)
            for i in range(self.samples_to_be_predicted.shape[-1]):
                #print(self.samples_to_be_predicted[0,:,i].shape)
                #plt.scatter(X,test.samples_testing.squeeze()[sample_time:sample_time+1,:,model_features].squeeze()[:,i],alpha=0.3,label=test.samples_description[model_features][i])
                plt.scatter(X,self.samples_to_be_predicted[0,:,i],alpha=0.3,label=i)
            plt.scatter(X,heatmap,label='gradients',c='black')
            plt.title(self.layer_name_l[name_idx])
            name_idx = name_idx+1

    

        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()   
        
        
        
        
class Encoder_prediction:
    '''
    '''
    def __init__(self, 
                 encoder_name:str = 'Encoder_Decoder',
                 encoder_hidden_layers:list = [256,128],
                 kernel_initializer:str ='glorot_uniform', 
                 kernel_regularizer:float = 0.0001,
                 bias_regularizer:float = 0.0001,
                 recurrent_regularizer:float = 0.0001,
                 dynamic_dropout:float = 0.2,
                 combiend_dropout:float = 0.2,
                 static_dropout:float = 0.2,
                 BN_momentum:float = 0.99,
                 static_FN_depth:list = [200],
                 combied_FN_depth:list = [200],
                 FN_activation:str = 'relu',
                 sampels_shape_features:int = 5,
                 samples_shape_length:int = 20,
                 meta_shape:int = 1923,
                 using_attention:bool = True,
                 verbose:int = 0):
        '''
        
        '''
        self.encoder_name = encoder_name
        self.encoder_hidden_layers = encoder_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer =kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.dynamic_dropout = dynamic_dropout
        self.BN_momentum = BN_momentum
        self.static_FN_depth = static_FN_depth
        self.combied_FN_depth = combied_FN_depth
        self.combiend_dropout = combiend_dropout
        self.static_dropout = static_dropout
        self.FN_activation = FN_activation
        self.sampels_shape_features = sampels_shape_features
        self.samples_shape_length = samples_shape_length
        self.meta_shape = meta_shape
        self.encoder_dynamic_inputs = None
        self.encoder_static_inputs = None
        self.using_attention= using_attention
        self.verbose = verbose
        
        try:
            keras.backend.clear_session()
        except:
            pass
        
    def enocoder_inputs(self):
        '''
        '''
        self.encoder_dynamic_inputs = Input(shape=(self.samples_shape_length,
                                                   self.sampels_shape_features),
                                            name='encoder_dynamic_inputs')
        
        self.encoder_static_inputs = Input(shape=(self.meta_shape,),name=f'encoder_static_inputs')
        
        
    def build_encoder(self):
        '''
        '''
        #build BLSTM part for dynamic inputs
        if self.verbose>0:
            print('Encoder| building')
        i = None

        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
            
        else:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
        
        if len(self.encoder_hidden_layers)>2:
            for i in range(1,len(self.encoder_hidden_layers)-1):
                x = Bidirectional(LSTM(self.encoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_{i}'))(x)
                x = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'encoder_blsmt_BN_{i}')(x)
                x = Dropout(self.dynamic_dropout,
                            name=f'encoder_blsmt_dropout_{i}')(x)
            
        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[-1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                   name=f'encoder_blsmt_{len(self.encoder_hidden_layers)}'))(x)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_{len(self.encoder_hidden_layers)}')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_{len(self.encoder_hidden_layers)}')(x)
            
            
        #encoder_outputs, state_h, state_c = x(self.encoder_dynamic_inputs)
        #print(encoder_outputs_and_states.shape)
        # Build FC part for static inptus
        x_meta = Dense(self.static_FN_depth[0],
                       activation=self.FN_activation,
                       name='encoder_static_FN_0')(self.encoder_static_inputs)
        x_meta = Dropout(self.static_dropout,name='encoder_static_FN_dropout_0')(x_meta)

        if len(self.static_FN_depth)>1:
            for i in range(1, len(self.static_FN_depth)):
                x_meta = Dense(self.static_FN_depth[i],
                               activation=self.FN_activation,
                               name=f'encoder_static_FN_{i}')(x_meta)
                x_meta = Dropout(self.static_dropout,name=f'encoder_static_FN_dropout_{i}')(x_meta)
            
        #combining static and dynamic data:    
        merged_encoder = concatenate([x, x_meta],name='encoder_combined_data') # (samples, 101)
        for i in range(len(self.combied_FN_depth)):
            merged_encoder = Dense(self.combied_FN_depth[i],
                                   activation=self.FN_activation,
                                   name=f'encoder_combined_FC_{i}')(merged_encoder)
            merged_encoder = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput{i}')(merged_encoder)
        merged_encoder = Dense(self.sampels_shape_features,
                               activation=self.FN_activation,
                               name=f'prediction_FC')(merged_encoder)
        merged_encoder = Dropout(self.combiend_dropout,
                                 name=f'prediction_FC_droput')(merged_encoder)
        
        #co
        encoder_model =  Model(inputs= [self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [merged_encoder],name='Encoder_model')
        if self.verbose>0:
            print('Encoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(encoder_model.trainable_weights)
            print(f'Encoder| trainable parameters: {trainable_count}')
            
        return encoder_model
    
    

        
class Encoder_prediction:
    '''
    '''
    def __init__(self, 
                 encoder_name:str = 'Encoder_Decoder',
                 encoder_hidden_layers:list = [256,128],
                 kernel_initializer:str ='glorot_uniform', 
                 kernel_regularizer:float = 0.0001,
                 bias_regularizer:float = 0.0001,
                 recurrent_regularizer:float = 0.0001,
                 dynamic_dropout:float = 0.2,
                 combiend_dropout:float = 0.2,
                 static_dropout:float = 0.2,
                 BN_momentum:float = 0.99,
                 static_FN_depth:list = [200],
                 combied_FN_depth:list = [200],
                 FN_activation:str = 'relu',
                 sampels_shape_features:int = 5,
                 samples_shape_length:int = 10,
                 meta_shape:int = 1923,
                 using_attention:bool = True,
                 verbose:int = 0):
        '''
        
        '''
        self.encoder_name = encoder_name
        self.encoder_hidden_layers = encoder_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer =kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.dynamic_dropout = dynamic_dropout
        self.BN_momentum = BN_momentum
        self.static_FN_depth = static_FN_depth
        self.combied_FN_depth = combied_FN_depth
        self.combiend_dropout = combiend_dropout
        self.static_dropout = static_dropout
        self.FN_activation = FN_activation
        self.sampels_shape_features = sampels_shape_features
        self.samples_shape_length = samples_shape_length
        self.meta_shape = meta_shape
        self.encoder_dynamic_inputs = None
        self.encoder_static_inputs = None
        self.using_attention= using_attention
        self.verbose = verbose
        
        try:
            keras.backend.clear_session()
        except:
            pass
        
    def enocoder_inputs(self):
        '''
        '''
        self.encoder_dynamic_inputs = Input(shape=(self.samples_shape_length,
                                                   self.sampels_shape_features),
                                            name='encoder_dynamic_inputs')
        
        self.encoder_static_inputs = Input(shape=(self.meta_shape,),name=f'encoder_static_inputs')
        
        
    def build_encoder(self):
        '''
        '''
        #build BLSTM part for dynamic inputs
        if self.verbose>0:
            print('Encoder| building')
        i = None

        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
            
        else:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
        
        if len(self.encoder_hidden_layers)>2:
            for i in range(1,len(self.encoder_hidden_layers)-1):
                x = Bidirectional(LSTM(self.encoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_{i}'))(x)
                x = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'encoder_blsmt_BN_{i}')(x)
                x = Dropout(self.dynamic_dropout,
                            name=f'encoder_blsmt_dropout_{i}')(x)
            
        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[-1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                   name=f'encoder_blsmt_{len(self.encoder_hidden_layers)}'))(x)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_{len(self.encoder_hidden_layers)}')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_{len(self.encoder_hidden_layers)}')(x)
            
            
        #encoder_outputs, state_h, state_c = x(self.encoder_dynamic_inputs)
        #print(encoder_outputs_and_states.shape)
        # Build FC part for static inptus
        x_meta = Dense(self.static_FN_depth[0],
                       activation=self.FN_activation,
                       name='encoder_static_FN_0')(self.encoder_static_inputs)
        x_meta = Dropout(self.static_dropout,name='encoder_static_FN_dropout_0')(x_meta)

        if len(self.static_FN_depth)>1:
            for i in range(1, len(self.static_FN_depth)):
                x_meta = Dense(self.static_FN_depth[i],
                               activation=self.FN_activation,
                               name=f'encoder_static_FN_{i}')(x_meta)
                x_meta = Dropout(self.static_dropout,name=f'encoder_static_FN_dropout_{i}')(x_meta)
            
        #combining static and dynamic data:    
        merged_encoder = concatenate([x, x_meta],name='encoder_combined_data') # (samples, 101)
        for i in range(len(self.combied_FN_depth)):
            merged_encoder = Dense(self.combied_FN_depth[i],
                                   activation=self.FN_activation,
                                   name=f'encoder_combined_FC_{i}')(merged_encoder)
            merged_encoder = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput{i}')(merged_encoder)
        merged_encoder = Dense(self.sampels_shape_features,
                               activation=self.FN_activation,
                               name=f'prediction_FC')(merged_encoder)
        merged_encoder = Dropout(self.combiend_dropout,
                                 name=f'prediction_FC_droput')(merged_encoder)
        
        #co
        encoder_model =  Model(inputs= [self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [merged_encoder],name='Encoder_model')
        if self.verbose>0:
            print('Encoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(encoder_model.trainable_weights)
            print(f'Encoder| trainable parameters: {trainable_count}')
            
        return encoder_model
    
    

class Encoder_attention_prediction:
    '''
    '''
    def __init__(self, 
                 encoder_name:str = 'Encoder_Decoder',
                 encoder_hidden_layers:list = [256,128],
                 kernel_initializer:str ='glorot_uniform', 
                 kernel_regularizer:float = 0.0001,
                 bias_regularizer:float = 0.0001,
                 recurrent_regularizer:float = 0.0001,
                 dynamic_dropout:float = 0.2,
                 combiend_dropout:float = 0.2,
                 static_dropout:float = 0.2,
                 BN_momentum:float = 0.99,
                 static_FN_depth:list = [200],
                 combied_FN_depth:list = [200],
                 FN_activation:str = 'relu',
                 sampels_shape_features:int = 5,
                 samples_shape_length:int = 20,
                 meta_shape:int = 1923,
                 using_attention:bool = True,
                 verbose:int = 0,
                 attention_type:str='multihead'):
        '''
        
        '''
        assert (attention_type.upper() in ['MULTIHEAD', 'SOFT']),'wrong attentiontype'
        self.encoder_name = encoder_name
        self.encoder_hidden_layers = encoder_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer =kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.dynamic_dropout = dynamic_dropout
        self.BN_momentum = BN_momentum
        self.static_FN_depth = static_FN_depth
        self.combied_FN_depth = combied_FN_depth
        self.combiend_dropout = combiend_dropout
        self.static_dropout = static_dropout
        self.FN_activation = FN_activation
        self.sampels_shape_features = sampels_shape_features
        self.samples_shape_length = samples_shape_length
        self.meta_shape = meta_shape
        self.encoder_dynamic_inputs = None
        self.encoder_static_inputs = None
        self.using_attention= using_attention
        self.verbose = verbose
        self.attention_type = attention_type
        try:
            keras.backend.clear_session()
        except:
            pass
        
    def enocoder_attention_inputs(self):
        '''
        '''
        self.encoder_dynamic_inputs = Input(shape=(self.samples_shape_length,
                                                   self.sampels_shape_features),
                                            name='encoder_dynamic_inputs')
        
        self.encoder_static_inputs = Input(shape=(self.meta_shape,),name=f'encoder_static_inputs')
        
        
    def build_encoder_attention(self):
        '''
        '''
        ####################################
        #build BLSTM part for dynamic inputs
        ####################################
        if self.verbose>0:
            print('Encoder| building')
        i = None

        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
            
        else:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
        
        if len(self.encoder_hidden_layers)>2:
            for i in range(1,len(self.encoder_hidden_layers)-1):
                x = Bidirectional(LSTM(self.encoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_{i}'))(x)
                x = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'encoder_blsmt_BN_{i}')(x)
                x = Dropout(self.dynamic_dropout,
                            name=f'encoder_blsmt_dropout_{i}')(x)
            
        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[-1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                   name=f'encoder_blsmt_{len(self.encoder_hidden_layers)}'))(x)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_{len(self.encoder_hidden_layers)}')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_{len(self.encoder_hidden_layers)}')(x)
        ########################################
        #build attention part for dynamic inputs
        ########################################  
        
        if self.attention_type.upper() == 'SOFT':
            attn_out = tf.keras.layers.Attention()([x, self.encoder_dynamic_inputs])
        elif self.attention_type.upper() =='MULTIHEAD':
            attn_out = tf.keras.layers.Attention()([x, x])
            
            
        attn_out = Dropout(self.dynamic_dropout)(attn_out)
        attn_out = Flatten()(attn_out)
        
        
        #encoder_outputs, state_h, state_c = x(self.encoder_dynamic_inputs)
        #print(encoder_outputs_and_states.shape)
        # Build FC part for static inptus
        x_meta = Dense(self.static_FN_depth[0],
                       activation=self.FN_activation,
                       name='encoder_static_FN_0')(self.encoder_static_inputs)
        x_meta = Dropout(self.static_dropout,name='encoder_static_FN_dropout_0')(x_meta)

        if len(self.static_FN_depth)>1:
            for i in range(1, len(self.static_FN_depth)):
                x_meta = Dense(self.static_FN_depth[i],
                               activation=self.FN_activation,
                               name=f'encoder_static_FN_{i}')(x_meta)
                x_meta = Dropout(self.static_dropout,name=f'encoder_static_FN_dropout_{i}')(x_meta)
            
        #combining static and dynamic data:    
        merged_encoder = concatenate([attn_out, x_meta],name='encoder_combined_data') # (samples, 101)    
        
        for i in range(len(self.combied_FN_depth)):
            merged_encoder = Dense(self.combied_FN_depth[i],
                                   activation=self.FN_activation,
                                   name=f'encoder_combined_FC_{i}')(merged_encoder)
            merged_encoder = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput{i}')(merged_encoder)
        merged_encoder = Dense(self.sampels_shape_features,
                               activation=self.FN_activation,
                               name=f'prediction_FC')(merged_encoder)
        merged_encoder = Dropout(self.combiend_dropout,
                                 name=f'prediction_FC_droput')(merged_encoder)
        
        #co
        encoder_model =  Model(inputs= [self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [merged_encoder],name='Encoder_model')
        if self.verbose>0:
            print('Encoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(encoder_model.trainable_weights)
            print(f'Encoder| trainable parameters: {trainable_count}')
            
        return encoder_model
    
    

        

            
from keras.utils.layer_utils import count_params

class EncoderDecoder:
    '''
    '''
    def __init__(self, 
                 encoder_decoder_name:str = 'Encoder_Decoder',
                 encoder_hidden_layers:list = [256,128],
                 decoder_hidden_layers:list = [256,128],
                 kernel_initializer:str ='glorot_uniform' , 
                 kernel_regularizer:float = 0.0001,
                 bias_regularizer:float = 0.0001,
                 recurrent_regularizer:float = 0.0001,
                 dynamic_dropout:float = 0.2,
                 combiend_dropout:float = 0.2,
                 static_dropout:float = 0.2,
                 decoder_dropout:float = 0.2,
                 BN_momentum:float = 0.99,
                 static_FN_depth:list = [200],
                 combied_FN_depth:list = [200],
                 decoder_FC_depth:list = [200],
                 FN_activation:str = 'relu',
                 sampels_shape_features:int = 5,
                 samples_shape_length:int = 10,
                 meta_shape:int = 1923,
                 using_attention:bool = False,
                 verbose:int = 0):
        '''
        
        '''
        self.encoder_decoder_name = encoder_decoder_name
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer =kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.dynamic_dropout = dynamic_dropout
        self.BN_momentum = BN_momentum
        self.static_FN_depth = static_FN_depth
        self.decoder_FC_depth = decoder_FC_depth
        self.combied_FN_depth = combied_FN_depth
        self.combiend_dropout = combiend_dropout
        self.decoder_dropout = decoder_dropout
        self.static_dropout = static_dropout
        self.FN_activation = FN_activation
        self.sampels_shape_features = sampels_shape_features
        self.samples_shape_length = samples_shape_length
        self.meta_shape = meta_shape
        self.decoder_inputs = None
        self.encoder_dynamic_inputs = None
        self.encoder_static_inputs = None
        self.using_attention= using_attention
        self.verbose = verbose
        
        try:
            keras.backend.clear_session()
        except:
            pass
        
        
    def enocoder_decoder_inputs(self):
        '''
        '''
        self.encoder_dynamic_inputs = Input(shape=(self.samples_shape_length,
                                                   self.sampels_shape_features),
                                            name='encoder_dynamic_inputs')
        
        self.encoder_static_inputs = Input(shape=(self.meta_shape,),name=f'encoder_static_inputs')
        
        #decoder input should be same size as encoder output. When using self-attention, two same-size vectors are used as input, therefore *2
        if self.using_attention==True:
            self.decoder_inputs = Input(shape=(self.combied_FN_depth[-1]*2),name='decoder_input')
        else:
            self.decoder_inputs = Input(shape=(self.combied_FN_depth[-1]),name='decoder_input')
            
        
            
    def build_encoder(self):
        '''
        '''
        #build BLSTM part for dynamic inputs
        if self.verbose>0:
            print('Encoder| building')
        i = None

        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
            
        else:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'encoder_blsmt_0'))(self.encoder_dynamic_inputs)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_0')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_0')(x)
        
        if len(self.encoder_hidden_layers)>2:
            for i in range(1,len(self.encoder_hidden_layers)-1):
                x = Bidirectional(LSTM(self.encoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'encoder_blsmt_{i}'))(x)
                x = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'encoder_blsmt_BN_{i}')(x)
                x = Dropout(self.dynamic_dropout,
                            name=f'encoder_blsmt_dropout_{i}')(x)
            
        if len(self.encoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.encoder_hidden_layers[-1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                   name=f'encoder_blsmt_{len(self.encoder_hidden_layers)}'))(x)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'encoder_blsmt_BN_{len(self.encoder_hidden_layers)}')(x)
            x = Dropout(self.dynamic_dropout,
                        name=f'encoder_blsmt_dropout_{len(self.encoder_hidden_layers)}')(x)
            
            
        #encoder_outputs, state_h, state_c = x(self.encoder_dynamic_inputs)
        #print(encoder_outputs_and_states.shape)
        # Build FC part for static inptus
        x_meta = Dense(self.static_FN_depth[0],
                       activation=self.FN_activation,
                       name='encoder_static_FN_0')(self.encoder_static_inputs)
        x_meta = Dropout(self.static_dropout,name='encoder_static_FN_dropout_0')(x_meta)

        if len(self.static_FN_depth)>1:
            for i in range(1, len(self.static_FN_depth)):
                x_meta = Dense(self.static_FN_depth[i],
                               activation=self.FN_activation,
                               name=f'encoder_static_FN_{i}')(x_meta)
                x_meta = Dropout(self.static_dropout,name=f'encoder_static_FN_dropout_{i}')(x_meta)
            
        #combining static and dynamic data:    
        merged_encoder = concatenate([x, x_meta],name='encoder_combined_data') # (samples, 101)
        for i in range(len(self.combied_FN_depth)):
            merged_encoder = Dense(self.combied_FN_depth[i],
                                   activation=self.FN_activation,
                                   name=f'encoder_combined_FC_{i}')(merged_encoder)
            merged_encoder = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput{i}')(merged_encoder)
        
        
        
        encoder_model =  Model(inputs= [self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [merged_encoder],name='Encoder_model')
        if self.verbose>0:
            print('Encoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(encoder_model.trainable_weights)
            print(f'Encoder| trainable parameters: {trainable_count}')
            
    
    

    

        y = RepeatVector(self.samples_shape_length)(self.decoder_inputs)
        #y = RepeatVector(self.samples_shape_length)(inputs)
        
        if self.verbose>0:
            print('Decoder| building')
            
        i = None
        if len(self.decoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.decoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'decoder_blsmt_0'))(y)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'decoder_blsmt_BN_0')(x)
            x = Dropout(self.decoder_dropout,
                        name=f'decoder_blsmt_dropout_0')(x)
            
        else:
            x = Bidirectional(LSTM(self.decoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'decoder_blsmt_0'))(y)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'decoder_blsmt_BN_0')(x)
            x = Dropout(self.decoder_dropout,
                        name=f'decoder_blsmt_dropout_0')(x)
        
        if len(self.decoder_hidden_layers)>2:
            for i in range(1,len(self.decoder_hidden_layers)-1):
                x = Bidirectional(LSTM(self.decoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'decoder_blsmt_{i}'))(x)
                x = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'decoder_blsmt_BN_{i}')(x)
                x = Dropout(self.decoder_dropout,
                            name=f'decoder_blsmt_dropout_{i}')(x)
            
        if len(self.decoder_hidden_layers)>1:
            x = Bidirectional(LSTM(self.decoder_hidden_layers[-1], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                   name=f'decoder_blsmt_{len(self.decoder_hidden_layers)}'))(x)
            x = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'decoder_blsmt_BN_{len(self.decoder_hidden_layers)}')(x)
            x = Dropout(self.decoder_dropout,
                        name=f'decoder_blsmt_dropout_{len(self.decoder_hidden_layers)}')(x)
            
        
        
            
        #FC part of decoder:    
        for i in range(len(self.decoder_FC_depth)):
            x = Dense(self.decoder_FC_depth[i],
                      activation=self.FN_activation,
                      name=f'decoder_combined_FC_{i}')(x)
            x = Dropout(self.combiend_dropout,
                        name=f'decoder_combined_FC_droput{i}')(x)
                
        x = Dense(self.sampels_shape_features,activation=self.FN_activation,name='decoder_prediction')(x)   
        
        
        decoder_model =  Model(inputs= [self.decoder_inputs],outputs = [x],name='Decoder_model')
        if self.verbose>0:
            print('Decoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(decoder.trainable_weights)
            print(f'Decoder| trainable parameters: {trainable_count}')
            
        
        #### 
        # combined
        ###
        if self.verbose>0:
            print('Encoder/Decoder| build')
        encoder_outputs = encoder_model([self.encoder_dynamic_inputs,self.encoder_static_inputs])
        decoder_outputs = decoder_model(encoder_outputs)

        if self.verbose>1:
            print(encoder_outputs.summary())
            print(decoder_outputs.summary())
            
        Name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.encoder_decoder_name}"
        #os.makedirs('trained_DL_models/'+Name, exist_ok=True) 
        #os.makedirs('trained_DL_models', exist_ok=True)
        
        Encoder_Decoder_model = Model(inputs=[self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [decoder_outputs],name=Name)
            
            
        if self.verbose>1:
            trainable_count = count_params(Encoder_Decoder_model.trainable_weights)
            print(f'Encoder/Decoder| trainable parameters: {trainable_count}')
        
        
        return Encoder_Decoder_model
    
    


            

            
  
            
  