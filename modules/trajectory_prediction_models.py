'''
Module with different deep learning models for AIS data.. Some models use both dynamic and static data, som use only dynamic. 

They are made and the hyperparamters can be changed. The models themselves are not optimized. Instead, they work as a sceleton for future models.


Classes:

Encoder_prediction:
    Encoder encoder. Takes input and finds a smaller representation using BLSTM 
    
Encoder_attention_prediction:
    Encodes data using BLSTM and uses either a soft attention layer or multi headed self-attention layer.

EncoderDecoder:
    EncoderDecoder using BLSTM.
    
EncoderDecoderAttention:
    EncoderDecoder using BLSTM and Attention.

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten,Concatenate,Attention,Add,BatchNormalization, RepeatVector, Input, Flatten
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, GRU, TimeDistributed, Masking, Dropout, concatenate, TimeDistributed
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
#general 
import numpy as np
import pandas as pd
import math
import os
import keras.backend as K
from sklearn.metrics import mean_squared_error
import datetime


import datetime, os
from keras.callbacks import LambdaCallback   


import random

import warnings
warnings.filterwarnings('ignore')

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
    
class EncoderDecoderAttention:
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
            
        
            
    def build_autoencoder_attention(self):
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
                                return_sequences=True,
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
            
            
       
        
        

    

    

        #y = RepeatVector(self.samples_shape_length)(self.decoder_inputs)
        #y = RepeatVector(self.samples_shape_length)(inputs)
        
        if self.verbose>0:
            print('Decoder| building')
            
        i = None
        z = Bidirectional(LSTM(self.decoder_hidden_layers[0], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=False,
                                name=f'decoder_blsmt_0'))(x)
        z = BatchNormalization(momentum=self.BN_momentum,
                                   scale=True, 
                                   center=True,
                                   trainable=False,
                                   name=f'decoder_blsmt_BN_0')(z)
        z = Dropout(self.decoder_dropout,
                        name=f'decoder_blsmt_dropout_0')(z)
        
        if len(self.decoder_hidden_layers)>1:
            for i in range(1,len(self.decoder_hidden_layers)-1):
                z = Bidirectional(LSTM(self.decoder_hidden_layers[i], 
                                kernel_initializer = self.kernel_initializer, 
                                kernel_regularizer=l2(self.recurrent_regularizer), 
                                recurrent_regularizer=l2(self.kernel_regularizer),
                                bias_regularizer=l2(self.bias_regularizer),
                                return_sequences=True,
                                name=f'decoder_blsmt_{i}'))(z)
                z = BatchNormalization(momentum=self.BN_momentum, 
                                       scale=True, 
                                       center=True,
                                       trainable=False,
                                       name=f'decoder_blsmt_BN_{i}')(z)
                z = Dropout(self.decoder_dropout,
                            name=f'decoder_blsmt_dropout_{i}')(z)
            
        
        
        attn_out = tf.keras.layers.Attention()([x, z])
        attn_out = Dropout(self.decoder_dropout)(attn_out)
        attn_out = Flatten()(attn_out)

        
        
        
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
        merged_encoder = concatenate([z,attn_out,x_meta],name='encoder_combined_data') # (samples, 101)
        
        for i in range(len(self.combied_FN_depth)):
            merged_encoder = Dense(self.combied_FN_depth[i],
                                   activation=self.FN_activation,
                                   name=f'encoder_combined_FC_{i}')(merged_encoder)
            merged_encoder = Dropout(self.combiend_dropout,
                                     name=f'encoder_combined_FC_droput{i}')(merged_encoder)
            
        
        merged_encoder = Flatten()(merged_encoder)

                
        pred = Dense(self.sampels_shape_features,activation=self.FN_activation,name='decoder_prediction')(merged_encoder)   
        
        Name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.encoder_decoder_name}"        
        Encoder_Decoder_Attention_model = Model(inputs=[self.encoder_dynamic_inputs,self.encoder_static_inputs],outputs = [pred],name=Name)
            
        if self.verbose>0:
            print('Decoder| built')
        
        if self.verbose>1:
            trainable_count = count_params(Encoder_Decoder_Attention_model.trainable_weights)
            print(f'Autoencoder| trainable parameters: {trainable_count}')
         
        

    
        
        
        return Encoder_Decoder_Attention_model
    
    


            

            
  
            

            
  