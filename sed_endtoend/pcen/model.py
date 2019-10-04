from keras.layers import Input, Conv2D, MaxPooling2D,Conv1D,MaxPooling1D,AveragePooling1D,RNN,Layer
from keras.layers import Dense, Flatten,Lambda,Activation,TimeDistributed,LocallyConnected1D,Concatenate,LocallyConnected2D
from keras.layers.core import Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import Regularizer
from keras.constraints import NonNeg
import numpy as np


def pcen(input):
    x = input[0]
    M = input[1]
    r = K.variable(0.25)
    delta = K.variable(10.0)
    alpha2= K.variable(0.8)
    eps = K.variable(1e-6)
    return    K.pow(x/K.pow(eps+M,alpha2) + delta,r) - K.pow(delta,r) 


class PCEN(Layer):

    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(PCEN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alpha = self.add_weight(name='alpha', 
                                      shape=(input_shape[0][2],1),
                                      initializer='uniform',
                                      constraint=NonNeg(),
                                      trainable=True)
        self.delta = self.add_weight(name='delta', 
                                      shape=(input_shape[0][2],1),
                                      initializer='uniform',
                                      constraint=NonNeg(),
                                      trainable=True)
        self.r = self.add_weight(name='r', 
                                      shape=(input_shape[0][2],1),
                                      initializer='uniform',
                                      constraint=NonNeg(),
                                      trainable=True)
        super(PCEN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        S = inputs[0]
        S_smooth = inputs[1]
        #alpha = K.tile(self.alpha,(S.shape[1],1))
        alpha = K.repeat(self.alpha,S.shape[1])
        alpha = K.squeeze(alpha,-1)
        alpha = K.permute_dimensions(alpha,(1,0))
        delta = K.repeat(self.delta,S.shape[1])
        delta = K.squeeze(delta,-1)
        delta = K.permute_dimensions(delta,(1,0))
        r = K.repeat(self.r,S.shape[1])        
        r = K.squeeze(r,-1)
        r = K.permute_dimensions(r,(1,0))
        return K.pow(K.abs(K.abs(S/K.pow(self.eps+S_smooth,K.abs(alpha)) + delta)),K.abs(r)) - K.pow(K.abs(delta),K.abs(r))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class LPF(Layer):

    def __init__(self, b, units=128,**kwargs):
        #self.b = b
        self.state_size = units
        super(LPF, self).__init__(**kwargs)

    def build(self, input_shape):
       # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
       #                               initializer='uniform',
       #                               name='kernel')
       # self.recurrent_kernel = self.add_weight(
       #     shape=(self.units, self.units),
       #     initializer='uniform',
       #     name='recurrent_kernel')
        self.b = self.add_weight(shape=(input_shape[-1], ),
                                      initializer='uniform',
                                      name='kernel')        
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        output = self.b*inputs + (1-self.b)*prev_output
        return output, [output]

class DOT(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DOT, self).__init__(**kwargs)

   # def l2_reg(self,weight_matrix):
    #    return 0.01 * K.sum(K.square(weight_matrix-kernel))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      #regularizer=regularizers.l2(0.0001),
                                      #regularizer=self.l2_reg,
                                      constraint = NonNeg(),
                                      trainable=True)
        super(DOT, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

def SMel_PCEN(N_mels,wins,audio_win,audio_hop,alpha=1,scaler=None,amin=1e-10,filters=None,r =0.25,delta = 10,alpha2= 0.8,eps = 1e-6,n_freqs=513):

    x = Input(shape=(wins,n_freqs), dtype='float32') #(1025,50)
    

    layer= DOT(N_mels)
    y = layer(x)

    T = 0.06 * 22050 / 512.
    b = (np.sqrt(1 + 4* T**2) - 1) / (2 * T**2)
    cell = LPF(b,N_mels)
    layer = RNN(cell,return_sequences=True)
    M = layer(y)


    y = PCEN()([y,M])
    

    if scaler is not None:
        y = Lambda(lambda x: 2*((x-scaler[0])/(scaler[1]-scaler[0])-0.5))(y)

    m = Model(inputs=x, outputs=y)
    
    return m

    
def concatenate(wins,audio_win,model_cnn,model_mel,sequence_samples=22050,frames=True):
    if frames:
        x = Input(shape=(wins,audio_win,1), dtype='float32')
    else:
        x = Input(shape=(sequence_samples,1), dtype='float32')        
    y_mel = model_mel(x)
    y = model_cnn(y_mel)
    m = Model(x,[y,y_mel])
    return m
    
def concatenate_stft(N_hops,N_freqs,model_cnn,model_mel,sequence_samples=22050,frames=True):

    x = Input(shape=(N_hops,N_freqs), dtype='float32')    
    y_mel = model_mel(x)
    y = model_cnn(y_mel)
    m = Model(x,[y,y_mel])
    return m
    
