from keras.layers import Input,Conv1D ,Lambda, TimeDistributed
from keras.models import Model
from keras import backend as K

def SMel(N_mels,wins,audio_win,audio_hop,alpha=1,scaler=None,amin=1e-10):
    """ Return SMel as a keras model
    Parameters
    ----------
    N_mels : int
        Number of mel bands

    wins : int
        Number of frames in each input

    audio_win : int
        Number of samples in each frame

    audio_hop : int
        Hop time (in samples) 
        
    alpha : float
        The input is multiplied by alpha  
        
    scaler : array or tuple
        If is not None, the output of the model is normalized by the scaler (minmax)        

    amin : float
        Minimum magnitud to calculate the logarithm
        

    Return
    ----------
    m : Model
        SMel keras model.

    """ 
    x = Input(shape=(wins,audio_win,1), dtype='float32') #(1025,50)

    y = TimeDistributed(Conv1D(N_mels,1024, strides=16, padding='same',use_bias=True))(x)

    y = Lambda(lambda x: x*x)(y)

    y = Lambda(lambda x: audio_win*K.mean(x,axis=2))(y)

    y = Lambda(lambda x: 10*K.log(K.maximum(amin, x*alpha))/K.log(10.))(y)

    if scaler is not None:
        y = Lambda(lambda x: 2*((x-scaler[0])/(scaler[1]-scaler[0])-0.5))(y)

    m = Model(inputs=x, outputs=y)
    
    return m
    