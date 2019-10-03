from keras.layers import Input, Conv1D, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization

def MST(N_mels,sequence_samples,audio_win,audio_hop):
    """ Return SMel as a keras model
    Parameters
    ----------
    N_mels : int
        Number of mel bands

    sequence_samples : int
        Number of samples in each input

    audio_win : int
        Number of samples in each frame

    audio_hop : int
        Hop time (in samples) 
        

    Return
    ----------
    m : Model
        SMel keras model.

    """ 
    x = Input(shape=(sequence_samples,1), dtype='float32')

    y = Conv1D(512,audio_win, strides=audio_hop, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256,3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv1D(N_mels,3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('tanh')(y)

    m = Model(inputs=x, outputs=y)

    return m
