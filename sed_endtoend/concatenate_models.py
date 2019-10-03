from keras.layers import Input
from keras.models import Model
    
def concatenate(wins,audio_win,model_cnn,model_mel,sequence_samples=22050,frames=True):
    """ Makes a model that concatenates a Mel extraction (MST or SMel)
        model with a CNN model.
    Parameters
    ----------
    wins : int
        Number of frames (only for frames=True)

    audio_win : int
        Lenght of audio frame (only for frames=True)
        
    model_cnn : Model
        Keras model for CNN network
        
    model_mel : Model
        Keras model for MEL extraction network        

    sequence_samples : int
        Number of audio samples (only for frames=True)

    frames : bool
        If True, the input of the network is a matrix with the audio frames as columns
        If False, the input of the network is a vector with the audio samples.
        
    Return
    ----------
    m : Model
        Concatenated keras Model.

    """
    
    if frames:
        x = Input(shape=(wins,audio_win,1), dtype='float32')
    else:
        x = Input(shape=(sequence_samples,1), dtype='float32')        
    y_mel = model_mel(x)
    y = model_cnn(y_mel)
    m = Model(x,[y,y_mel])
    return m

def concatenate_stft(N_hops,N_freqs,model_cnn,model_mel):
    """ Makes a model that concatenates a Mel extraction (MST or SMel)
        model with a CNN model. The input of this model is the STFT.
    Parameters
    ----------
    N_hops : int
        Hop samples of the input

    N_freqs : int
        Frquency bins of the input
        
    model_cnn : Model
        Keras model for CNN network
        
    model_mel : Model
        Keras model for MEL extraction network        
        
    Return
    ----------
    m : Model
        Concatenated keras Model.

    """
    x = Input(shape=(N_hops,N_freqs), dtype='float32')    
    y_mel = model_mel(x)
    y = model_cnn(y_mel)
    m = Model(x,[y,y_mel])
    return m