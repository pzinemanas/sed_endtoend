from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
eps = 1e-6

def F1(y_true, y_pred):
    """ Calculates F1 in a resolution given by the input vectors.
    Note that the resolution of the system has to be 1s, to calculates F1_1s
    Parameters
    ----------
    y_true : array
        Ground-truth of the data

    y_pred : array
        Prediction given by the model
        
    Return
    ----------
    Fmesaure : float
        F1 value.

    """    
    
    y_pred = (y_pred>0.5).astype(int)
    Ntp = np.sum(y_pred + y_true > 1)
    Ntn = np.sum(y_pred + y_true > 0)
    Nfp = np.sum(y_pred - y_true > 0)
    Nfn = np.sum(y_true - y_pred > 0)
    Nref = np.sum(y_true)
    Nsys = np.sum(y_pred)
    
    P = Ntp / float(Nsys + eps)
    R = Ntp / float(Nref + eps)

    Fmeasure = 2*P*R/(P + R + eps)
    return Fmeasure
    
## not linear! 
def ER(y_true, y_pred):
    """ Calculates ER in a resolution given by the input vectors.
    Note that the resolution of the system has to be 1s, to calculates ER_1s
    Parameters
    ----------
    y_true : array
        Ground-truth of the data

    y_pred : array
        Prediction given by the model
        
    Return
    ----------
    Fmesaure : float
        ER value.

    """ 
    y_pred = (y_pred>0.5).astype(int)
    Ntp = np.sum(y_pred + y_true > 1)
    Nref = np.sum(y_true)
    Nsys = np.sum(y_pred)    
    
    S = min(Nref, Nsys) - Ntp
    D = max(0.0, Nref - Nsys)
    I = max(0.0, Nsys - Nref)    

    ER = (S+D+I)/float(Nref + eps)
    
    return(ER)

class MetricsCallback(Callback):
    """ Callback to calculate metrics on epochs end"""
    def __init__(self, x_val, y_val, f1s_current, f1s_best, file_weights):
        """ Init
        Parameters
        ----------
        x_val : array
            Validation input data

        y_val : array
            Validation ground-truth
            
        f1s_current : float
            Last F1 value before training (0 if start)
            
        f1s_current : float
            Last best F1 value before training  (0 if start)   
            
        file_weights : string
            Path to the file weights

        """ 
        
        self.x_val = x_val
        self.y_val = y_val
        self.f1s_current = f1s_current
        self.f1s_best = f1s_best
        self.file_weights = file_weights
        self.epochs_since_improvement = 0
        self.epoch_best = 0
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
            y_val = self.y_val[0]
        else:
            y_val = self.y_val       
        F = F1(y_val,y_pred)
        E = ER(y_val,y_pred)
        logs['F1'] = F
        logs['ER'] = E

        self.f1s_current = F
        
        #self.model.save_weights('weights_final.hdf5') # Graba SIEMPRE!!

        if self.f1s_current > self.f1s_best:
            self.f1s_best = self.f1s_current
            self.model.save_weights(self.file_weights)
            print('F1 = {:.4f}, ER = {:.4f} -  Best val F1s: {:.4f} (IMPROVEMENT, saving)\n'.format(F, E, self.f1s_best))
            self.epochs_since_improvement = 0
            self.epoch_best = epoch
        else:
            print('F1 = {:.4f}, ER = {:.4f} - Best val F1s: {:.4f} ({:d})\n'.format(F, E, self.f1s_best, self.epoch_best))
            self.epochs_since_improvement += 1

