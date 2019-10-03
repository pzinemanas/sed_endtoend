import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning

def mel_filters(sr = 22050,n_fft = 1024,N_mels = 128,fmin = 0,fmax = 11025,htk=True,file_name=None, plot=False):
    """ Calculate time-domain MEL filters
    Parameters
    ----------
    sr : int
        Sampling rate

    n_fft : int
        Number of samples used to calculate FFT

    N_mels : int
        Number of mel bands

    fmin : float
        Minimum frequency of mel bands   
        
    fmax : float
        Maximum frequency of mel bands   
        
    htk : bool
        If True uses htk formula          

    file_name : string
        If is not None, save the filters matrix in the path file_name
        
    plot : bool
        If True, plots the filters

    Return
    ----------
    hs : array
        Time-domain mel filters

    """ 
    M = librosa.filters.mel(sr,n_fft,N_mels,htk=htk)#,fmin,fmax,norm=1,htk=False)
    freqs = librosa.mel_frequencies(N_mels+2,fmin,fmax,htk=htk)
    widths = np.diff(freqs)
    freqs = freqs[1:-1]
    #print(freqs.shape)
    
    #print(widths)
    f = np.linspace(0,sr/2,n_fft/2+1)
    
    N = n_fft
    Ts = 1/float(sr)
    T = N*Ts
    t = np.arange(-T/2,T/2,Ts)
    
    plt.figure()
    #hs = np.zeros((4*N_mels,N))
    hs = np.zeros((N_mels,N))
    for j in range(N_mels):
    
        fc = freqs[j]
        f0 = widths[j]
        h = 2*f0*np.sinc(t*f0)**2*np.cos(2*np.pi*fc*t)/sr
        h = h/f0 #normalizo por el width
        h = h*hanning(N)
        h_old = h
        H = np.abs(np.fft.fft(h))#/np.sqrt(N)
        M1 = M/np.amax(M)
        
        h = np.amax(M[j,:])*h/np.amax(H)
        seguir = False
        if (j == 0) | (seguir==True):
            factor = np.amax(M[j,:])/np.amax(H)
            if factor == 0:
                factor = 1
                seguir = True
                
        h = h/factor
    
        H = np.abs(np.fft.fft(h))

        hs[j,:] = h
        
        if plot:
            plt.subplot(2,1,1)
            plt.plot(f,H[:N/2+1])
        #
            plt.subplot(2,1,2)
            plt.plot(f,M[j,:])
            i = np.argmin(np.abs(freqs[j]-f))
            plt.plot(freqs[j],M[j,i],'ro')
    if plot:
        plt.plot(freqs,np.zeros_like(freqs),'ro')
        plt.grid()
        plt.show()
    if file_name is not None:
        np.save(file_name,hs)
    
    return hs