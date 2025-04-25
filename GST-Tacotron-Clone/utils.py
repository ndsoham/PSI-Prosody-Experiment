import librosa
import matplotlib.pyplot as plt
import numpy as np
from Hyperparameters import Hyperparameters as hp

def extract_spectrograms(filepath):
    """
    Returns normalized log(melspectrogram) and log(magnitude) from filepath
    Params:
        filepath: path to audio file
    Returns:
        mel: A 2d array of shape (T, n_mels)
        mag: A 2d array of shape (T, 1+n_fft/2)
    """
    
    # load the sound file
    y, sr = librosa.load(filepath)
    
    # remove silence
    y, _ = librosa.effects.trim(y)
    
    # preemphasis: used to overcome noise at high frequencies
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    
    # stft
    S = librosa.stft(y=y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)
    
    # magnitude spectrogrma
    mag = np.abs(S)
    
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    mel = np.dot(mel_basis, mag) 
    
    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    
    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    
    # transpose
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)
    
    return mel, mag
