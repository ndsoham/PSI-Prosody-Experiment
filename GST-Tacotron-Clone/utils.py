import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_log_melspectrogram(filepath):
    """
    Returns normalized log(melspectrogram) and log(magnitude) from filepath
    Params:
        filepath: path to audio file
    Returns:

    """
    
    # load the sound file
    y, sr = librosa.load(filepath)
    
    # remove silence
    y, _ = librosa.effects.trim(y)
    
    # TODO: preemphasis hyperparam
    
    # TODO: stft with n_fft, hop_length, and win_length hyperparams
    
    
    

log_mel_spec = extract_log_melspectrogram("obama-sample.wav")
