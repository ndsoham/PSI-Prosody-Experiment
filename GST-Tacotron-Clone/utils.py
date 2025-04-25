import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_log_melspectrogram(filepath):
    """
    filepath: path to audio file
    """
    y, sr = librosa.load(filepath)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec)
    plt.figure().set_figwidth(12)
    librosa.display.specshow(log_mel_spec, x_axis="time", y_axis="mel", sr=sr)
    plt.colorbar()
    plt.show()
    return log_mel_spec
    

log_mel_spec = extract_log_melspectrogram("obama-sample.wav")
