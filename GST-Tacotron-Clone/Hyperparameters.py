
class Hyperparameters():
    # preprocessing
    sr = 16000 
    preemphasis = .97
    n_fft = 1024
    n_mels = 80
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr * frame_shift)
    win_length = int(sr * frame_length)
    ref_db = 20
    max_db = 100
    
    # reference encoder
    E = 256
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    
    # style token layer
    token_num = 10
    num_heads = 8
    
    # tacotron
    dropout_p = 0.5
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"
    K = 16
    num_highways = 4
    r = 5
    max_Ty = max_iter = 200
    decoder_K = 8
    
    
    