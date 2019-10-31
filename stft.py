import librosa 
import tensorflow as tf 
import scipy 
import scipy.io.wavfile as wavf 
import numpy as np

def stft_tf(wav, win_length, hop_length, n_fft, window='hann', mode='REFLECT'):
    '''
    implement stft in tensorflow
    the output is same as librosa.stft with center=True in 10*-6 error
    link: https://github.com/zhang-wy15/stft_from_librosa_to_tensorflow
    '''
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft
    
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    window = scipy.signal.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    window = np.pad(window,((n_fft - win_length) // 2, (n_fft - win_length) // 2), mode='constant', constant_values=(0, 0))

    # Reshape so that the window can be broadcast
    # We don't need this
    # window = window.reshape((-1,1))

    # Pad the time series so that frames are centered
    center = True
    if center:
        wav = tf.pad(wav, [[n_fft // 2, n_fft // 2]], mode=mode)
    
    # Window the time series.
    f = tf.contrib.signal.frame(wav, n_fft, hop_length, pad_end=False)

    # fft method 1: divide block and caculate fft separately
    # fft method 2: whole frame to tf.spectral.fft
    # result are same, but method 2 is faster

    # method 1:
    '''
    linear = tf.zeros((f.shape[0],int(1 + n_fft // 2)))
    MAX_MEM_BLOCK = 2**8 * 2**10
    itemsieze = 8
    n_columns = int(MAX_MEM_BLOCK / (int(1 + n_fft // 2) * itemsieze))
    for bl_s in range(0, linear.shape[0], n_columns):
        bl_t = min(bl_s + n_columns, linear.shape[0])
        temp = tf.spectral.fft(tf.to_complex64(f[bl_s:bl_t,:] * window))[:,:linear.shape[1]]
        print(temp)
        if not bl_s:
            linear_spect = temp
        else:
            linear_spect = tf.concat([linear_spect, temp],axis=0)
    '''

    # method 2:
    linear = tf.spectral.fft(tf.to_complex64(f * window))[:,:int(1 + n_fft // 2)]

    return linear

if __name__ == "__main__":
    _, wav = wavf.read('./00001.wav')
    wav = np.asarray(wav / 2**15)
    win_length = 400
    hop_length = 160
    n_fft = 512

    linear_tf = stft_tf(wav, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    linear_lb = librosa.core.stft(wav, win_length=win_length, hop_length=hop_length, n_fft=n_fft).T

    with tf.Session() as sess:
        print(sess.run(linear_tf))
        print(linear_lb)
        print(np.mean(np.abs(sess.run(linear_tf) - linear_lb)))