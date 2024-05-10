import math
import numpy as np
import math
from scipy.signal import find_peaks
import scipy
from music import general_spectrum_function

SOUND_SPEED = 343
    
def compute_stfd(samples, nperseg, normalized_freq_range):
    """TODO NOT SURE
    Computes and returns the STFT of the given samples in a certain frequency range

    Parameters:
    samples : matrix of shape (number of microphones) x (number of times sampled) holding all the information about
              the received signals of each microphone
    nperseg : length of each segment
    normalized_freq_range : normalized frequency range

    Returns:
    The STFT of the samples
    """
    sources_stft = []
    
    # Iterate over every microphone    
    for samples_channel in samples:
        freq, time_intervals, stft = scipy.signal.stft(samples_channel, nperseg=nperseg, return_onesided=True)
        
        low_freq_mask = (normalized_freq_range[0] <= freq) & (freq <= normalized_freq_range[1])        
        selected_freq_id = np.argwhere(low_freq_mask).flatten()
        stft = stft[selected_freq_id, :]
        
        sources_stft.append(stft)
        

    sources_stft = np.array(sources_stft).transpose((1, 2, 0))
    
    print("Shape of sources STFT:", sources_stft.shape)
    stfd_matrices = np.einsum('ijk,ijl->ijkl', sources_stft.conj(), sources_stft)
    
    stfd = stfd_matrices.mean(axis=(0, 1))
    print("STFD shape:", stfd.shape)
    
    return stfd

def music_with_frequency(samples, n_sources, fs, mics_coords, segment_duration=None, freq_range=None,
                         correlated=False):
    """todo
    Computes and returns the estimated spatial spectrum by computing the eigendecomposition of the covariance matrix
    of the samples

    Parameters:
    samples : matrix of shape (number of microphones) x (number of times sampled) holding all the information about
              the received signals of each microphone
    n_sources : number of sources
    fs : sampling frequency
    mic_coords : matrix of shape 2 x (number of microphones) x with the coordinates of each microphone
    segment_duration: duration in time of each segment
    freq_range : range of frequency to consider for the STFT (??? todo)
    correlated : whether or not the signals sent by the sources are correlated

    Returns:
    The estimated spatial spectrum
    """
    if freq_range is None:
        freq_range = [0, fs]

    if segment_duration is None:
        nperseg = len(samples)
    else:
        # Number of samples to span the segment duration
        nperseg = segment_duration * fs

    freq_range = np.array(freq_range)
    normalized_freq_range = freq_range / fs
    
    
    samples = (samples.T - samples.mean(axis=1).T).T
    stfd = compute_stfd(samples, nperseg, normalized_freq_range)

    if correlated:
        J = np.flip(np.eye(mics_coords.shape[1]), axis=1)
        stfd_flipped = compute_stfd(J @ samples.conj(), nperseg, normalized_freq_range)
        stfd = (stfd + stfd_flipped) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(stfd)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]
    
    print("Shape of noise eigenvectors:", noise_eigenvectors.shape)
    
    # Assume main frequency is the middle of the frequency range
    main_frequency = (np.mean(freq_range) / nperseg) * fs

    return general_spectrum_function(noise_eigenvectors, mics_coords.T, main_frequency)