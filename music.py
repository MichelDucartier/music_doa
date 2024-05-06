import math
import numpy as np
import math
from scipy.signal import find_peaks
import scipy    

SOUND_SPEED = 343
    
def music(samples, n_sources):
    M = len(samples)
    
    samples = (samples.T - samples.mean(axis=1).T).T

    print('Samples shape:', samples.shape)
    covariance = samples @ samples.conj().T / M
    print("Covariance shape:", covariance.shape)
        
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]
    
    return signal_eigenvalues, signal_eigenvectors, noise_eigenvalues, noise_eigenvectors


def compute_stfd(samples, nperseg, normalized_freq_range=[0, 0.5]):
    sources_stft = []
    
    # Iterate over every microphone    
    for samples_channel in samples:
        freq, time_intervals, stft = scipy.signal.stft(samples_channel, nperseg=nperseg, return_onesided=True)
        
        sources_stft.append(stft)
        
    sources_stft = np.array(sources_stft).transpose((1, 2, 0))
    
    print(sources_stft.shape)
    stfd_matrices = np.einsum('ijk,ijl->ijkl', sources_stft, sources_stft.conj())
    print(stfd_matrices.shape)
    
    stfd = stfd_matrices.mean(axis=(0, 1))
    print(stfd.shape)
    return stfd

def music_with_frequency(samples, n_sources, fs, mics_coords, segment_duration=0.5, freq_range=None):
    if freq_range is None:
        freq_range = [0, fs]
    freq_range = np.array(freq_range)
    
    # Number of samples to span the segment duration
    nperseg = segment_duration * fs
        
    samples = (samples.T - samples.mean(axis=1).T).T
    
    stfd = compute_stfd(samples, nperseg)
    
    eigenvalues, eigenvectors = np.linalg.eigh(stfd)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]
    
    print("Shape of noise eigenvectors:", noise_eigenvectors.shape)
    
    return general_spectrum_function(noise_eigenvectors, mics_coords.T, (np.mean(freq_range) / nperseg) * fs)


def general_spectrum_function(noise_eigenvectors, mic_locations, main_frequency):
    wavelength = SOUND_SPEED / main_frequency
    
    M = noise_eigenvectors.shape[0]
    def helper(theta):
        a = np.array([np.cos(theta), np.sin(theta)])
        atheta = np.exp(1j * 2 * np.pi / wavelength * np.dot(mic_locations, a))
        temp = atheta.conj() @ noise_eigenvectors
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper