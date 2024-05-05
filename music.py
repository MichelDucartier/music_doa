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


def covariance_in_frequency(samples, nperseg):
    spectrums = []
    
    # Iterate over every microphone    
    for samples_channel in samples:
        freq, time_intervals, stft = scipy.signal.stft(samples_channel, nperseg=nperseg, return_onesided=True)
                
        # Average over every time interval
        power_in_freq = stft.mean(axis=1)
        
        spectrums.append(power_in_freq)
    
    spectrums = np.array(spectrums).T # Matrix of size number_of_freq * number_of_mics
    
    # ChatGPT stuff, hmmmm
    print(spectrums.shape)
    covariance_matrices = np.einsum('ij,ik->ijk', spectrums, spectrums.conj())
    print("Covariance matrices shape:", covariance_matrices.shape)
    
    return covariance_matrices


def music_with_frequency(samples, n_sources, fs, mics_coords, segment_duration=0.5, freq_range=None):
    if freq_range is None:
        freq_range = [0, fs]
    freq_range = np.array(freq_range)
    
    # Number of samples to span the segment duration
    nperseg = segment_duration * fs
        
    samples = (samples.T - samples.mean(axis=1).T).T
    
    covariance_matrices = covariance_in_frequency(samples, nperseg)
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[:, -n_sources :], eigenvectors[:, :, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[:, : -n_sources], eigenvectors[:, :, : -n_sources]
    
    
    selected_freq_ids = (freq_range * nperseg) / (2 * fs)
    print(selected_freq_ids)
    spectrums = []
    for freq_id in range(int(selected_freq_ids[0]), int(selected_freq_ids[1])):
        spectrums.append(
            general_spectrum_function(noise_eigenvectors[freq_id], mics_coords.T, (freq_id / nperseg) * fs)
        )
    
    return spectrums


def general_spectrum_function(noise_eigenvectors, mic_locations, main_frequency):
    wavelength = SOUND_SPEED / (main_frequency + 1e-5)
    
    M = noise_eigenvectors.shape[0]
    def helper(theta):
        a = np.array([np.cos(theta), np.sin(theta)])
        atheta = np.exp(-1j * 2 * np.pi / wavelength * np.dot(mic_locations, a))
        temp = atheta.conj() @ noise_eigenvectors
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper