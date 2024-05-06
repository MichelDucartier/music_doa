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


def compute_stfd(samples, nperseg):
    sources_stft = []
    
    # Iterate over every microphone    
    for samples_channel in samples:
        freq, time_intervals, stft = scipy.signal.stft(samples_channel, nperseg=nperseg, return_onesided=True)
        sources_stft.append(stft)
        
    sources_stft = np.array(sources_stft).transpose((1, 2, 0))
    
    print("Shape of sources STFT:", sources_stft.shape)
    stfd_matrices = np.einsum('ijk,ijl->ijkl', sources_stft, sources_stft.conj())
    
    stfd = stfd_matrices.mean(axis=(0, 1))
    print("STFD shape:", stfd.shape)
    
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
    
    def helper(theta):
        a = np.array([np.cos(theta), np.sin(theta)])
        atheta = np.exp(1j * 2 * np.pi / wavelength * np.dot(mic_locations, a))
        temp = atheta.conj() @ noise_eigenvectors
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper


def mic_array_spectrum_function(noise_eigenvectors, mic_distance, wavelength):    
    M = noise_eigenvectors.shape[0]
    def helper(sin_value):
        phi = 2 * np.pi * mic_distance * sin_value / wavelength
        x = [np.exp(-1j*phi)]
        a = np.vander(x, M, increasing=True)
        temp = a.conj() @ noise_eigenvectors
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper


def extract_frequencies(spectrum, n_sources, input_range, resolution=10000):
    X = np.linspace(input_range[0], input_range[1], resolution)
    Y = np.array([spectrum(x) for x in X])
    
    threshold = np.sort(Y)[int(0.8 * resolution)]
    
    peak_indices, _ = find_peaks(Y, threshold)
    indices = np.argsort(Y[peak_indices])[-n_sources:]  # Sort peak indices by y value and take the top num_peaks    
    estimated_freq = peak_indices[indices]
    
    return (estimated_freq / resolution) * (input_range[1] - input_range[0]) + input_range[0]