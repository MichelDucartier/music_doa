import math
import numpy as np
import math
from scipy.signal import find_peaks
import scipy    

SOUND_SPEED = 343
    
def music(samples, n_sources, mic_coords, main_frequency, correlated=False):
    M = len(samples)
    
    samples = (samples.T - samples.mean(axis=1).T).T

    print('Samples shape:', samples.shape)
    covariance = samples @ samples.conj().T / M
    print("Covariance shape:", covariance.shape)
    
    if correlated:
        J = np.flip(np.eye(mic_coords.shape[1]), axis=1)
        covariance = covariance + (J @ covariance.conj() @ J)


    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]
    
    return general_spectrum_function(noise_eigenvectors, mic_coords.T, main_frequency)


def compute_stfd(samples, nperseg, normalized_freq_range):
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
        # stfd_flipped = compute_stfd(J @ samples.conj(), nperseg, normalized_freq_range)
        stfd = (stfd + (J @ stfd.conj() @ J)) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(stfd)
    
    # Decompose signal and noise
    # Assumption : the noise has the smallest eigenvalues
    signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
    noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]
    
    print("Shape of noise eigenvectors:", noise_eigenvectors.shape)
    
    # Assume main frequency is the middle of the frequency range
    main_frequency = (np.mean(freq_range) / nperseg) * fs

    return general_spectrum_function(noise_eigenvectors, mics_coords.T, main_frequency)


def general_spectrum_function(noise_eigenvectors, mic_locations, main_frequency):
    wavelength = SOUND_SPEED / main_frequency
    
    def helper(theta):
        a = np.array([np.cos(theta), np.sin(theta)])
        atheta = np.exp(-2j * np.pi / wavelength * np.dot(mic_locations, a))
        temp = atheta.conj() @ noise_eigenvectors
        
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


@DeprecationWarning
def mic_array_spectrum_function(noise_eigenvectors, mic_distance, wavelength=1):
    M = noise_eigenvectors.shape[0]
    def helper(sin_value):
        phi = 2 * np.pi * mic_distance * sin_value / wavelength
        x = [np.exp(-1j*phi)]
        a = np.vander(x, M, increasing=True)
        temp = a.conj() @ noise_eigenvectors
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper