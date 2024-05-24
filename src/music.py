import math
import numpy as np
import math
from scipy.signal import find_peaks
import scipy    

SOUND_SPEED = 343
    
def music(samples, n_sources, mic_coords, main_frequency, correlated=False):
    """
    Computes and returns the estimated spatial spectrum by computing the eigendecomposition of the covariance matrix
    of the samples

    Parameters:
    samples : matrix of shape (number of microphones) x (number of times sampled) holding all the information about
              the received signals of each microphone
    n_sources : number of sources
    mic_coords : matrix of shape 2 x (number of microphones) containing coordinates of each microphone
    main_frequency: main frequency
    correlated : whether or not the signals sent by the sources are correlated

    Returns:
    The estimated spatial spectrum
    """
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


def general_spectrum_function(noise_eigenvectors, mic_locations, main_frequency):
    """
    Computes and returns the a function representing the estimated spatial spectrum to be evaluated at values of theta
    using the noise eigenvectors of the covariance matrix of the samples

    Parameters:
    noise_eigenvectors : noise eigenvectors in columns of a matrix
    mic_locations : coordinates of the microphones
    main_frequency : main frequency
    
    Returns:
    The estimated spatial spectrum as a function of theta
    """
    wavelength = SOUND_SPEED / main_frequency
    
    def helper(theta):
        a = np.array([np.cos(theta), np.sin(theta)])
        atheta = np.exp(-2j * np.pi / wavelength * np.dot(mic_locations, a))
        temp = atheta.conj() @ noise_eigenvectors #noise_eigenvectors.T.conj() @ atheta also works
        
        return 1 / np.linalg.norm(temp)**2
    
    return helper


def extract_frequencies(spectrum, n_sources, input_range, resolution=10000, return_peak_indices=False):
    """
    Extracts the frequencies at which the spectrum function has its peaks

    Parameters:
    spectrum : the spatial spectrum, function of theta
    n_sources: number of sources
    input_range : input range of theta
    resolution : number of points to evaluate the spectrum function
    
    Returns:
    Frequencies at which the spectrum function peaks
    """
    X = np.linspace(input_range[0], input_range[1], resolution)
    Y = np.array([spectrum(x) for x in X])
    
    threshold = np.sort(Y)[int(0.8 * resolution)]
    
    peak_indices, _ = find_peaks(Y, threshold)
    indices = np.argsort(Y[peak_indices])[-n_sources:]  # Sort peak indices by y value and take the top num_peaks    
    estimated_freq = peak_indices[indices]
    
    frequencies = (estimated_freq / resolution) * (input_range[1] - input_range[0]) + input_range[0]
    
    if return_peak_indices:
        return frequencies, estimated_freq
        
    return frequencies