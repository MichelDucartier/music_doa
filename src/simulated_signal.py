import numpy as np
from music import SOUND_SPEED

def narrowband_signal(k, w0):
    """
    Creates and returns a function (representing a signal) to be evaluated at a time step t with parameters k and w0

    Parameters:
    k : index of the source sending the signal
    w0 : main frequency

    Returns:
    A deterministic function of k and t, to be evaluated at some t
    """
    def helper(t):
        # Use a hash as a seed to ensure that if (k1, t1) = (k2, t2), then we give the exact result
        rng = np.random.RandomState(hash((k, t)) & 0xffffffff)
        return rng.randn() * np.exp(-1j * w0 * t)
    return helper


def received_signal(thetas, narrowband_signals, mic_index, main_frequency, mic_distance=0.2, noise_var=0):
    """
    Creates and returns a function (representing a signal) to be evaluated at a time step t representing the received signal
    of the microphone at the given index assuming the coordinates of the m'th microphone are (0, mic_distance * (m-1)),
    sum of all sources and additive white Gaussian noise

    Parameters:
    thetas : angles of the sources
    narrowband_signals : signals sent by the sources
    mic_index : index of the microphone
    main_frequency : main frequency
    mic_distance : distance between microphones assuming they are aligned
    noise_var : variance/power of the additive white Gaussian noise

    Returns:
    A function to be evaluated at time step t representing the received signal of the mic_index'th microphone at that time
    """
    wavelength = SOUND_SPEED / main_frequency
    
    def helper(t):
        signal = 0
        for theta, narrowband in zip(thetas, narrowband_signals):
            phase = np.exp(-2j * np.pi * (mic_index - 1) * mic_distance * np.sin(theta) / wavelength)
            signal += narrowband(t) * phase
            
        return signal + np.random.normal(size=2, scale=noise_var).view(np.complex128)[0]
    
    return helper