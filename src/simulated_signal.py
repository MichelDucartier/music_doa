import numpy as np

def narrowband_signal(k, w0):    
    def helper(t):
        # Use a hash as a seed to ensure that if (k1, t1) = (k2, t2), then we give the exact result
        rng = np.random.RandomState(hash((k, t)) & 0xffffffff)
        return rng.randn() * np.exp(-1j * w0 * t)
    return helper

def received_signal(thetas, narrowband_signals, mic_index, mic_distance=0.2, noise_var=0, wavelength=1):
    def helper(t):
        signal = 0
        for theta, narrowband in zip(thetas, narrowband_signals):
            phase = np.exp(-2j * np.pi * (mic_index - 1) * mic_distance * np.sin(theta) / wavelength)
            signal += narrowband(t) * phase
            
        return signal + np.random.normal(size=2, scale=noise_var).view(np.complex128)[0]
    
    return helper