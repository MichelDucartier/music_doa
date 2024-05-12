import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np
from scipy.signal import stft
import itertools
from torch.autograd import Variable

import sys
sys.append("src/")

from music import SOUND_SPEED


def torch_spectrum_function(noise_eigenvectors, mic_locations, main_frequency, n_thetas):
    wavelength = SOUND_SPEED / main_frequency

    a = torch.tensor([torch.tensor([np.cos(theta), np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, n_thetas)])
    atheta = torch.exp(-2j * np.pi / wavelength * torch.dot(mic_locations, a))

    temp = torch.dot(atheta.conj(), noise_eigenvectors)

    return 1 / torch.linalg.norm(temp)


def rmspe_loss(estimated_thetas, true_thetas, n_sources):
    estimated_thetas = estimated_thetas[: n_sources]
    true_thetas = true_thetas[: n_sources]

    min_loss = float('inf')

    # For each permutation, we take 
    for permutation in itertools.permutations(range(n_sources)):
        current_loss = torch.linalg.norm(estimated_thetas[permutation] - true_thetas) ** 2

        if current_loss < min_loss:
            min_loss = current_loss

    return (1 / len(n_sources)) * torch.sqrt(min_loss)


class NeuralNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y
    

class DeepMUSIC(nn.Module):
    def __init__(self, mic_locations: List | np.ndarray, conf: Dict[str, Any]):
        super(DeepMUSIC, self).__init__()

        self.n_mics = len(mic_locations)
        self.mic_locations = mic_locations
        self.conf = conf

        n_freq = (conf["npperseg"] // 2) + 1 
        self.gru = nn.GRU(input_size=self.n_mics * n_freq,
                          hidden_size=self.conf["gru_hidden_size"])
        
        self.post_gru = nn.Linear(in_features=self.conf["gru_hidden_size"], 
                                  out_features=self.n_mics)
    
        # Input size is the resolution of our spectrum function
        # Output can't be larger than the number of microphones
        self.neural_net = NeuralNet(self.conf["n_thetas"], self.n_mics)


    def forward(self, samples, n_sources):
        ## Normalize samples + fft
        samples = ((samples.T - np.mean(samples, axis=1).T) / np.std(samples, axis=1).T).T
        fft = torch.fft.fft(samples)
        
        main_frequency = torch.argmax(fft[: len(fft) // 2]) * self.conf["fs"] / len(fft)

        # Forward pass through the recurrent neural network
        output = self.gru(samples)
        output = self.post_gru(output)

        # Compute outer product
        covariance = torch.outer(output, torch.conj(output))

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.symeig(covariance, eigenvectors=True)

        # Sort eigenvalues and eigenvectors
        indices = torch.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[indices]

        signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
        noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]

        spectrum_values = torch_spectrum_function(noise_eigenvectors, self.mic_locations, main_frequency, self.conf["n_thetas"])

        return self.neural_net(spectrum_values)
        
