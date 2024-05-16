import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np
from scipy.signal import stft
import itertools
from torch.autograd import Variable

import sys
sys.path.append("src/")

from music import SOUND_SPEED

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def torch_spectrum_function(noise_eigenvectors, mic_locations, main_frequency, n_thetas):
    wavelength = SOUND_SPEED / main_frequency

    a = torch.tensor([[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, n_thetas)]).to(device)
    atheta = torch.exp(-2j * np.pi / wavelength * torch.matmul(mic_locations, a.T).type(torch.cfloat))

    temp = torch.matmul(atheta.conj().type(torch.cfloat).T, noise_eigenvectors.type(torch.cfloat))
    spectrum = 1 / torch.linalg.norm(temp, dim=1)

    return spectrum

def rmspe_loss(estimated_thetas, true_thetas, n_sources):
    estimated_thetas = estimated_thetas[: n_sources]
    true_thetas = true_thetas[: n_sources]

    min_loss = float('inf')

    # For each permutation, we take 
    for permutation in itertools.permutations(range(n_sources)):
        # current_loss = torch.linalg.norm(torch.cos(estimated_thetas[[permutation]]) - torch.cos(true_thetas)) \
        #     + torch.linalg.norm(torch.sin(estimated_thetas[[permutation]]) - torch.sin(true_thetas))
        diff = (estimated_thetas[[permutation]] - true_thetas + np.pi / 2) % np.pi - np.pi / 2
        current_loss = torch.linalg.norm(diff)


        if current_loss < min_loss:
            min_loss = current_loss

    return (1 / n_sources) * min_loss


def spectrum_loss(estimated_spectrum, true_thetas):
    peak_indices = torch.round((true_thetas / (2 * np.pi)) * len(estimated_spectrum))

    ideal_spectrum = torch.zeros(size=estimated_spectrum.shape)

    window_len = int(0.01 * len(estimated_spectrum))
    max_value = torch.max(estimated_spectrum)
    for peak_index in peak_indices:
        ideal_spectrum[range(peak_index - window_len // 2, peak_index + window_len // 2)] = torch.signal.windows.gaussian(window_len) * max_value

    return (1 / len(estimated_spectrum)) * torch.dot(ideal_spectrum, estimated_spectrum)


class NeuralNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super(NeuralNet, self).__init__()
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

        # n_freq = (conf["npperseg"] // 2) + 1 
        self.gru = nn.GRU(input_size=self.n_mics * 2,
                          hidden_size=self.conf["gru_hidden_size"])
        
        self.post_gru = nn.Linear(self.conf["gru_hidden_size"], self.n_mics * self.n_mics * 2)
    
        # Input size is the resolution of our spectrum function
        # Output can't be larger than the number of microphones
        self.neural_net = NeuralNet(self.conf["n_thetas"], self.n_mics)


    def forward(self, samples, n_sources):
        ## Normalize samples + fft
        samples = (samples - torch.mean(samples, dim=1, keepdim=True)) / torch.std(samples, dim=1, keepdim=True)
        fft = torch.fft.fft(samples[0])

        main_frequency = torch.argmax(abs(fft[: len(fft) // 2])) * self.conf["fs"] / len(fft)

        # Forward pass through the recurrent neural network
        samples = samples.T.type(torch.cfloat)
        samples = torch.view_as_real(samples)
        samples = torch.flatten(samples, start_dim=1)

        _, output = self.gru(samples)
        output = self.post_gru(output)
        output = torch.reshape(output, (self.n_mics, self.n_mics, 2))
        covariance = torch.view_as_complex(output)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)

        # Sort eigenvalues and eigenvectors
        indices = torch.argsort(torch.abs(eigenvalues))
        eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[indices]

        signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], eigenvectors[:, -n_sources :]
        noise_eigenvalues, noise_eigenvectors = eigenvalues[: -n_sources], eigenvectors[:, : -n_sources]

        spectrum_values = torch_spectrum_function(noise_eigenvectors, self.mic_locations, main_frequency, self.conf["n_thetas"])

        return self.neural_net(spectrum_values.clone().detach()), spectrum_values


class DeepSourcesClassifier(nn.Module):
    def __init__(self, mic_locations: List | np.ndarray, conf: Dict[str, Any]):
        super(DeepSourcesClassifier, self).__init__()

        self.n_mics = len(mic_locations)
        self.mic_locations = mic_locations
        self.conf = conf

        self.gru = nn.GRU(input_size=self.n_mics * 2,
                          hidden_size=self.conf["gru_hidden_size"])
        
        self.classifier = nn.Sequential(
            nn.Linear(self.conf["gru_hidden_size"] * 3, 256),
            nn.GELU(),
            nn.Linear(256, self.conf["max_sources"]),
            nn.Softmax()
        )


    def forward(self, samples):
        # Forward pass through the recurrent neural network
        samples = samples.T.type(torch.cfloat)
        samples = torch.view_as_real(samples)
        samples = torch.flatten(samples, start_dim=1)
        
        hidden_layers, _ = self.gru(samples)
        first_hidden = hidden_layers[0]
        middle_hidden = hidden_layers[len(hidden_layers) // 2]
        last_hidden = hidden_layers[-1]

        output = torch.concatenate((first_hidden, middle_hidden, last_hidden))
        output = self.classifier(output)

        return output

