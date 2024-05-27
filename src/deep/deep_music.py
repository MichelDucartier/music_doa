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


def torch_spectrum_function(noise_eigenvectors, mic_locations, wavelength, n_thetas):
    # wavelength = SOUND_SPEED / main_frequency

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
        diff = (estimated_thetas[[permutation]] - true_thetas + np.pi / 2) % np.pi - np.pi / 2
        current_loss = torch.linalg.norm(diff)


        if current_loss < min_loss:
            min_loss = current_loss

    return (1 / n_sources) * min_loss



class NeuralNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden_size: int):
        super(NeuralNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_outputs),
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
        self.neural_net = NeuralNet(self.conf["n_thetas"], self.n_mics, self.conf["peak_finder_hidden_size"])


    def forward(self, samples, n_sources):
        wavelength = self.conf["wavelength"]

        # Forward pass through the recurrent neural network
        samples = samples.T.type(torch.cfloat)
        samples = torch.view_as_real(samples)
        samples = torch.flatten(samples, start_dim=-2)

        _, output = self.gru(samples)
        output = self.post_gru(output)
        output = torch.reshape(output, (self.n_mics, self.n_mics, 2))
        output = torch.view_as_complex(output)

        # Compute eigendecomposition
        eigenvalues, output = torch.linalg.eig(output)

        # Sort eigenvalues and eigenvectors
        indices = torch.argsort(torch.abs(eigenvalues))
        eigenvalues, output = eigenvalues[indices], output[indices]

        signal_eigenvalues, signal_eigenvectors = eigenvalues[-n_sources :], output[:, -n_sources :]
        noise_eigenvalues, output = eigenvalues[: -n_sources], output[:, : -n_sources]

        output = torch_spectrum_function(output, self.mic_locations, wavelength, self.conf["n_thetas"])

        return self.neural_net(output), output


def predict(model, audio, n_sources):
  audio = torch.tensor(audio).to(device)
  estimated_sources, spectrum = model(audio, n_sources)
  
  return estimated_sources.cpu().detach().numpy()[: n_sources], spectrum.cpu().detach().numpy()