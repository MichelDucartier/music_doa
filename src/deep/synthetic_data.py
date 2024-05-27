####################################################################################################
#                                          syntheticEx.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 10/03/21                                                                                #
#                                                                                                  #
# Purpose: Synthetic examples to test correctness and performance of algorithms estimating         #
#          directions of arrival (DoA) of multiple signals.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path


import random

# set random seed
# np.random.seed(42)


#********************#
#   initialization   #
#********************#

d = 2   # number of sources
snr = 10   # signal to noise ratio

mean_signal_power = 0
var_signal_power = 1

mean_noise = 0
var_noise = 1

doa = np.pi * (np.random.rand(d) - 1/2)   # random source directions in [-pi/2, pi/2]
p = np.sqrt(1) * (np.random.randn(d) + np.random.randn(d) * 1j)    # random source powers


def load_microphones():
    with open('../music_doa/last_year/protocol.json') as json_file:  
        protocol = json.load(json_file)
        
    microphone_3D_locations = np.array(protocol['geometry']['microphones']['locations'])
    top_mics = np.isclose(microphone_3D_locations[:,2], 0.06123724)
    microphone_2D_locations = microphone_3D_locations[top_mics, :2]

    return microphone_2D_locations


mics_coords = load_microphones()
m = len(mics_coords)   # number of array elements

angles = np.array((np.linspace(- np.pi/2, np.pi/2, 360, endpoint=False),))   # angle continuum

snapshots = 200    


def ULA_action_vector(mics_coords, theta):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.

        @returns -- The action vector.
    """

    rotation_vec = np.array([np.cos(theta), np.sin(theta)])
    return np.exp(- 1j * np.pi * mics_coords @ rotation_vec)


#***********************#
#   construct signals   #
#***********************#
def construct_signal(thetas):
    """
        Construct a signal with the given initializations.

        @param thetas -- The doa angles of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)

    signal = np.sqrt(var_signal_power) * (10 ** (snr / 10)) * \
             (np.random.randn(d, snapshots) + 1j * np.random.randn(d, snapshots)) + mean_signal_power
    A = np.array([ULA_action_vector(mics_coords, thetas[j]) for j in range(d)])
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j *
                                  np.random.randn(m, snapshots)) + mean_noise

    return np.dot(A.T, signal) + noise, signal


#********************************#
#   construct coherent signals   #
#********************************#
def construct_coherent_signal(thetas):
    """
        Construct a coherent signal with the given initializations.

        @param thetas -- The doa angles of the sources.

        @returns -- The measurement vector.
    """
    signal = np.sqrt(var_signal_power) * (10 ** (snr / 10)) * \
             (np.random.randn(1, snapshots) + 1j * np.random.randn(1, snapshots)) + mean_signal_power

    # all signals receive same amplitude and phase...
    signal = np.repeat(signal, len(thetas), axis=0)

    A = np.array([ULA_action_vector(mics_coords, thetas[j]) for j in range(len(thetas))])
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j *
                                  np.random.randn(m, snapshots)) + mean_noise

    return np.dot(A.T, signal) + noise, signal


#********************#
#   create dataset   #
#********************#
def create_dataset(name, size, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, d))
    n_sources_list = np.zeros((size, 1))

    for i in tqdm(range(size)):
        
        thetas = np.pi * (np.random.rand(d) - 1/2)  # random source directions
        
        # Random number of sources
        n_sources = d
        n_sources_list[i] = n_sources
        
        thetas[n_sources :] = np.nan

        if coherent: X[i] = construct_coherent_signal(thetas[: n_sources])[0]
        else: X[i] = construct_signal(thetas[: n_sources])[0]
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.create_dataset('n_sources', data=n_sources_list)
        hf.close()

    return X, Thetas, n_sources_list


#**************************#
#   create mixed dataset   #
#**************************#
def create_mixed_dataset(name, first, second, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param first -- The path/name of the first dataset to be mixed with...
        @param second -- The path/name of the second dataset.
        @param save -- If true the dataset is saved to filename.
    """
    hf1 = h5py.File(first + '.h5', 'r')
    hf2 = h5py.File(second + '.h5', 'r')

    assert hf1.keys() == hf2.keys()

    data_list = dict()

    permutations = np.random.permutation(range(len(hf1) + len(hf2)))

    for key in hf1.keys():
        data_value1 = np.array(hf1.get(key))
        data_value2 = np.array(hf2.get(key))
        data_list[key]  = np.stack((data_value1, data_value2))[permutations]

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        for key, data in data_list.items():
            hf.create_dataset(key, data=data)

        hf.close()

    return data_list


#*******************************#
#   create resolution dataset   #
#*******************************#
def create_res_cap_dataset(name, size, space, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves
        used for testing the resolution capabilities of DoA estimation algorithms
        by creating two closely spaced signals.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param space -- The distance of the closely spaced signals.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, 2))
    for i in tqdm(range(size)):
        theta = np.pi * (np.random.rand(1) - 1/2)   # random source direction
        thetas = [theta, (((theta + space) + np.pi/2) % np.pi) - np.pi/2]
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas




if __name__ == "__main__":
    # create_mixed_dataset('m_d2_l200_snr10_2k', first='data/d2_l200_snr10_1k',
    #                                            second='data/c_d2_l200_snr10_1k')

    # create_res_cap_dataset('m8/res0.20_l200_snr10_10k', 10000, 0.20)
    Path("data/").mkdir(parents=True, exist_ok=True)
    create_dataset('coherent_dataset', 100000, coherent=True)
    # create_dataset('non_coherent_dataset', 1000, coherent=False)
    # create_mixed_dataset("mixed_dataset", "data/coherent_dataset", "data/non_coherent_dataset")