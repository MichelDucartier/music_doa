import os
from scipy.io import wavfile
import numpy as np
import random
import json
import pyroomacoustics as pra
from tqdm import tqdm
from pathlib import Path
import datasets
from datasets import Dataset, Audio



EXAMPLES_DIRECTORY = "res/examples/dcase2016_task2_train/"
OUT_DIRECTORY = "res/train/"
SAMPLING_FREQ = 4410
NOISE_VAR = 0.1
ROOM_DIM = np.array([10, 10])


def load_examples(downsample_ratio=10):
    examples = list()

    for file in os.listdir(EXAMPLES_DIRECTORY):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"): 
            wav_name = os.path.join(EXAMPLES_DIRECTORY, filename)
            fs, data = wavfile.read(wav_name)
            data = data[::downsample_ratio]

            assert fs // downsample_ratio == SAMPLING_FREQ

            examples.append(data / np.std(data))

    return examples


def concatenate_samples(examples, n_concatenations=3):
    concatenated = np.empty(0)
    
    for _ in range(n_concatenations):
        sample = random.choice(examples)
        concatenated = np.concatenate((concatenated, sample))

    return concatenated


def load_microphones():
    with open('../music_doa/last_year/protocol.json') as json_file:  
        protocol = json.load(json_file)
        
    microphone_3D_locations = np.array(protocol['geometry']['microphones']['locations'])
    top_mics = np.isclose(microphone_3D_locations[:,2], 0.06123724)
    return microphone_3D_locations[top_mics, :2]



def dataset_generator(examples, max_sources=4, n_samples=1000):

    def generator():
        # Put microphones at the center of the room
        mics_coords = (load_microphones() + ROOM_DIM / 2)

        assert max_sources <= len(mics_coords)

        X = []
        Y = []
        for _ in tqdm(range(n_samples)):
            aroom = pra.ShoeBox(ROOM_DIM, fs=SAMPLING_FREQ, max_order=0, sigma2_awgn=NOISE_VAR)
            aroom.add_microphone_array(pra.MicrophoneArray(mics_coords.T, aroom.fs))

            n_sources = random.randrange(1, max_sources)
            doas = np.random.uniform(0, 360, size=n_sources)

            for doa in doas:
                # Generate a distance at random, not too close to microphones
                distance = np.random.uniform(ROOM_DIM / 4, ROOM_DIM / 2)
                source_location = ROOM_DIM / 2 + distance * np.r_[np.cos(doa), np.sin(doa)]
                
                data = concatenate_samples(examples)
                aroom.add_source(source_location, signal=data)

            aroom.simulate()

            yield {"audio" : aroom.mic_array.signals, "n_sources" : n_sources, "doas" : doas}    

    return generator

if __name__ == "__main__":
    Path(OUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    examples = load_examples()
    n_splits = 5

    for split in range(n_splits):
        ds = Dataset.from_generator(dataset_generator(examples, n_samples=100))
        ds.save_to_disk(os.path.join(OUT_DIRECTORY, f"split{split}", "dataset"))
