# MUSIC for Direction of Arrival

## Setup

Clone this repo with the submodules:
```
git clone --recurse-submodules https://github.com/MichelDucartier/music_doa.git
```

Make sure ton install the requirements:
```
pip install requirements.txt
```

## Project structure

```
├───checkpoints
├───conf
├───DA-MUSIC
│   └───model
├───data
├───Figures
├───img
├───last_year
├───models
├───res
│   └───examples
└───src
    └───deep
```

`src` stores the source code for the different notebooks.

You can find `last_year` work in [this folder](last_year)

### Demo notebooks

You can find 4 different notebooks in the root of this repository:

- `MUSIC_demo.ipynb`: Compare the non deep augmented MUSIC, namely standard MUSIC, Hilbert-based MUSIC and STFT-based MUSIC
- `MUSIC_plots.ipynb`: The notebook to generate the plots for the PDF report
- `MUSIC_ML_training.ipynb`: The notebook to train the Deep Augmented model, this notebook is initially meant to be run with Colab (it will clone this repository by itself), but it can also be used for other setups
- `MUSIC_ML_inference.ipynb`: This notebook can be used as a demo for the Deep Augmented model inference and benchmark


### Instructions for Deep Augmented MUSIC

If you wish to continue the training done. You can find our latest checkpoint [here](checkpoints/checkpoint-13-0.pth). In [the training notebook](MUSIC_ML_training.ipynb), you can specify this checkpoint to continue the training.

If you wish to do inference with our [latest model](models/finetuned_model.pt), you must specify the path to this trained model in [the testing notebook](MUSIC_ML_inference.ipynb).

Note that both of those notebooks will create a folder [data](data/) and store the training set/testing set here.


## Authors

- Sophie Strebel
- Michael Zhang

for the COM-500 Statistical Signal and Data Processing through applications at EPFL. 
