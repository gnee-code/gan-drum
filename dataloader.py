####################################################
# Data Loader for the training dataset
# Takes a fraction of files from the various
# sample `types` and returns a randomise batch of
# the data
####################################################

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
from glob import glob
import random
import librosa
import pydub
import scipy.io.wavfile as wavfile
from math import ceil, floor
import soundfile as sf

# All possible sample types
sample_types = {"kick","snare","hat","clap","closed_hat","open_hat","ride","crash","tom","fx","vox"}

DATA_PATH = "SAMPLES/"

# Select the `type` of sample that we're going to generate
main_type = "kick"

# Select total number of input samples
total_samples = 100

# We want to be able to mix in different types of sample to generate some weird and interesting stuff
# so we should define a list of types to mix in, allowing the user to deselect some of them
drop_types = {}
mix_types = sample_types - {main_type} - drop_types

# Define a fraction of subtypes to mix in (we'll mix in a random subset of these, and the fraction is
# with respect to the total number in main_type)
mix_fraction = 0.5

# Collect all of the filenames we'll be using
main_filenames = glob(os.path.join(DATA_PATH, main_type, "*.wav"))
mix_filenames = np.concatenate([glob(os.path.join(DATA_PATH, type, "*.wav")) for type in mix_types]).tolist()

# Find the number of samples from each set
n_main = ceil(total_samples * (1 - mix_fraction))
n_mix = floor(total_samples * mix_fraction)

# Select the training set of filenames
train_filenames = np.concatenate(random.sample(main_filenames, n_main),random.sample(mix_filenames,n_mix)).tolist()

# These should all be .wav files, so can use soundfile to load them. Then use librosa to extract the
# spectrogram. We want all files to be the same length, so we'll pad or trim them to have 2s in length
def load_spectrogram(filename):
    data, sr = sf.read(filename,channels=1)
    if len(data) < 2*sr:
        data = np.pad(data, (0, 2*sr-len(data)), "minimum")
    elif len(data) > 2*sr:
        data = data[:2*sr]
    return librosa.feature.melspectrogram(data, sr=sr, n_fft=1024, hop_length=256, n_mels=128)

for i, filename in enumerate(train_filenames):
    train_filenames[i] = load_spectrogram(filename)


