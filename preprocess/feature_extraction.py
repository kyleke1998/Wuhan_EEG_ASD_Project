import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import ast
import preprocess.cleaning as cleaning
from preprocess.cleaning import light_preprocessing, remove_bad_channels_kevin
from preprocess.cleaning import all_batch_1_edfs
from collections import defaultdict
from tqdm import tqdm
import time
from joblib import Parallel, delayed



BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGE = {'delta': (1,4), 'theta':(4,8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 42)}



start_time = time.time()


feature_df_list = [] # list of dataframes, one for each subject

for subject in tqdm(all_batch_1_edfs[:10]):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 60  # seconds
    overlap = 30 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
    df_features = defaultdict(list)
    
    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
        #good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
        psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i, epochs[i].info['sfreq'], fmin=0.5, fmax=42)
        # psd_i.shape = (n_good_channels, n_frequencies)
        # band power
        df_features['subject'].append(subject)
        df_features['epoch'].append(i)
        for bn in BAND_NAMES:
                band = BAND_RANGE[bn]
                # Find closest indices of band in frequency vector
                idx_band = np.logical_and(freq_i >= band[0], freq_i <= band[1])
                # Average power over frequencies
                power = psd_i[:, idx_band].sum(axis=1)*np.mean(np.diff(freq_i[idx_band])) # power.shape=(n_channels,)
                power_db = 10 * np.log10(power)
                for ch in range(len(cleaned.ch_names)):
                    df_features[f'bp_{bn}_{ch}'].append(power_db[ch])
    df_features = pd.DataFrame(df_features) 
    feature_df_list.append(df_features)



end_time = time.time()
print(end_time - start_time)




###############
# Parallelized #
###############

def extract_features(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 60  # seconds
    overlap = 30 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
    df_features = defaultdict(list)

    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
        #good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
        psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i, epochs[i].info['sfreq'], fmin=0.5, fmax=42)
        # psd_i.shape = (n_good_channels, n_frequencies)
        # band power
        df_features['subject'].append(subject)
        df_features['epoch'].append(i)
        for bn in BAND_NAMES:
            band = BAND_RANGE[bn]
            # Find closest indices of band in frequency vector
            idx_band = np.logical_and(freq_i >= band[0], freq_i <= band[1])
            # Average power over frequencies
            power = psd_i[:, idx_band].sum(axis=1)*np.mean(np.diff(freq_i[idx_band])) # power.shape=(n_channels,)
            power_db = 10 * np.log10(power)
            for ch in range(len(cleaned.ch_names)):
                df_features[f'bp_{bn}_{ch}'].append(power_db[ch])
    return pd.DataFrame(df_features)



start_time = time.time()
feature_df_list = Parallel(n_jobs=-1)(
    delayed(extract_features)(subject) for subject in all_batch_1_edfs[:10]
)

end_time = time.time()
print(end_time - start_time)



















################################
# Sandbox #
################################

### Get features for one epoch
signal_i = epochs.get_data()[0,:,:]
#good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
#psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i[good_channel_ids], epochs[0].info['sfreq'], fmin=0.5, fmax=42)
psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i, epochs[0].info['sfreq'], fmin=0.5, fmax=42)

df_features['epoch'].append(i)
for bn in BAND_NAMES:
        band = BAND_RANGE[bn]
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freq_i >= band[0], freq_i <= band[1])
        # Average power over frequencies
        power = psd_i[:, idx_band].sum(axis=1)*np.mean(np.diff(freq_i[idx_band])) # power.shape=(n_channels,)
        power_db = 10 * np.log10(power)
        for ch in range(len(cleaned.ch_names)):
            df_features[f'bp_{bn}_{ch}'].append(power_db[ch])



df_features = pd.DataFrame(df_features)