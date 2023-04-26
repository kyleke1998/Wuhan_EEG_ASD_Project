import mne
import pandas as pd
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import ast
import cleaning
from cleaning import light_preprocessing, remove_bad_channels_kevin, metadata
from cleaning import all_batch_1_edfs, subjects_to_remove
from collections import defaultdict
from tqdm import tqdm
import time
from joblib import Parallel, delayed
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity
from mne.datasets import sample
import itertools
import logging
from mne_features.univariate import compute_samp_entropy, compute_mean, compute_std, compute_kurtosis, compute_skewness
import philistine



BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGE = {'delta': (1,4), 'theta':(4,8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 42)}
CHANNEL_NAMES ={0: 'Fp1',1: 'F3',2: 'C3',3: 'P3',4: 'O1',5: 'F7',6: 'T3',7: 'T5',8: 'Fz',9: 'Fp2',10: 'F4',11: 'C4',12: 'P4',13: 'O2',14: 'F8',15: 'T4',16: 'T6',17: 'Cz',18: 'Pz'}
CHANNEL_NAMES_LIST = list(CHANNEL_NAMES.values())

subjects_to_remove = ['002.edf',
 '039.edf',
 '052.edf',
 '069.edf',
 '159.edf',
 '390.edf',
 '1064.edf',
 '7060.edf',
 '7089.edf',
 '7104.edf']

# the subjects that are not dropped 
good_subjects = list(set(all_batch_1_edfs) - set(subjects_to_remove))







###############
# Parallelized  extract relative power
###############

def extract_relative_power(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    reject = dict(
              eeg=1000e-6,      # unit: V (EEG channels)
              )

    flat = dict(eeg = 0.1e-6)

    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
   
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    if epochs.get_data().shape[0] < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
        #good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
        psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i, epochs[i].info['sfreq'], fmin=0.5, fmax=42)
        psd_i /= np.sum(psd_i, axis=-1, keepdims=True)
        # psd_i.shape = (n_good_channels, n_frequencies)
        # band power
        df_features['subject'].append(subject)
        df_features['epoch'].append(i)
        for bn in BAND_NAMES:
            band = BAND_RANGE[bn]
            # Find closest indices of band in frequency vector
            idx_band = np.logical_and(freq_i >= band[0], freq_i <= band[1])
            # Average power over frequencies
            power = psd_i[:, idx_band].sum(axis=1) #np.mean(np.diff(freq_i[idx_band])) # power.shape=(n_channels,)
            for ch in range(len(cleaned.ch_names)):
                df_features[f'bp_{bn}_{CHANNEL_NAMES[ch]}'].append(power[ch])
    df_features = pd.DataFrame(df_features)
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
    result['subject'] = subject
    return result



start_time = time.time()
feature_df_list = Parallel(n_jobs=-1)(
    delayed(extract_relative_power)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

all_features = pd.concat(feature_df_list, axis=0)

all_features.reset_index().to_feather('all_power.feather')




###############
# Parallelized  extract spectral coherence
###############

def extract_spectral_coh(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    reject = dict(
              eeg=1000e-6,      # unit: V (EEG channels)
              )

    flat = dict(eeg = 0.1e-6)

    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
   
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    if epochs.get_data().shape[0] < 1:
        return None

    all_features = []
    for bn in BAND_NAMES:
        band = BAND_RANGE[bn]
        sfreq = epochs.info['sfreq']  # the sampling frequency
        tmin = 0.0  # exclude the baseline period
        con = spectral_connectivity_epochs(
            epochs, method='coh', mode='multitaper', sfreq=sfreq, fmin=band[0], fmax=band[1],
            faverage=True, mt_adaptive=False)

        coherence_matrix = pd.DataFrame(con.get_data(output='dense')[:,:,0], columns=epochs.ch_names, index=epochs.ch_names)
        coherence_matrix = coherence_matrix.replace(0, np.nan)
        series = coherence_matrix .stack(dropna=True)


        col_names = [f'{i}_{j}' for i, j in series.index]

        flattened_features = pd.DataFrame([series.values], columns=col_names)

        flattened_features = flattened_features.add_prefix('{}_coh_'.format(bn))
        all_features.append(flattened_features)
    

    result = pd.concat(all_features, axis=1)
    result['subject'] = subject
    return result


start_time = time.time()
coh_df_list = Parallel(n_jobs=-1)(
    delayed(extract_spectral_coh)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

all_coh_df_list = pd.concat(coh_df_list , axis=0)

all_coh_df_list.reset_index().to_feather('all_spectral_coh.feather')




###############
# Parallelized  extract sample entropy
###############


def extract_sample_entropy(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    reject = dict(
              eeg=1000e-6,      # unit: V (EEG channels)
              )

    flat = dict(eeg = 0.1e-6)

    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
   
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    if epochs.get_data().shape[0] < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        # Getting the first segment
        signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
        good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]

        # calculate sample entropy for all 19 channels
        sample_entropy = compute_samp_entropy(signal_i[good_channel_ids])
        good_channel_names = np.array(CHANNEL_NAMES_LIST)[good_channel_ids]
        good_dict = {channel_name: entropy for channel_name, entropy in zip(good_channel_names, sample_entropy)}

        df_features['subject'].append(subject)
        df_features['epoch'].append(i)
        for ch in range(len(cleaned.ch_names)):
            if CHANNEL_NAMES[ch] not in good_channel_names:
                df_features[f'sample_EN_{CHANNEL_NAMES[ch]}'].append(np.nan)
            else:  
                df_features[f'sample_EN_{CHANNEL_NAMES[ch]}'].append(good_dict[CHANNEL_NAMES[ch]])

    df_features = pd.DataFrame(df_features)
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
    result['subject'] = subject
    return result



start_time = time.time()
sampleEN_df_list = Parallel(n_jobs=-1)(
    delayed(extract_sample_entropy)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

all_sampleEN_df_list = pd.concat(sampleEN_df_list , axis=0)

all_sampleEN_df_list.reset_index().to_feather('all_sampleEN_df.feather')



"""
Extract statistical features: mean, SD, skewness, kurtosis
"""

def extract_statistical(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    reject = dict(
              eeg=1000e-6,      # unit: V (EEG channels)
              )

    flat = dict(eeg = 0.1e-6)

    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
   
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    if epochs.get_data().shape[0] < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        # Getting the first segment
        signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
        good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]

        # calculate mean for all 19 channels
        mean = compute_mean(signal_i[good_channel_ids])
        sd = compute_std(signal_i[good_channel_ids])
        kurt = compute_kurtosis(signal_i[good_channel_ids])
        skew = compute_skewness(signal_i[good_channel_ids])

        good_channel_names = np.array(CHANNEL_NAMES_LIST)[good_channel_ids]
        good_dict_mean = {channel_name: avg for channel_name, avg in zip(good_channel_names, mean)}
        good_dict_sd = {channel_name: std for channel_name, std in zip(good_channel_names, sd)}
        good_dict_skew = {channel_name: skw for channel_name, skw in zip(good_channel_names, skew)}
        good_dict_kurt = {channel_name: krt for channel_name, krt in zip(good_channel_names, kurt)}

        df_features['subject'].append(subject)
        df_features['epoch'].append(i)
        for ch in range(len(cleaned.ch_names)):
            if CHANNEL_NAMES[ch] not in good_channel_names:
                df_features[f'mean_{CHANNEL_NAMES[ch]}'].append(np.nan)
                df_features[f'sd_{CHANNEL_NAMES[ch]}'].append(np.nan)
                df_features[f'skew_{CHANNEL_NAMES[ch]}'].append(np.nan)
                df_features[f'kurt_{CHANNEL_NAMES[ch]}'].append(np.nan)

            else:  
                df_features[f'mean_{CHANNEL_NAMES[ch]}'].append(good_dict_mean[CHANNEL_NAMES[ch]])
                df_features[f'sd_{CHANNEL_NAMES[ch]}'].append(good_dict_sd[CHANNEL_NAMES[ch]])
                df_features[f'skew_{CHANNEL_NAMES[ch]}'].append(good_dict_skew[CHANNEL_NAMES[ch]])
                df_features[f'kurt_{CHANNEL_NAMES[ch]}'].append(good_dict_kurt[CHANNEL_NAMES[ch]])

    df_features = pd.DataFrame(df_features)
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
    result['subject'] = subject
    return result


start_time = time.time()
statistical_df_list = Parallel(n_jobs=-1)(
    delayed(extract_statistical)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

all_statistical_df_list = pd.concat(statistical_df_list , axis=0)

all_statistical_df_list.reset_index().to_feather('all_statistical.feather')



def extract_alpha_presence(subject):
   
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    df_features = defaultdict(list)
    df_features['subject'].append(subject)
    # Preprocess the data using light_preprocessing and remove_bad_channels_kevin functions
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned, subject)

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs and remove bad epochs
    reject = dict(eeg=1000e-6)  # unit: V (EEG channels)
    flat = dict(eeg=0.1e-6)
    epochs = mne.make_fixed_length_epochs(cleaned, duration=epoch_length, overlap=overlap, preload=True)
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    if epochs.get_data().shape[0] < 1:
        df_features['alpha_presence'].append(np.nan)
        return df_features

    signal = epochs.get_data().transpose(1, 0, 2).reshape(epochs._data.shape[1], -1)

    # Create an MNE Info object to describe the data
    channel_names = epochs.ch_names
    sfreq = epochs.info['sfreq']
    info = mne.create_info(channel_names, sfreq, ch_types='eeg')

    # Create an MNE RawEDF object from the signal and info
    concatenated_epoch = mne.io.RawArray(signal, info)
    result = philistine.mne.savgol_iaf(cleaned,picks=['O1','O2'], 
                            fmin=8, fmax=12,pink_max_r2 =1)
 
    df_features['alpha_presence'].append(result.PeakAlphaFrequency is not None)
    df_features = pd.DataFrame(df_features)
    # Return True if peak alpha frequency is not None, indicating alpha wave presence, and False otherwise
    return df_features

start_time = time.time()
alpha_df_list = Parallel(n_jobs=-1)(
    delayed(extract_alpha_presence)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

all_alpha_df_list = pd.concat(alpha_df_list , axis=0)

all_alpha_df_list.reset_index().to_feather('all_alpha.feather')


all_alpha_df_list.head()



def extract_length_after_dropped_epoch(subject):
    raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
    cleaned = light_preprocessing(raw)
    cleaned = remove_bad_channels_kevin(cleaned,subject)

    original_length = cleaned._data.shape[1] / 256 / 60

    epoch_length = 10  # seconds
    overlap = 2 # seconds

    # Segment the raw data into epochs
    # epochs shape (n_epochs, n_channels, n_times)
    reject = dict(
              eeg=500e-6,      # unit: V (EEG channels)
              )

    flat = dict(eeg = 0.1e-6)

    epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)
   
    n_epochs_before_drop = epochs.get_data().shape[0]
    
    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    
    n_epochs_after_drop = epochs.get_data().shape[0]
    


    length_remaining = n_epochs_after_drop * (epoch_length - overlap) / 60.0
    
    df = pd.DataFrame({'subject': [subject], 'length_remaining': [length_remaining], 'original_length': [original_length]})
    
    
    return df


start_time = time.time()
length_df_list = Parallel(n_jobs=-1)(
    delayed(extract_length_after_dropped_epoch)(subject) for subject in good_subjects)

end_time = time.time()
print(end_time - start_time)

length_df = pd.concat(length_df_list , axis=0)


length_df.to_csv('length_df.csv', index=False)

################################
# Sandbox #
################################


raw = mne.io.read_raw_edf('../eeg_data/main_edf/002.edf', infer_types=True, preload=True, misc=['Status'])
cleaned = light_preprocessing(raw)
cleaned = remove_bad_channels_kevin(cleaned,'002.edf')

epoch_length = 10  # seconds
overlap = 2 # seconds

# Segment the raw data into epochs
# epochs shape (n_epochs, n_channels, n_times)
epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)


############## figure out the parameters in reject {} and flat {} ################
psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
# take PSD between 0.5 to 40 Hz
freq_ids = (freqs>=0.5)&(freqs<=40)
psds = psds[:, :,freq_ids]
freqs = freqs[freq_ids]

ten_times_log_psd = 10 * np.log10(psds)
ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-white')
plt.plot(ten_times_log_psd_mean.sum(axis=1))
##################################################################################
reject = dict(
              eeg=1000e-6,      # unit: V (EEG channels)
              )

flat = dict(eeg = 0.1e-6)


epochs.drop_bad(reject=reject, flat=flat, verbose=None)





### Get features for one epoch
signal_i = epochs.get_data()[0,:,:]
#good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
#psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i[good_channel_ids], epochs[0].info['sfreq'], fmin=0.5, fmax=42)
psd_i, freq_i = mne.time_frequency.psd_array_multitaper(signal_i, epochs[0].info['sfreq'], fmin=0.5, fmax=42)


#########################################
# spectral coherence

# Compute connectivity for band containing the evoked response.
# We exclude the baseline period:
fmin, fmax = 1,4



#fmin = (1,4,8,13,30)
#fmax = (4,8,13,30,42)

sfreq = epochs.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
con = spectral_connectivity_epochs(
    epochs, method='coh', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

coherence_matrix = pd.DataFrame(con.get_data(output='dense')[:,:,0], columns=epochs.ch_names, index=epochs.ch_names)
coherence_matrix = coherence_matrix.replace(0, np.nan)
series = coherence_matrix .stack(dropna=True)

# concatenate the levels of the MultiIndex to form the column names of the output row
col_names = [f'{i}_{j}' for i, j in series.index]

# create a new DataFrame with a single row
flattened_features = pd.DataFrame([series.values], columns=col_names)

flattened_features = flattened_features.add_prefix('{}_coh_'.format('delta'))
# should have 171 columns



# create a new DataFrame with the selected columns
flattened_features.to_csv('coherence_flatten.csv')
# create a Pandas series with the feature names and values

plot_sensors_connectivity(
    epochs.info,
     con.get_data(output='dense')[:,:,0])
#########################################
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


################################
# Sample entropy
subject = '041.edf'
raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
cleaned = light_preprocessing(raw)
cleaned = remove_bad_channels_kevin(cleaned,subject)

epoch_length = 10  # seconds
overlap = 2 # seconds

# Segment the raw data into epochs
# epochs shape (n_epochs, n_channels, n_times)
reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
            )

flat = dict(eeg = 0.1e-6)

epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)

epochs.drop_bad(reject=reject, flat=flat, verbose=None)



df_features = defaultdict(list)

for i in range(len(epochs)):
    # Getting the first segment
    signal_i = epochs.get_data()[i,:,:] # shape = (n_channels, n_times)
    good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]

    # calculate sample entropy for all 19 channels
    sample_entropy = compute_samp_entropy(signal_i[good_channel_ids])
    good_channel_names = np.array(CHANNEL_NAMES_LIST)[good_channel_ids]
    good_dict = {channel_name: entropy for channel_name, entropy in zip(good_channel_names, sample_entropy)}

    df_features['subject'].append(subject)
    df_features['epoch'].append(i)
    for ch in range(len(cleaned.ch_names)):
        if CHANNEL_NAMES[ch] not in good_channel_names:
            df_features[f'sample_EN_{CHANNEL_NAMES[ch]}'].append(np.nan)
        else:  
            df_features[f'sample_EN_{CHANNEL_NAMES[ch]}'].append(good_dict[CHANNEL_NAMES[ch]])

df_features = pd.DataFrame(df_features)
result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
result['subject'] = subject




############################### 
# get whether have alpha

#################################

# pick O1 and O2 from epochs._data
# pass it into philistine.mne.savgol_iaf


signal = epochs.get_data().transpose(1, 0, 2).reshape(epochs._data.shape[1], -1)

# Create an MNE Info object to describe the data
channel_names = epochs.ch_names
sfreq = epochs.info['sfreq']
info = mne.create_info(channel_names, sfreq, ch_types='eeg')

# Create an MNE RawEDF object from the signal and info
concatenated_epoch = mne.io.RawArray(signal, info)
result = philistine.mne.savgol_iaf(cleaned,picks=[4,13], 
                          fmin=8, fmax=12)




subject = "001.edf"
raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
cleaned = light_preprocessing(raw)
cleaned = remove_bad_channels_kevin(cleaned,subject)

epoch_length = 10  # seconds
overlap = 2 # seconds

# Segment the raw data into epochs
# epochs shape (n_epochs, n_channels, n_times)
reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
            )

flat = dict(eeg = 0.1e-6)

epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)

n_epochs_before_drop = epochs.get_data().shape[0]

epochs.drop_bad(reject=reject, flat=flat, verbose=None)





subject = "335.edf"
raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(subject), infer_types=True, preload=True, misc=['Status'])
cleaned = light_preprocessing(raw)
cleaned = remove_bad_channels_kevin(cleaned,subject)

original_length = cleaned._data.shape[1] / 256 / 60

epoch_length = 10  # seconds
overlap = 2 # seconds

# Segment the raw data into epochs
# epochs shape (n_epochs, n_channels, n_times)
reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
            )

flat = dict(eeg = 0.1e-6)

epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap,preload=True)

n_epochs_before_drop = epochs.get_data().shape[0]

epochs.drop_bad(reject=reject, flat=flat, verbose=None)

n_epochs_after_drop = epochs.get_data().shape[0]

if n_epochs_after_drop < 1:
    return df_features

length_remaining = n_epochs_after_drop * (epoch_length - overlap) / 60.0
