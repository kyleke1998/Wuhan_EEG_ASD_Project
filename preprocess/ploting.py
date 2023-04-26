import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import ast
import cleaning
from cleaning import light_preprocessing, remove_bad_channels_kevin, metadata, good_metadata, channels_to_drop_dict_from_kevin

# Plot Side by side PSDS from multitaper
# Same x-axis, y-axis.

with PdfPages('psd_from_multitaper.pdf') as pdf:
    for sample in good_metadata['file_name'].values:
        exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
        eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
        age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
        sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5),sharex=True)
        fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))


        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        channel_renaming_dict = {name: name.replace('-LE', '-A1')
                                for name in raw.ch_names}
        raw.rename_channels(mapping=channel_renaming_dict)
        # Plotting psd
        raw.plot_psd(ax = ax1)


        ax1.set_xlabel('Frequency (Hz)')
        cleaned = light_preprocessing(raw)
        cleaned = remove_bad_channels_kevin(cleaned,sample)

        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

        epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap)
        reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
            )

        flat = dict(eeg = 0.1e-6)

        epochs.drop_bad(reject=reject, flat=flat, verbose=None)



        # AFTER DROPPING BAD EPOCHS
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        ten_times_log_psd = 10 * np.log10(psds)
        power = ten_times_log_psd.mean(axis=0)
        # x freq
        # y power
        for i, channel in enumerate(cleaned.ch_names):
            ax2.plot(freqs, power[i], label=channel)

        # set plot properties
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('10 * log10(PSD)')
        ax2.set_title('PSDS From Multitaper (After Cleaning)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        pdf.savefig(fig)
       









# Plot Side by side
# Same x-axis, y-axis.

for sample in all_batch_1_edfs:
    
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5),sharex=True,sharey=True)
        fig.suptitle('{}'.format(sample))


        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        channel_renaming_dict = {name: name.replace('-LE', '-A1')
                                for name in raw.ch_names}
        raw.rename_channels(mapping=channel_renaming_dict)
        # Plotting psd
        raw.plot_psd(ax = ax1)
        fig2 = raw.plot(start=50, duration=10,show=False,scalings={'eeg':10e-5},title='Raw')
        # Plot raw waves

        # Step 1
        # fix channel namead


        channel_renaming_dict = {name: name.replace('-A1', '')
                                for name in raw.ch_names}
                                
        raw.rename_channels(mapping=channel_renaming_dict)
        #raw.ch_names

        # keep only EEG channels and apply montage
        raw.pick_types(eeg=True)
        # Original Montages
        raw.set_montage('standard_1020')
    
        raw.drop_channels('A2')
        # Notch filtering. Filter line noise at 50Hz
        raw.notch_filter(50)



        # low pass filter at 42Hz and high pass filter at 0.5Hz
        raw.filter(l_freq = 0.5, h_freq=42)
        raw.set_eeg_reference(ref_channels='average')
        ax1.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Frequency (Hz)')
        raw.plot_psd(ax=ax2)
        fig3 = raw.plot(start=50, duration=10,show=False,scalings={'eeg':10e-5},title='Cleaned')
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)









############################ Multitaper Spectrogram ############################

# For asd subjects
with PdfPages('spectrogram.pdf') as pdf:
    for sample in good_metadata['file_name'].values:
    #for sample in ['135.edf','136.edf','228.edf','333.edf','7016.edf','7044.edf']:
        exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
        eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
        age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
        sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]
        
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        

        #### T = 10 ######
        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

      

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5),sharex=False,sharey=True)
        fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))
        epochs = mne.make_fixed_length_epochs(raw,duration=epoch_length,overlap=overlap)
        
        # BEFORE DROPPING BAD EPOCHS
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        # take PSD between 0.5 to 20 Hz
        freq_ids = (freqs>=0.5)&(freqs<=20)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        #psds_mean = psds.mean(axis=1)
        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
        #vmin, vmax = -95, -65
        #xmax = ((epochs.get_data().shape[0] - 1)  * 8  + 10 )/ 60
        xmax = raw.times[-1] / 60
        im1 = ax1.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')
        plt.colorbar(im1)
        ax1.set_title('Before Dropping Bad Epochs')
        ax1.set_xlabel('Time (mins)')
        ax1.set_ylabel('Frequency (Hz)')



        cleaned = light_preprocessing(raw)
        cleaned = remove_bad_channels_kevin(cleaned,sample)

        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

        epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap)
        reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
            )

        flat = dict(eeg = 0.1e-6)


        epochs.drop_bad(reject=reject, flat=flat, verbose=None)
        # Set parameters for multitaper analysis
        # Compute the power spectral density using multitapers
        #TBP = N * (1 - k / N)
        #bandwidth = 2 * (TBP / window_length)
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        # take PSD between 0.5 to 20 Hz
        freq_ids = (freqs>=0.5)&(freqs<=20)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        #psds_mean = psds.mean(axis=1)
        #vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
        #vmin, vmax = -95, -65
        xmax = ((epochs.get_data().shape[0] - 1)  * 8  + 10 ) / 60
        #xmax = raw.times[-1] / 60
        im2 = ax2.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')
        plt.colorbar(im2)
        ax2.set_title('After Preprocessing and Dropping Bad Epochs')
        ax2.set_xlabel('Time (mins)')
        ax2.set_ylabel('Frequency (Hz)')
        
        

        # Adjust the spacing between subplots
        fig.tight_layout()
        pdf.savefig(fig)






with PdfPages('spectrogram_2.pdf') as pdf:
    for sample in good_metadata['file_name'].values[:5]:
        exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
        eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
        age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
        sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]
        
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        
        

        cleaned = light_preprocessing(raw)
        cleaned = remove_bad_channels_kevin(cleaned,sample)

        #### T = 10 ######
        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

      

        fig, ax1 = plt.subplots(1, 1, figsize=(20,5), sharex=False, sharey=True)
        fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))
        epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap)
        
    
        # Set parameters for multitaper analysis

        # BEFORE DROPPING BAD EPOCHS
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        

        # take PSD between 0.5 to 20 Hz
        freq_ids = (freqs>=0.5)&(freqs<=20)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram

        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
        xmax = raw.times[-1] / 60
        im1 = ax1.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')


            # Drop bad epochs
        reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
        )
        flat = dict(eeg = 0.1e-6)
        dropped_epochs = epochs.drop_bad(reject=reject, flat=flat, verbose=None)
        
        # Annotate the spectrogram with the location of dropped epochs
        for idx in dropped_epochs.selection:
            start_time = dropped_epochs.events[idx, 0] / cleaned.info['sfreq']
            end_time = dropped_epochs.events[idx, 0] / cleaned.info['sfreq'] + epoch_length
            rect = plt.Rectangle((start_time, freqs[0]), end_time - start_time, freqs[-1] - freqs[0], edgecolor='black',
                    facecolor='none')
            ax1.add_patch(rect)

        plt.colorbar(im1)
        fig.tight_layout()
        pdf.savefig(fig)




with PdfPages('spectrogram_cleaned.pdf') as pdf:
    for sample in good_metadata['file_name'][0:2]:
        exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
        eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
        age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
        sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]
        
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        

        cleaned = light_preprocessing(raw)
        cleaned = remove_bad_channels_kevin(cleaned,sample)
        
        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

        fig, ax1 = plt.subplots(1, 1, figsize=(20,5), sharex=False, sharey=True)
        fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))
        epochs = mne.make_fixed_length_epochs(raw,duration=epoch_length,overlap=overlap)

        # Set parameters for multitaper analysis
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        
        # take PSD between 0.5 to 20 Hz
        freq_ids = (freqs>=0.5)&(freqs<=20)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
        xmax = raw.times[-1] / 60
        im1 = ax1.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')


        # Preprocess the data
        #cleaned = light_preprocessing(raw)
        #cleaned = remove_bad_channels_kevin(cleaned,sample)

       
        # Drop bad epochs and mark their location on spectrogram with vertical lines
        reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
        )
        flat = dict(eeg=0.1e-6)

        good_epochs = epochs.copy()
        dropped_epochs = epochs.drop_bad(reject=reject, flat=flat, verbose=None)
        dropped_log = dropped_epochs.drop_log
        for idx,event in enumerate(dropped_log):
            if len(event) != 0:
                onset = good_epochs.events[idx, 0] / raw.info['sfreq'] / 60
                ax1.vlines(onset, freqs[0], freqs[-1], colors='red', linestyle='dashed')

        plt.colorbar(im1)
        fig.tight_layout()
        pdf.savefig(fig)




with PdfPages('spectrogram_cleaned_horizontal_3.pdf') as pdf:
    for sample in good_metadata['file_name']:
        exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
        eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
        age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
        sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]

        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])

        cleaned = light_preprocessing(raw)
        cleaned = remove_bad_channels_kevin(cleaned,sample)

        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 2  # percentage of overlap between epochs

        fig, ax1 = plt.subplots(1, 1, figsize=(20,5), sharex=False, sharey=True)
        fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))
        epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap)

        # Set parameters for multitaper analysis
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)

        # take PSD between 0.5 to 20 Hz
        freq_ids = (freqs>=0.5)&(freqs<=20)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
        xmax = raw.times[-1] / 60
        im1 = ax1.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')

        # Preprocess the data
        #cleaned = light_preprocessing(raw)
        #cleaned = remove_bad_channels_kevin(cleaned,sample)

        # Drop bad epochs and mark their location on spectrogram with horizontal bars
        reject = dict(
            eeg=1000e-6,      # unit: V (EEG channels)
        )
        flat = dict(eeg=0.1e-6)

        good_epochs = epochs.copy()
        dropped_epochs = epochs.drop_bad(reject=reject, flat=flat, verbose=None)
        dropped_log = dropped_epochs.drop_log
        for idx,event in enumerate(dropped_log):
            if len(event) != 0:
                onset = good_epochs.events[idx, 0] / raw.info['sfreq'] / 60
                ax1.hlines(y=freqs[-1]+1, xmin=onset, xmax=onset+epoch_length/60, colors='red', linestyle='solid', linewidth=2)

        plt.colorbar(im1)
        fig.tight_layout()
        pdf.savefig(fig)
           
           
           
           
sample = '001.edf'
exp_group = metadata.query("file_name == '{}'".format(sample))['exp_group'].to_numpy()[0]
eoec = metadata.query("file_name == '{}'".format(sample))['EOEC'].to_numpy()[0]
age = metadata.query("file_name == '{}'".format(sample))['age'].to_numpy()[0]
sex = metadata.query("file_name == '{}'".format(sample))['sex'].to_numpy()[0]

raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])

cleaned = light_preprocessing(raw)
cleaned = remove_bad_channels_kevin(cleaned,sample)

# Define epoch parameters for T = 10
epoch_length = 10  # seconds
overlap = 2  # percentage of overlap between epochs

fig, ax1 = plt.subplots(1, 1, figsize=(20,5), sharex=False, sharey=True)
fig.suptitle('{} ({}, {}, [{}], {})'.format(sample,eoec,exp_group,age,sex))
epochs = mne.make_fixed_length_epochs(cleaned,duration=epoch_length,overlap=overlap)

# Set parameters for multitaper analysis
psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)

# take PSD between 0.5 to 20 Hz
freq_ids = (freqs>=0.5)&(freqs<=20)
psds = psds[:, :,freq_ids]
freqs = freqs[freq_ids]

ten_times_log_psd = 10 * np.log10(psds)
ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

# Plot the multitaper spectrogram
vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,95])
xmax = raw.times[-1] / 60
im1 = ax1.imshow(ten_times_log_psd_mean.T, extent=[0,xmax, freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax, interpolation='sinc')

# Preprocess the data
#cleaned = light_preprocessing(raw)
#cleaned = remove_bad_channels_kevin(cleaned,sample)

# Drop bad epochs and mark their location on spectrogram with horizontal bars
reject = dict(
    eeg=500e-6,      # unit: V (EEG channels)
)
flat = dict(eeg=0.1e-6)

good_epochs = epochs.copy()
dropped_epochs = epochs.drop_bad(reject=reject, flat=flat, verbose=None)
dropped_log = dropped_epochs.drop_log
for idx,event in enumerate(dropped_log):
    if len(event) != 0:
        onset = good_epochs.events[idx, 0] / raw.info['sfreq'] / 60
        ax1.hlines(y=freqs[-1]+1, xmin=onset, xmax=onset+epoch_length/60, colors='red', linestyle='solid', linewidth=2)

plt.colorbar(im1)

length_df.query("length_remaining < 1").merge(metadata,right_on='file_name',left_on='subject', how='left').exp_group.value_counts()