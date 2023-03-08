import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import ast



path = '../eeg_data/main_edf'
all_batch_1_edfs = [f for f in os.listdir(path) if f.endswith('.edf')]
path_cleaned = '../eeg_data/main_set_cleaned'
all_cleaned = [elem[:-3] + 'set' for elem in all_batch_1_edfs]


metadata = pd.read_csv('../eeg_data/wuhan_study_clinical_data.csv')
metadata['exp_group'] = metadata['exp_group'].fillna('NA')
metadata['EOEC'] = metadata['EOEC'].fillna('NA')
not_na_subjects = metadata.query("exp_group != 'NA' & EOEC != 'NA' ")['file_name'].values
good_metadata = metadata.query("exp_group != 'NA' & EOEC != 'NA'")


artifacts_list = pd.read_csv('../preprocessing/artifact_list.csv')
artifacts_list['additional_bad_channel_index'] = artifacts_list['additional_bad_channel_index'].str.split(',')
# list of subjects to remove completely
subjects_to_remove = artifacts_list.query("(ecg_artifact == 1) | (channel_drift_other == 1)")['file_name'].to_list()

# list of subjects to not remove bad channels
subjects_do_not_drop_channels = artifacts_list.query("(bad_channels_visual == 0) & (bad_channel_auto == 1)")['file_name'].to_list()


# Additional subjects with bad channels
additional_subjects_with_bad_channels = artifacts_list[~artifacts_list['additional_bad_channel_index'].isna()]['file_name'].to_list()




# all subjects with exp_group equals ASD
asd_subjects = good_metadata.query("exp_group == 'ASD'")['file_name'].values
# all subjects with exp_group equals control
control_subjects = good_metadata.query("exp_group != 'ASD'")['file_name'].values



# Get all bad channels from claned data identified by Kevin
channels_to_drop_dict_from_kevin = pd.read_csv('channel_to_drop_kevin_dict.csv',sep=',')
channels_to_drop_dict_from_kevin['Value'] = channels_to_drop_dict_from_kevin['Value'].apply(lambda x: ast.literal_eval(x))
channels_to_drop_dict_from_kevin = channels_to_drop_dict_from_kevin.set_index('Key').to_dict(orient='dict')['Value']



"""
# Plot Side by side
# Same x-axis, y-axis.
with PdfPages('time_and_psd_2.pdf') as pdf:
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





"""



############################ Multitaper Spectrogram ############################

# For asd subjects
with PdfPages('spectrogram_asd.pdf') as pdf:
    for sample in asd_subjects:

        fig, axs = plt.subplots(1, 1)
        fig.suptitle('{}'.format(sample))
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        

        #### T = 10 ######
        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 5  # percentage of overlap between epochs

        # Segment the raw data into epochs
        
        epochs = mne.make_fixed_length_epochs(raw,duration=epoch_length,overlap=overlap)
        
        # Set parameters for multitaper analysis
        # Compute the power spectral density using multitapers
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        # take PSD between 0.5 to 40 Hz
        freq_ids = (freqs>=0.5)&(freqs<=40)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        #psds_mean = psds.mean(axis=1)
        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,90])
        im1 = axs.imshow(ten_times_log_psd_mean.T, extent=[epochs.times[0], epochs.times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax)
        axs.set_title('{} T = 10'.format(sample))
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Frequency (Hz)')
        
        

        # Adjust the spacing between subplots
        fig.tight_layout()
        pdf.savefig(fig)

# for control subjects

with PdfPages('spectrogram_controls.pdf') as pdf:
    for sample in control_subjects:

        fig, axs = plt.subplots(1, 1)
        fig.suptitle('{}'.format(sample))
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        

        #### T = 10 ######
        # Define epoch parameters for T = 10
        epoch_length = 10  # seconds
        overlap = 5  # overlap in seconds

        # Segment the raw data into epochs
        
        epochs = mne.make_fixed_length_epochs(raw,duration=epoch_length,overlap=overlap)
        
        # Set parameters for multitaper analysis
        # Compute the power spectral density using multitapers
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), epochs.info['sfreq'],bandwidth=1)
        # take PSD between 0.5 to 40 Hz
        freq_ids = (freqs>=0.5)&(freqs<=40)
        psds = psds[:, :,freq_ids]
        freqs = freqs[freq_ids]

        ten_times_log_psd = 10 * np.log10(psds)
        ten_times_log_psd[np.isinf(ten_times_log_psd)] = np.nan
        ten_times_log_psd_mean = np.nanmean(ten_times_log_psd, axis=1)

        # Plot the multitaper spectrogram
        #psds_mean = psds.mean(axis=1)
        vmin, vmax = np.nanpercentile(ten_times_log_psd_mean.flatten(), [10,90])
        im1 = axs.imshow(ten_times_log_psd_mean.T, extent=[epochs.times[0], epochs.times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax)
        axs.set_title('{} T = 10'.format(sample))
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Frequency (Hz)')
        
        

        # Adjust the spacing between subplots
        fig.tight_layout()
        pdf.savefig(fig)



##################################################################################################
""" Get all the bad channels from kevin's cleaned data"""



def get_all_bad_channels_from_cleaned():
    #### Get all the bad channels from kevin's cleaned data ####
    channel_names_list = ['Fp1','F3','C3','P3','O1','F7','T3','T5','Fz','Fp2','F4','C4','P4',
    'O2','F8','T4','T6','Cz','Pz']
    channel_names = {i: channel_names_list[i] for i in range(len(channel_names_list))}
    channel_names  = {value: key for key, value in channel_names.items()}

    channels_to_drop_dict = {}
    for sample_cleaned, sample in zip(all_cleaned,all_batch_1_edfs):
        channel_to_drop_index = []
        raw = mne.io.read_raw_eeglab('../eeg_data/main_set_cleaned/{}'.format(sample_cleaned),preload=True)
        channels_to_drop = list(set(channel_names_list) - set(raw.ch_names))

        for channel in channels_to_drop:
            channel_to_drop_index.append(channel_names[channel])
        channels_to_drop_dict[sample] = channel_to_drop_index 
    return channels_to_drop_dict


##############################################################################################
## Remove Bad Channels##
###################################################################################

logger = logging.getLogger('bad_channels')
logger.propagate = False

# Set the logging level
logger.setLevel(logging.INFO)
# Create a file handler and set the logging level
file_handler = logging.FileHandler('bad_channels.txt')
# Add the file handler to the logger
logger.addHandler(file_handler)       


def light_preprocessing(raw):
    
    """
    This function changes LE to A1
    drops A1 and A2 
    """
    channel_renaming_dict = {name: name.replace('-LE', '-A1')
    for name in raw.ch_names}
    raw.rename_channels(mapping=channel_renaming_dict)

    # Step 1
    # fix channel names


    channel_renaming_dict = {name: name.replace('-A1', '')
    for name in raw.ch_names}

    raw.rename_channels(mapping=channel_renaming_dict)
    #raw.ch_names

    # keep only EEG channels and apply montage
    raw.pick_types(eeg=True)

    raw.drop_channels('A2')
    # Notch filtering. Filter line noise at 50Hz
    raw.notch_filter(50)
    # low pass filter at 42 Hz
    raw.filter(l_freq = 0.5, h_freq=42)
    raw.set_eeg_reference(ref_channels='average')
    cleaned = raw.copy()
    return cleaned




###### Function to remove bad channels ####
def remove_bad_channels(cleaned,sample):
    psds, freqs = mne.time_frequency.psd_array_welch(cleaned.get_data(), cleaned.info['sfreq'], fmin=0, fmax=70)
    # 10 * log10 to get psds to avoid numerical underflow
    transformed_psds = 10 * np.log10(psds)
    # Replace all values that are infinite with np.nan
    transformed_psds[np.isinf(transformed_psds)] = np.nan
    # Calculate the median across all frequencies
    average_over_freqs = np.nanmedian(transformed_psds, axis=1)
    # Steps to complete the function 
    # if Q3 + 1.5IQR or < Q1 - 1.5IQR, remove channel
    Q3 = np.nanpercentile(average_over_freqs, 75)
    Q1 = np.nanpercentile(average_over_freqs, 25)
    IQR = Q3 - Q1
    bad_channels_index = np.where((average_over_freqs > Q3 + 1.5*IQR) | (average_over_freqs < Q1 - 1.5*IQR))

    
    # Replace all bad channel values with np.nan
    if ((len(bad_channels_index) != 0) & (sample not in subjects_do_not_drop_channels)):
        cleaned._data[bad_channels_index, :] = np.nan
        # Writing dropped channel to log file
        logger.info('Channel {} removed from {}'.format(np.array(cleaned.ch_names)[bad_channels_index], sample))

   
    return cleaned


##### Function to remove bad channels from the bad channels identified by kevin
def remove_bad_channels_kevin(cleaned,sample):

    cleaned._data[channels_to_drop_dict_from_kevin[sample], :] = np.nan
    return cleaned


"""
for sample in not_na_subjects[:1]:
    if sample not in subjects_to_remove:
        raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
        cleaned = light_preprocessing(raw)
        #cleaned = remove_bad_channels(cleaned,sample)
        cleaned = remove_bad_channels_kevin(cleaned,sample)
        #test.append(remove_bad_channels(cleaned,sample))
"""

"""
################################ Plot Spectrogram of Notch filter @ 50Hz, low pass filter @ 70Hz vs all of these steps plus autoreject ################################
with PdfPages('autoreject.pdf') as pdf:
    for sample in not_na_subjects[:1]:
            fig, axes = plt.subplots(1, 2,figsize=(20,5),sharex=True,sharey=True)
            fig.suptitle('{}'.format(sample))
            raw = mne.io.read_raw_edf('../eeg_data/main_edf/{}'.format(sample), infer_types=True, preload=True, misc=['Status'])
            cleaned = light_preprocessing(raw)

            epoch_length = 10  # seconds
            overlap = 0.5  # percentage of overlap between epochs

            # Segment the raw data into epochs
            events = mne.make_fixed_length_events(cleaned, duration=epoch_length, overlap=epoch_length*overlap)
            cleaned.set_montage('standard_1020')
            epochs = mne.Epochs(cleaned, events, tmin=0, tmax=epoch_length, baseline=None, preload=True)
            ar = autoreject.AutoReject(random_state=11, n_jobs=10, verbose=True)
            ar.fit(epochs)
            epochs_ar, reject_log = ar.transform(epochs, return_log=True)
            epochs_ar.plot_psd(ax=axes[1])
            cleaned.plot_psd(ax=axes[0])
            # Set parameters for multitaper analysis
            fig2, ax = plt.subplots(1, 1,figsize=(20,5))
            reject_log.plot('horizontal',ax=ax)
            axes[0].set_title('Notch filter @ 50Hz, low pass filter @ 70Hz')
            axes[1].set_title('Notch filter @ 50Hz, low pass filter @ 70Hz + Autoreject')
            pdf.savefig(fig)
            pdf.savefig(fig2)
            

"""
