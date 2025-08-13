import os
import sys
import glob
import time
import argparse
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import mne
from mne_connectivity import spectral_connectivity_epochs
from mne_features.univariate import (
    compute_samp_entropy,
    compute_mean,
    compute_std,
    compute_kurtosis,
    compute_skewness,
)
import philistine  # for alpha presence 


BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGE = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 42),
}

CHANNEL_NAMES = {
    0: "Fp1", 1: "F3", 2: "C3", 3: "P3", 4: "O1", 5: "F7", 6: "T3", 7: "T5",
    8: "Fz", 9: "Fp2", 10: "F4", 11: "C4", 12: "P4", 13: "O2", 14: "F8",
    15: "T4", 16: "T6", 17: "Cz", 18: "Pz"
}
CHANNEL_NAMES_LIST = list(CHANNEL_NAMES.values())

DEFAULT_EPOCH_LENGTH = 10  # seconds
DEFAULT_OVERLAP = 2        # seconds
DEFAULT_REJECT = dict(eeg=1000e-6)
DEFAULT_FLAT = dict(eeg=0.1e-6) #max peak-to-peak signal amplitude > 1000e-6 and min PTP < 0.1e-6

# -----------------------------
# Preprocessing
# -----------------------------
def light_preprocessing(raw):
    """
    This function changes LE to A1, drops A1 and A2,
    keeps only EEG channels, applies montage, and filters.
    """
    channel_renaming_dict = {name: name.replace('-LE', '-A1') for name in raw.ch_names}
    raw.rename_channels(mapping=channel_renaming_dict)

    channel_renaming_dict = {name: name.replace('-A1', '') for name in raw.ch_names}
    raw.rename_channels(mapping=channel_renaming_dict)

    raw.pick_types(eeg=True)

    if 'A2' in raw.ch_names:
        raw.drop_channels('A2')

    # Notch filtering for line noise at 50Hz
    raw.notch_filter(50)

    # Bandpass filter 0.5–42 Hz
    raw.filter(l_freq=0.5, h_freq=42)

    # Average reference
    raw.set_eeg_reference(ref_channels='average')

    return raw.copy()

# -----------------------------
# Utility helpers
# -----------------------------
def edf_path(data_root: str, subject: str) -> str:
    return os.path.join(data_root, subject)


def load_clean_epochs(
    data_root: str,
    subject: str,
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
    reject: dict = DEFAULT_REJECT,
    flat: dict = DEFAULT_FLAT,
):
    raw_path = edf_path(data_root, subject)
    raw = mne.io.read_raw_edf(raw_path, infer_types=True, preload=True, misc=["Status"])
    cleaned = light_preprocessing(raw)

    epochs = mne.make_fixed_length_epochs(
        cleaned, duration=epoch_length, overlap=overlap, preload=True
    )
    # Count before drop in case a function needs it
    n_before = len(epochs)

    epochs.drop_bad(reject=reject, flat=flat, verbose=None)
    n_after = len(epochs)

    return cleaned, epochs, n_before, n_after


# -----------------------------
# Feature extractors
# -----------------------------
def extract_relative_power(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    """ Extracts relative power spectral density for each band in each channel averaged across epochs."""
    cleaned, epochs, _, _ = load_clean_epochs(data_root, subject, epoch_length, overlap)
    if len(epochs) < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i, :, :]  # (n_channels, n_times)
        psd_i, freq_i = mne.time_frequency.psd_array_multitaper(
            signal_i, epochs.info["sfreq"], fmin=0.5, fmax=42
        )
        # Relative power normalization across frequency bins per channel
        denom = np.sum(psd_i, axis=-1, keepdims=True)
        denom[denom == 0] = np.nan
        psd_i = psd_i / denom

        df_features["subject"].append(subject)
        df_features["epoch"].append(i)
        # For each frequency band, sum the relative PSD over the band’s frequency range.
        for bn in BAND_NAMES:
            fmin, fmax = BAND_RANGE[bn]
            idx_band = (freq_i >= fmin) & (freq_i <= fmax)
            power = np.nansum(psd_i[:, idx_band], axis=1)  # (n_channels,)
            for ch in range(len(cleaned.ch_names)):
                df_features[f"bp_{bn}_{CHANNEL_NAMES[ch]}"].append(power[ch] if ch < len(power) else np.nan) # bp_alpha_Fp1

    df_features = pd.DataFrame(df_features)
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T  # average across epochs
    result["subject"] = subject
    return result


def extract_spectral_coh(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    """ Extracts channel-to-channel spectral coherence averaged across epochs for each channel pair for each band."""

    _, epochs, _, _ = load_clean_epochs(data_root, subject, epoch_length, overlap)
    if len(epochs) < 1:
        return None

    all_features = []
    sfreq = epochs.info["sfreq"]

    for bn in BAND_NAMES:
        fmin, fmax = BAND_RANGE[bn]
        con = spectral_connectivity_epochs(
            epochs,
            method="coh",
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True, # average across the coherence values over all bins in that the band 
            mt_adaptive=False,
        )
        # Value (i, j) is the coherence between channel i and channel j for this frequency band.
        coh_mat = pd.DataFrame(
            con.get_data(output="dense")[:, :, 0], columns=epochs.ch_names, index=epochs.ch_names
        ).replace(0, np.nan)
        # Create multi-index pandas series for coherence values
        series = coh_mat.stack(dropna=True)  
        col_names = [f"{i}_{j}" for (i, j) in series.index]
        flat = pd.DataFrame([series.values], columns=col_names)
        flat = flat.add_prefix(f"{bn}_coh_") # e.g. alpha_coh_F3_Pz
        all_features.append(flat)

    result = pd.concat(all_features, axis=1)
    result["subject"] = subject
    return result


def extract_sample_entropy(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    """ Extracts sample entropy for each channel in each epoch and then averages across epochs.
    Returns a DataFrame with average sample entropy"""

    cleaned, epochs, _, _ = load_clean_epochs(data_root, subject, epoch_length, overlap)
    if len(epochs) < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i, :, :]  # (n_channels, n_times)
        # Finds the indices of channels that are not entirely NaN in this epoch.
        good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
       # If all channels are bad for this epoch, skip to the next epoch.
        if len(good_channel_ids) == 0:
            continue

        samp_en = compute_samp_entropy(signal_i[good_channel_ids])
        good_names = np.array(CHANNEL_NAMES_LIST)[good_channel_ids]
        # Creates a dictionary mapping: Channel name to computed sample entropy value.
        good_map = {ch: en for ch, en in zip(good_names, samp_en)}

        df_features["subject"].append(subject)
        df_features["epoch"].append(i)
        for ch in range(len(cleaned.ch_names)):
            nm = CHANNEL_NAMES[ch]
            df_features[f"sample_EN_{nm}"].append(good_map.get(nm, np.nan))

    if len(df_features) == 0:
        return None

    df_features = pd.DataFrame(df_features)
    # Get average across epochs for each channel
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
    result["subject"] = subject
    return result


def extract_time_domain_statistical(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    """ Extracts common time domain statistical features for each channel in each epoch and then averages across epochs.
    Returns a DataFrame with average time domain features"""

    cleaned, epochs, _, _ = load_clean_epochs(data_root, subject, epoch_length, overlap)
    if len(epochs) < 1:
        return None

    df_features = defaultdict(list)

    for i in range(len(epochs)):
        signal_i = epochs.get_data()[i, :, :]  # (n_channels, n_times)
        good_channel_ids = np.where(~np.all(np.isnan(signal_i), axis=1))[0]
        if len(good_channel_ids) == 0:
            continue

        mean = compute_mean(signal_i[good_channel_ids])
        sd = compute_std(signal_i[good_channel_ids])
        kurt = compute_kurtosis(signal_i[good_channel_ids])
        skew = compute_skewness(signal_i[good_channel_ids])

        good_names = np.array(CHANNEL_NAMES_LIST)[good_channel_ids]
        map_mean = {n: v for n, v in zip(good_names, mean)}
        map_sd = {n: v for n, v in zip(good_names, sd)}
        map_kurt = {n: v for n, v in zip(good_names, kurt)}
        map_skew = {n: v for n, v in zip(good_names, skew)}

        df_features["subject"].append(subject)
        df_features["epoch"].append(i)
        for ch in range(len(cleaned.ch_names)):
            nm = CHANNEL_NAMES[ch]
            df_features[f"mean_{nm}"].append(map_mean.get(nm, np.nan))
            df_features[f"sd_{nm}"].append(map_sd.get(nm, np.nan))
            df_features[f"skew_{nm}"].append(map_skew.get(nm, np.nan))
            df_features[f"kurt_{nm}"].append(map_kurt.get(nm, np.nan))

    if len(df_features) == 0:
        return None

    df_features = pd.DataFrame(df_features)
    result = df_features.iloc[:, 2:].mean(axis=0).to_frame().T
    result["subject"] = subject
    return result


def extract_alpha_presence(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    """ detect alpha presence by finding Peak Alpha Frequency in occipital channels O1 and/or O2."""

    cleaned, _, _, _ = load_clean_epochs(data_root, subject, epoch_length, overlap)

    df = pd.DataFrame({"subject": [subject]})

    try:
        # Use channel names for O1/O2 if available
        picks = []
        for nm in ["O1", "O2"]:
            if nm in cleaned.ch_names:
                picks.append(nm)
        if len(picks) == 0:
            df["alpha_presence"] = np.nan
            return df

        res = philistine.mne.savgol_iaf(
            cleaned, picks=picks, fmin=8, fmax=12, pink_max_r2=1
        )
        df["alpha_presence"] = res.PeakAlphaFrequency is not None
        return df
    except Exception:
        logging.exception(f"Alpha presence failed for {subject}")
        df["alpha_presence"] = np.nan
        return df


def extract_length_after_dropped_epoch(
    subject,
    data_root: str = "../eeg_data/main_edf",
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
):
    cleaned, epochs, n_before, n_after = load_clean_epochs(
        data_root, subject, epoch_length, overlap
    )

    # Original minutes (assumes 256 Hz if not present; prefer info if available)
    sfreq = cleaned.info.get("sfreq", 256.0)
    original_length_min = cleaned._data.shape[1] / float(sfreq) / 60.0

    length_remaining_min = n_after * (epoch_length - overlap) / 60.0

    return pd.DataFrame(
        {
            "subject": [subject],
            "length_remaining_min": [length_remaining_min],
            "original_length_min": [original_length_min],
            "n_epochs_before": [n_before],
            "n_epochs_after": [n_after],
        }
    )


# -----------------------------
# Parallel processing functions #
# -----------------------------
def _safe_parallel_map(fn, subjects, n_jobs, desc):
    """Run a function over subjects in parallel"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_safe_wrap)(fn, s) for s in tqdm(subjects, desc=desc)
    )
    results = [r for r in results if r is not None]
    if len(results) == 0:
        return pd.DataFrame()
    return pd.concat(results, axis=0, ignore_index=True)


def _safe_wrap(fn, subject):
    try:
        return fn(subject)
    except Exception as e:
        logging.exception(f"Subject {subject} failed in {fn.__name__}: {e}")
        return None


def _merge_feature_tables(base_df: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-merge all feature tables on 'subject' starting from base_df."""
    out = base_df.copy()
    for t in tables:
        if t is None or t.empty:
            continue
        # de-duplicate columns except 'subject'
        dupes = [c for c in t.columns if c != "subject" and c in out.columns]
        if dupes:
            t = t.drop(columns=dupes)
        out = out.merge(t, on="subject", how="left")
    return out



def parse_args():
    p = argparse.ArgumentParser(description="EEG Feature Extraction CLI")
    p.add_argument("--data-root", type=str, default="../eeg_data/main_edf",
                   help="Directory containing EDF files.")
    p.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH,
                   help="Epoch length in seconds.")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help="Overlap in seconds.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for subject-level feature extraction.")
    p.add_argument("--output", type=str, required=True,
                   help="Output path for merged features (csv/parquet/feather).")
    p.add_argument("--format", type=str, choices=["csv", "parquet", "feather"], default="feather",
                   help="Output format.")
    p.add_argument("--enable", type=str, nargs="*", default=["power", "coherence", "sample_entropy", "statistical", "alpha", "length"],
                   help="Feature groups to enable: power, coherence, sample_entropy, statistical, alpha, length")
    return p.parse_args()


def resolve_subject_filepath(data_root: str):
    """Resolve subject EDF file paths in the given data root directory."""
    subs = [os.path.basename(p) for p in glob.glob(os.path.join(data_root, "*.edf"))]

    if not subs:
        logging.error(f"No EDF files found in: {data_root}")
        sys.exit(2)
    present = []
    for s in subs:
        if os.path.exists(edf_path(data_root, s)):
            present.append(s)

    if not present:
        logging.error("No valid subjects found to process.")
        sys.exit(2)

    return sorted(present)


def save_output(df: pd.DataFrame, path: str, fmt: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "feather":
        df.reset_index(drop=True).to_feather(path)
    logging.info(f"Wrote merged features to: {path} ({fmt})")


def main():
    args = parse_args()
    start = time.time()

    subjects = resolve_subject_filepath(args.data_root)
    base = pd.DataFrame({"subject": subjects})
    enabled = set([e.lower() for e in args.enable])
    feature_tables = []

    if "power" in enabled:
        logging.info("Extracting relative power…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_relative_power(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "relative_power"
            )
        )

    if "coherence" in enabled:
        logging.info("Extracting spectral coherence…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_spectral_coh(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "spectral_coherence"
            )
        )

    if "sample_entropy" in enabled:
        logging.info("Extracting sample entropy…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_sample_entropy(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "sample_entropy"
            )
        )

    if "statistical" in enabled:
        logging.info("Extracting statistical features…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_time_domain_statistical(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "statistical"
            )
        )

    if "alpha" in enabled:
        logging.info("Detecting alpha presence…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_alpha_presence(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "alpha_presence"
            )
        )

    if "length" in enabled:
        logging.info("Computing remaining length after dropped epochs…")
        feature_tables.append(
            _safe_parallel_map(
                lambda subject: extract_length_after_dropped_epoch(subject, args.data_root, args.epoch_length, args.overlap),
                subjects,
                args.n_jobs,
                "length_after_drop"
            )
        )

    merged = _merge_feature_tables(base, feature_tables)
    save_output(merged, args.output, args.format)
    logging.info(f"Done in {time.time() - start:.1f}s. Subjects processed: {len(subjects)}")


if __name__ == "__main__":
    main()
