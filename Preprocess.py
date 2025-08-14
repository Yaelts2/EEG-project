import mne
import pandas as pd
import numpy as np
from pathlib import Path
from mne.filter import filter_data
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.signal import welch

# === Paths ===
DATASET_ROOT = Path(r"../1ds006104")
SAVE_DIR = Path("processed_data")
SAVE_DIR.mkdir(exist_ok=True)
X_CSV = SAVE_DIR / "X.csv"
Y_CSV = SAVE_DIR / "y.csv"
LABELS_FILE = SAVE_DIR / "labels.txt"

# === Config ===
TIME_WINDOW = 1.0  # seconds, shortened for phoneme specificity
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOISE_THRESHOLD = 10.0
MIN_CHANNELS = 10

# === Utilities ===

def remove_noisy_channels(eeg, threshold=10.0):
    max_vals = np.max(np.abs(eeg), axis=1)
    keep_mask = max_vals < threshold
    return eeg[keep_mask, :]

def extract_features_statistical(eeg):
    features = []
    features.extend(np.mean(eeg, axis=1))
    features.extend(np.std(eeg, axis=1))
    features.extend(np.max(eeg, axis=1))
    features.extend(np.min(eeg, axis=1))
    features.extend(np.sum(eeg ** 2, axis=1))
    return np.array(features)

def compute_bandpower(eeg_segment, sfreq):
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    features = []
    for ch_data in eeg_segment:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))
        for fmin, fmax in bands.values():
            idx = (freqs >= fmin) & (freqs <= fmax)
            bandpower = np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0
            features.append(bandpower)
    return np.array(features)

def extract_features_combined(eeg, sfreq):
    stat_feats = extract_features_statistical(eeg)
    bp_feats = compute_bandpower(eeg, sfreq)
    return np.concatenate([stat_feats, bp_feats])

# === Step 1: Gather all phoneme labels from all subjects to fit LabelEncoder once ===
print("🔍 Gathering phonemes from all subjects to build LabelEncoder...")

subject_dirs = sorted(DATASET_ROOT.glob("sub-S14*"))  # Adjust subject pattern if needed
all_phonemes = []

for subject_dir in subject_dirs:
    eeg_path = subject_dir / "ses-02" / "eeg"
    tsv_file = next(eeg_path.glob("*_task-singlephoneme_events.tsv"), None)
    if not tsv_file:
        continue
    try:
        events_df = pd.read_csv(tsv_file, sep="\t")
    except Exception:
        continue
    stim_df = events_df[events_df['trial_type'] == 'stimulus'].dropna(subset=['phoneme1'])
    all_phonemes.extend(stim_df['phoneme1'].unique())

unique_phonemes = sorted(set(all_phonemes))
print(f"✅ Found {len(unique_phonemes)} unique phonemes: {unique_phonemes}")

label_encoder = LabelEncoder()
label_encoder.fit(unique_phonemes)
with open(LABELS_FILE, "w") as f:
    f.write(",".join(label_encoder.classes_))

# === Step 2: Process each subject and save features & labels ===

for subject_dir in subject_dirs:
    subject = subject_dir.name
    print(f"🔄 Processing {subject}...")

    eeg_path = subject_dir / "ses-02" / "eeg"
    edf_file = next(eeg_path.glob("*_task-singlephoneme_eeg.edf"), None)
    tsv_file = next(eeg_path.glob("*_task-singlephoneme_events.tsv"), None)
    if not edf_file or not tsv_file:
        print(f"❌ Missing files for {subject}, skipping.")
        continue

    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        events_df = pd.read_csv(tsv_file, sep="\t")
    except Exception as e:
        print(f"⚠️ Error loading {subject}: {e}")
        continue

    stim_df = events_df[events_df['trial_type'] == 'stimulus'].dropna(subset=['phoneme1'])
    sfreq = raw.info['sfreq']

    all_trials = []
    labels = []

    for _, row in stim_df.iterrows():
        onset_sample = int(row['onset'] * sfreq)
        start_sample = onset_sample - int(0.8 * sfreq)
        end_sample = start_sample + int(TIME_WINDOW * sfreq)

        if start_sample < 0 or end_sample > raw.n_times:
            continue

        eeg = raw.get_data(start=start_sample, stop=end_sample)
        eeg = filter_data(eeg, sfreq=sfreq, l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose=False)
        eeg = zscore(eeg, axis=1)
        eeg = remove_noisy_channels(eeg, threshold=NOISE_THRESHOLD)

        if eeg.shape[0] < MIN_CHANNELS:
            continue

        all_trials.append(eeg)
        labels.append(row['phoneme1'])

    if not all_trials:
        print(f"⚠️ No valid trials for {subject}")
        continue

    # Keep only common shape
    shapes = [t.shape for t in all_trials]
    most_common_shape = Counter(shapes).most_common(1)[0][0]
    common_trials = [t for t in all_trials if t.shape == most_common_shape]
    common_labels = [l for t, l in zip(all_trials, labels) if t.shape == most_common_shape]

    print(f"✅ Trials kept (common shape {most_common_shape}): {len(common_trials)}")

    # Extract combined features and encode labels
    X = np.array([extract_features_combined(eeg, sfreq) for eeg in common_trials])
    y = label_encoder.transform(common_labels)

    # Append to CSV files
    pd.DataFrame(X).to_csv(X_CSV, mode='a', header=not X_CSV.exists(), index=False)
    pd.DataFrame(y).to_csv(Y_CSV, mode='a', header=not Y_CSV.exists(), index=False)

    print(f"📥 Appended {len(X)} samples from {subject}.")

print("✅ Done processing all subjects.")
