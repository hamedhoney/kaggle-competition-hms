import os
from typing import Any, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from scipy.signal import butter, lfilter

# Default collate functions
tr_collate_fn = torch.utils.data.dataloader.default_collate
val_collate_fn = torch.utils.data.dataloader.default_collate

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Moves a batch of data to the specified device."""
    return {key: value.to(device) for key, value in batch.items()}


def butter_bandpass_filter(data: np.ndarray, high_freq: float = 20, low_freq: float = 0.5, sampling_rate: int = 200, order: int = 2) -> np.ndarray:
    """Applies a Butterworth bandpass filter to the data."""
    nq_freq = 0.5 * sampling_rate
    high_cutoff = high_freq / nq_freq
    low_cutoff = low_freq / nq_freq
    b, a = butter(order, [low_cutoff, high_cutoff], btype='band', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

# EEG columns and their flipped counterparts for augmentation
eeg_cols = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
eeg_cols_flipped = ['Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1']
flip_map = dict(zip(eeg_cols, eeg_cols_flipped))


class CustomDataset(Dataset):
    """Custom Dataset for EEG data with optional augmentation."""
    
    def __init__(
            self,
            df: pd.DataFrame,
            data_folder: str,
            aug_bandpass_prob: float=0.2,
            aug_bandpass_max: float=8.0,
            aug: Any = None,
            mode: str = "train"
        ):
        """
        Initializes the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing dataset information.
         (Any): Configuration object with dataset parameters.
            aug (Any, optional): Augmentation object (default: None).
            mode (str): Mode of the dataset, either 'train' or 'val'/'test'.
        """
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        self.data_folder = data_folder
        self.aug_bandpass_prob=aug_bandpass_prob
        self.aug_bandpass_max=aug_bandpass_max
        
        print(f"Mode: {mode}, DataFrame shape: {df.shape}")

        votes = [n for n in df.columns if 'vote' in n]
        self.votes = votes

        # Normalize votes
        self.df[votes] = self.df[votes].values / self.df[votes].values.sum(axis=1, keepdims=True)
        self.eegs = self.df['eeg_id'].values

        # Define specific EEG columns for signal processing
        self.label1 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.label2 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a data sample from the dataset."""
        row = self.df.iloc[idx]
        eeg_id, eeg_label_offset_seconds = row[['eeg_id', 'eeg_label_offset_seconds']].astype(np.int64)
        y = row[self.votes].values.astype(np.float32)
        
        eeg, center = self.load_one(eeg_id, eeg_label_offset_seconds)

        return {
            "input": torch.from_numpy(eeg),
            "center": torch.tensor(center, dtype=torch.long),
            "target": torch.from_numpy(y)
        }

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.eegs)

    def load_one(self, eeg_id: int, eeg_label_offset_seconds: int = 0) -> Tuple[np.ndarray, int]:
        """
        Loads a single EEG record and applies preprocessing and augmentation.
        
        Args:
            eeg_id (int): Identifier of the EEG record.
            eeg_label_offset_seconds (int): Time offset for label alignment.
        
        Returns:
            Tuple[np.ndarray, int]: Processed EEG data and the center point.
        """
        file_path = os.path.join(self.data_folder, f'{eeg_id}.parquet')
        try:
            eeg_combined = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return np.zeros((10000, len(self.label1))), 0  # Return dummy data in case of error

        start = int(200 * eeg_label_offset_seconds)
        win_len = 10000

        # Data augmentation during training
        if self.mode == "train":
            if np.random.rand() < 0.5:
                eeg_combined = eeg_combined.rename(columns=flip_map)
            
            start_shifted = int(np.random.uniform(start - win_len // 3, start + win_len // 3))
            start_shifted = np.clip(start_shifted, 0, eeg_combined.shape[0] - win_len)
        else:
            start_shifted = start
        
        shift = start - start_shifted
        eeg = eeg_combined.iloc[start_shifted:start_shifted + win_len]
        x = eeg[self.label1].values - eeg[self.label2].values
        x[np.isnan(x)] = 0

        # Apply bandpass filter
        x = butter_bandpass_filter(x)
        
        # Additional filtering augmentation
        if self.mode == "train" and np.random.random() < self.aug_bandpass_prob:
            filt_idx = np.random.choice(np.arange(x.shape[-1]), 1 + np.random.randint(self.aug_bandpass_max))
            high_freq_aug = np.random.randint(10, 20)
            low_freq_aug = np.random.uniform(0.0001, 2)
            x[:, filt_idx] = butter_bandpass_filter(x[:, filt_idx], high_freq=high_freq_aug, low_freq=low_freq_aug)
        
        # Clip and normalize
        x = np.clip(x, -1024, 1024)
        x /= 32
        
        center = shift + win_len // 2
        
        return x, center
