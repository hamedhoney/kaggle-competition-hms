import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader, WeightedRandomSampler, Dataset
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Callable, Tuple, List, Union
import math
from preprocessing.dataset import CustomDataset
from configs.conf import Config, TrainingConfig

# Set seed for reproducibility
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Initialize worker function to ensure different random states in data loading
def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Function to get cosine learning rate scheduler with warmup
def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Calculate gradient norm for monitoring purposes
def calc_grad_norm(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    norm_type: float = 2.
) -> Optional[torch.Tensor]:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return None if torch.logical_or(total_norm.isnan(), total_norm.isinf()) else total_norm

# Synchronize tensors across GPUs
def sync_across_gpus(t: torch.Tensor, world_size: int) -> torch.Tensor:
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)

# Create checkpoint for model saving
def create_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> dict:
    state_dict = model.state_dict()
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

# Dataset and DataLoader creation
def get_dataset(
    df: pd.DataFrame,
    data_path: str,
    aug_bandpass_prob: float,
    aug_bandpass_max: float,
    aug: Callable,
    data_sample: int,
    mode: str = 'train'
) -> Dataset:
    data_set = CustomDataset(df, data_path, aug_bandpass_prob, aug_bandpass_max, aug, mode=mode)
    if data_sample > 0:
        data_set = torch.utils.data.Subset(data_set, np.arange(data_sample))
    return data_set

def get_dataloader(
    ds: CustomDataset,
    config: Config,
    mode: str = 'train'
) -> DataLoader:
    if mode == 'train':
        return get_train_dataloader(ds, config)
    elif mode == 'val':
        return get_val_dataloader(ds, config)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def get_train_dataloader(train_ds: CustomDataset, config: Config) -> DataLoader:
    sampler = None
    try:
        if config.random_sampler_frac > 0:
            num_samples = int(len(train_ds) * config.random_sampler_frac)
            sampler = WeightedRandomSampler(train_ds.sample_weights, num_samples=num_samples)
    except:
        pass
    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=config.dataset.batch_size,
        num_workers=config.resources.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=config.dataset_functions.tr_collate_fn,
        drop_last=config.dataset.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader

def get_val_dataloader(val_ds: CustomDataset, config: Config) -> DataLoader:
    batch_size = config.dataset.batch_size_val if config.dataset.batch_size_val else config.dataset.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=SequentialSampler(val_ds),
        batch_size=batch_size,
        num_workers=config.resources.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=config.dataset_functions.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader

# Optimizer creation
def get_optimizer(model: torch.nn.Module, config: TrainingConfig) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# Scheduler creation
def get_scheduler(config: Config, optimizer: optim.Optimizer, total_steps: int, world_size: int) -> LambdaLR:
    num_warmup_steps = config.training.warmup * (total_steps // config.dataset.batch_size) // world_size
    num_training_steps = config.training.epochs * (total_steps // config.dataset.batch_size) // world_size
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# K-fold creation and data splitting
def make_kfold(df: pd.DataFrame, k: int) -> None:
    if 'patient_id' not in df.columns:
        raise ValueError("The 'patient_id' column is not found in the CSV file.")

    unique_ids = df['patient_id'].unique()
    np.random.shuffle(unique_ids)
    split_ids = np.array_split(unique_ids, k)
    fold_mapping = {id_: fold + 1 for fold, ids in enumerate(split_ids) for id_ in ids}
    df['fold'] = df['patient_id'].map(fold_mapping)

def get_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(config.dataset.train_df)
    if 'fold' not in df.columns:
        make_kfold(df, 4)
        df.to_csv(config.dataset.train_df, index=False)

    test_df = pd.read_csv(config.dataset.test_df) if config.stages.test else None

    if config.dataset.val_df:
        val_df = pd.read_csv(config.dataset.val_df)
        val_df = val_df[val_df["fold"] == (0 if config.training.fold == -1 else config.training.fold)]
    else:
        val_df = df[df["fold"] == (0 if config.training.fold == -1 else config.training.fold)]
    
    train_df = df[df["fold"] != config.training.fold]
    return train_df, val_df, test_df

# Utility function to flatten lists of lists
def flatten(t: List[List]) -> List:
    return [item for sublist in t for item in sublist]

# Configure pandas display options
def set_pandas_display() -> None:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
