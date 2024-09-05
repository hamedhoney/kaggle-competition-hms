from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional, Dict, Any
import numpy as np

# Training Stages Configuration
@dataclass
class StagesConfig:
    train: bool = True
    val: bool = True
    test: bool = True
    train_val: bool = True
    eval_epochs: int = 1

# Dataset Configuration
@dataclass
class DatasetConfig:
    name: str = "sample"
    data_path: Optional[str] = None
    out_dir: Optional[str] = None
    data_folder: Optional[str] = None
    data_folder_spec: Optional[str] = None
    test_data_folder: Optional[str] = None
    train_df: Optional[str] = None
    val_df: Optional[str] = None
    test_df: Optional[str] = None
    batch_size: int = 8
    batch_size_val: Optional[int] = None
    normalization: str = 'image'
    train_aug: Optional[Any] = None
    val_aug: Optional[Any] = None
    data_sample: int = -1
    aug_drop_spec_prob: float = 0.5
    aug_drop_spec_max: int = 8
    aug_bandpass_prob: float = 0.2
    aug_bandpass_max: int = 8
    enlarge_len: int = 0
    drop_last: bool = True

# Model Configuration
@dataclass
class ModelConfig:
    name: str = "mdl_1"
    backbone: str = "mixnet_xl"
    pool: str = "gem"
    gem_p_trainable: bool = True
    pretrained: bool = True
    in_channels: int = 1
    mixup_spectrogram: bool = False
    mixup_signal: bool = False
    mixup_beta: float = 1.0
    spec_args: Dict[str, Any] = field(default_factory=lambda: {
        "sample_rate": 200,
        "n_fft": 1024,
        "n_mels": 128,
        "f_min": 0.53,
        "f_max": 40,
        "win_length": 128,
        "hop_length": 39
    })
    model_args: Dict[str, Any] = field(default_factory=lambda: {
        "drop_rate": 0.2,
        "drop_path_rate": 0.2
    })

# Training Routine Configuration
@dataclass
class TrainingConfig:
    fold: int = 0
    val_fold: int = -1
    epochs: int = 12
    lr: float = 0.0012
    schedule: str = "cosine"
    weight_decay: float = 0.0
    optimizer: str = "Adam"
    seed: int = -1
    resume_training: bool = False
    simple_eval: bool = False
    do_test: bool = True
    do_seg: bool = False
    eval_ddp: bool = True
    clip_grad: float = 20.0
    debug: bool = False
    save_val_data: bool = True
    gradient_checkpointing: bool = False
    awp: bool = False
    awp_per_step: bool = False
    pseudo_df: Optional[str] = None
    warmup: int = 0
    pin_memory: bool = False
    grad_accumulation: float = 4.0
    aug_drop_spec_prob: float = 0.5
    aug_drop_spec_max: int = 8
    aug_bandpass_prob: float = 0.2
    aug_bandpass_max: int = 8
    enlarge_len: int = 0

# Evaluation Configuration
@dataclass
class EvaluationConfig:
    calc_metric: bool = False
    calc_metric_epochs: int = 1
    eval_steps: int = 0
    post_process_pipeline: str = "pp_dummy"
    metric: str = "default_metric"

# Resources and Performance Configuration
@dataclass
class ResourcesConfig:
    find_unused_parameters: bool = False
    mixed_precision: bool = False
    syncbn: bool = False
    gpu: int = 0
    dp: bool = False
    num_workers: int = 8
    drop_last: bool = True
    save_checkpoint: bool = True
    save_only_last_ckpt: bool = True
    save_weights_only: bool = True
    save_first_batch: bool = False

# Logging Configuration
@dataclass
class LoggingConfig:
    flush_period: int = 30
    tags: Optional[str] = None
    save_first_batch: bool = False
    save_first_batch_preds: bool = False
    sgd_nesterov: bool = True
    sgd_momentum: float = 0.9
    clip_mode: str = "norm"
    track_grad_norm: bool = True
    grad_norm_type: float = 2.0
    norm_eps: float = 1e-4
    disable_tqdm: bool = False

# Dataset Functions and Collation
@dataclass
class DatasetFunctionsConfig:
    CustomDataset: Optional[Any] = None
    tr_collate_fn: Optional[Any] = None
    val_collate_fn: Optional[Any] = None
    batch_to_device: Optional[Any] = None

# Main Configuration Dataclass
@dataclass
class Config:
    stages: StagesConfig = StagesConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    resources: ResourcesConfig = ResourcesConfig()
    logging: LoggingConfig = LoggingConfig()
    dataset_functions: DatasetFunctionsConfig = DatasetFunctionsConfig()
