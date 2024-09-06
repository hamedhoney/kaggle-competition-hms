from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Beta
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import timm
from configs.conf import Config

def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized Mean (GeM) Pooling"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

'''
ref:
https://amaarora.github.io/posts/2020-08-30-gempool.html
'''
class GeM(nn.Module):
    """Generalized Mean (GeM) Pooling Layer"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, p_trainable: bool = False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p) if p_trainable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self) -> str:
        p_value = self.p.data.tolist()[0] if isinstance(self.p, Parameter) else self.p
        return f"{self.__class__.__name__}(p={p_value:.4f}, eps={self.eps})"

class Mixup(nn.Module):
    """Mixup Augmentation Layer"""
    def __init__(self, mix_beta: float, mixadd: bool = False):
        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample((bs,)).to(X.device)

        # Apply Mixup
        X = coeffs.view(*([-1] + [1] * (n_dims - 1))) * X + (1 - coeffs.view(*([-1] + [1] * (n_dims - 1)))) * X[perm]

        # Modify Y based on mixadd flag
        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        return (X, Y, Z) if Z is not None else (X, Y)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Net(nn.Module):
    """Main Network Architecture"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_classes = 6
        self.preprocessing = nn.Sequential(
            MelSpectrogram(**cfg.model.spec_args),
            AmplitudeToDB()
        )

        # Backbone Model
        self.backbone = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.model.in_channels,
            **cfg.model.model_args
        )

        # Mixup Initialization
        self.mixup = Mixup(cfg.model.mixup_beta)
        self.mixup_signal = cfg.model.mixup_signal
        self.mixup_spectrogram = cfg.model.mixup_spectrogram
        backbone_out = self.backbone.num_features

        # Pooling Layer Selection
        self.global_pool = {
            "gem": GeM(p_trainable=cfg.model.gem_p_trainable),
            "identity": nn.Identity(),
            "avg": nn.AdaptiveAvgPool2d(1)
        }.get(cfg.model.pool, nn.AdaptiveAvgPool2d(1))

        # Head and Loss Function
        self.head = nn.Linear(backbone_out, self.n_classes)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        print('Model parameters:', count_parameters(self))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for training and inference."""
        x = batch['input'].permute(0, 2, 1).float()
        y = batch['target'].float()

        # Apply Mixup to Signal
        if self.training and self.mixup_signal:
            x, y = self.mixup(x, y)

        # Reshape and Preprocess
        bs, c, l = x.shape
        x = x.reshape(bs * c, l)
        x = self.preprocessing(x)
        _, h, w = x.shape
        x = x.reshape(bs, c, h, w)

        # Apply Spectrogram Mixup during Training
        if self.training:
            for tt in range(x.shape[0]):
                if self.cfg.training.aug_drop_spec_prob > np.random.random():
                    drop_ct = np.random.randint(1, 1 + self.cfg.training.aug_drop_spec_max)
                    drop_idx = np.random.choice(np.arange(x.shape[1]), drop_ct)
                    x[tt, drop_idx] = 0.0

        x = x.reshape(bs, 1, c * h, w)

        # Apply Mixup to Spectrogram
        if self.training and self.mixup_spectrogram:
            x, y = self.mixup(x, y)

        # Forward through Backbone and Pooling
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]

        # Calculate Logits and Loss
        logits = self.head(x)
        loss = self.loss_fn(F.log_softmax(logits, dim=1), y)
        outputs = {'loss': loss}

        # Add logits to outputs during inference
        if not self.training:
            outputs['logits'] = logits

        return outputs
