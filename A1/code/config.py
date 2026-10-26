from dataclasses import dataclass, asdict, field

from pathlib import Path

import json

from typing import Union, List, Optional

from utils import get_net_dims


@dataclass
class DataConfig:
    dataset: str = "fashionmnist"
    batch_size: int = 1024
    seed: int = 111
    split_seed: int = 2026
    val_split: float = 0.1


@dataclass
class ModelConfig:
    model: str = "cnn"
    num_heads: int = 2
    clock_method: Optional[int] = 1
    num_clock_classes: int = 720
    n_hidden_fc_layers: int = 2
    fc_width: int = 256
    kernel_size: int = 7
    dropout: float = 0.2
    batch_norm: bool = False
    bias: bool = False
    activation_method: str = "relu"
    init_method: str = "kaiming_uniform"

    H: int = field(init=False)
    W: int = field(init=False)
    C_i: int = field(init=False)
    out_feat: Union[int, List[int]] = field(init=False)
    num_output_heads: int = field(init=False)

    def compute_dimensions(self, dataset: str):
        """Computes dependent spatial dimensions and head output features."""
        H, W, out_feat, C_i, num_output_heads = get_net_dims(
            self.clock_method, dataset.lower(), self.num_clock_classes
        )
        self.H = H
        self.W = W
        self.C_i = C_i
        self.out_feat = out_feat
        self.num_output_heads = num_output_heads


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    regularization_term: str = ""
    save: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Ensure derived network parameters are calculated upon configuration build."""
        self.model.compute_dimensions(self.data.dataset)

    @classmethod
    from_argparse(cls, args: argparse.Namespace):
        """
        Creates an ExperimentConfig directly from an argparse Namespace.
        Maps matching keys dynamically and calculates derived dimensions automatically.
        """
        arg_dict = vars(args)

        data_fields = {f.name for f in DataConfig.__dataclass_fields__.values()}
        model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
        training_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}

        data_cfg = DataConfig(**{k: arg_dict[k] for k in data_fields if k in arg_dict})
        model_cfg = ModelConfig(**{k: arg_dict[k] for k in model_fields if k in arg_dict})
        training_cfg = TrainingConfig(**{k: arg_dict[k] for k in training_fields if k in arg_dict})

        return cls(data=data_cfg, model=model_cfg, training=training_cfg)

    def to_dict(self) -> dict:
        """Returns full nested config dictionary for JSON logging."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }

    def save_json(self, filepath: Path):
        """Saves experiment hyperparameters to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)