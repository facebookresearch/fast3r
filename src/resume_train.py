import os
import sys
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Import the train function from train.py
from train import train

from src.utils import (
    extras,
    get_metric_value,
)

def load_resume_config(run_dir: str) -> DictConfig:
    """
    Loads the configuration from 'hydra/config.yaml' in the specified run directory for resuming training.

    :param run_dir: Path to the run directory.
    :return: Loaded DictConfig.
    """
    config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Load the configuration
    cfg = OmegaConf.load(config_path)

    # Set checkpoint path to resume from last checkpoint
    last_ckpt_path = os.path.join(run_dir, 'checkpoints', 'last.ckpt')
    if os.path.exists(last_ckpt_path):
        cfg.ckpt_path = last_ckpt_path
    else:
        print("Warning: No checkpoint file found; starting from scratch.")
    
    return cfg


def resume_train(run_dir: str) -> Optional[float]:
    """
    Resumes training from the last checkpoint and configuration found in the specified directory.
    
    :param run_dir: Path to the directory containing the last checkpoint and config.
    :return: Optional[float] with optimized metric value.
    """
    cfg = load_resume_config(run_dir)

    # Apply any extras such as tags or printing config
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resume_train.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    resume_train(run_dir)
