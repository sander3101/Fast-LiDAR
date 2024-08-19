import dataclasses
import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclasses.dataclass
class TrainingConfig:
    """
        Config class that defines all the parameters for training.
    """
    dataset: Literal["synlidar", "kitti_raw", "kitti_360", "all"] = "kitti_360"
    image_format: str = "log_depth"
    lidar_projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    train_mask: int = None
    resolution: tuple[int, int] = (64, 1024)
    min_depth = 1.45
    max_depth = 80.0
    batch_size_train: int = 8
    batch_size_eval: int = 8
    num_workers: int = 4
    num_steps: int = 300_000
    save_image_steps: int = 5_000
    save_model_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    criterion: str = "l2"
    lr: float = 1e-4
    lr_warmup_steps: int = 10_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    output_dir: str = "logs/diffusion"
    seed: int = 0
    mixed_precision: str = "fp16"
    dynamo_backend: str = "inductor"
    model_name: str = "efficient_unet"
    model_base_channels: int = 64
    model_temb_channels: int | None = None
    model_channel_multiplier: tuple[int] | int = (1, 2, 4, 8)
    model_num_residual_blocks: tuple[int] | int = 3
    model_gn_num_groups: int = 32 // 4
    model_gn_eps: float = 1e-6
    model_attn_num_heads: int = 8
    model_coords_embedding: Literal[
        "spherical_harmonics", "polar_coordinates", "fourier_features", None
    ] = "fourier_features"
    model_dropout: float = 0.0
    diffusion_num_training_steps: int = 1024
    diffusion_num_sampling_steps: int = 128
    diffusion_objective: Literal["eps", "v", "x_0"] = "eps"
    diffusion_beta_schedule: str = "cosine"
    diffusion_timesteps_type: Literal["continuous", "discrete"] = "continuous"
    # Parameter defines if model should train on generating point clouds, or be masked to train on inpainting
    diffusion_task: Literal["inpaint", "generate"] = "inpaint"
    # Defines the inpainting task for training, if mixed is set, all features will be combined
    diffusion_mask: Literal["simple", "complex", "pepper", "upsample", "mixed"] = (
        "mixed"
    )
    # List contains the desired features for mixed training. 
    # Features should be written and included in with the same words as in diffusion_mask (mixed is not a feature)
    mixed_tasks = ["complex", "pepper", "upsample"]
    # Defines sparsity level of the conditional masks
    sparsity_level: float = None
    # Name of model
    project_name: str = "r2dm_only_complex_pepper_upsample_continous"

    ### Remove this before more experiments with coordinates ###
    # model_coords_embedding: Literal[
    #     "spherical_harmonics", "polar_coordinates", "fourier_features", None
    # ] = "None"
    ############################################################

    #### DEBUGGING PARAMETERS (COMMENT OUT WHEN USING) ####
    # Use this for running code without compiling
    ### CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py --dynamo_backend no ###
    # num_steps: int = 300
    # save_image_steps: int = 30
    # diffusion_num_training_steps: int = 10
    # diffusion_num_sampling_steps: int = 10
    # save_model_steps: int = 10
    # project_name: str = "test"

    ## HF_DATASETS_CACHE=/media/sensing/ssd1/huggingface/datasets

    ## Run this for all three datasets ##
    ## HF_DATASETS_CACHE=/media/sensing/ssd1/huggingface/datasets/ CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py ##


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
