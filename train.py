import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path

import datasets as ds
import einops
import matplotlib.cm as cm
import torch
import torch._dynamo
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random as rnd

import utils.render
import utils.training
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.efficient_unet import EfficientUNet
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles

warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True


def train(cfg):
    """
        Training loop script
    """

    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.output_dir) / cfg.dataset / cfg.lidar_projection

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        if cfg.project_name is None:
            project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        else:
            project_name = cfg.project_name

        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.train_depth else 0,
        1 if cfg.train_reflectance else 0,
    ]

    if cfg.model_name == "efficient_unet":
        unet = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.resolution,
            base_channels=cfg.model_base_channels,
            temb_channels=cfg.model_temb_channels,
            channel_multiplier=cfg.model_channel_multiplier,
            num_residual_blocks=cfg.model_num_residual_blocks,
            gn_num_groups=cfg.model_gn_num_groups,
            gn_eps=cfg.model_gn_eps,
            attn_num_heads=cfg.model_attn_num_heads,
            coords_embedding=cfg.model_coords_embedding,
            ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model_name}")

    if "spherical" in cfg.lidar_projection:
        accelerator.print("set HDL-64E linear ray angles")
        unet.coords = get_hdl64e_linear_ray_angles(*cfg.resolution)
    elif "unfolding" in cfg.lidar_projection:
        accelerator.print("set dataset ray angles")
        _coords = torch.load(f"data/{cfg.dataset}/unfolding_angles.pth")
        unet.coords = F.interpolate(_coords, size=cfg.resolution, mode="nearest-exact")
    else:
        raise ValueError(f"Unknown: {cfg.lidar_projection}")

    if accelerator.is_main_process:
        print(f"number of parameters: {utils.training.count_parameters(unet):,}")

    if cfg.diffusion_timesteps_type == "discrete": # Discrete time diffusion
        ddpm = DiscreteTimeGaussianDiffusion(
            denoiser=unet,
            criterion=cfg.criterion,
            num_training_steps=cfg.diffusion_num_training_steps,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    elif cfg.diffusion_timesteps_type == "continuous": # Continuous time diffusion
        ddpm = ContinuousTimeGaussianDiffusion(
            denoiser=unet,
            criterion=cfg.criterion,
            objective=cfg.diffusion_objective,
            beta_schedule=cfg.diffusion_beta_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion_timesteps_type}")
    ddpm.train()
    ddpm.to(device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.ema_decay,
            update_every=cfg.ema_update_every,
            update_after_step=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    lidar_utils = LiDARUtility(
        resolution=cfg.resolution,
        image_format=cfg.image_format,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        ray_angles=ddpm.denoiser.coords,
    )
    lidar_utils.to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Defines datasets for training
    if cfg.dataset == "all": # All three datasets: SynLiDAR, KITTI-RAW, KITTI-360
        dataset1 = ds.load_dataset(
            path=f"data/synlidar",
            name=cfg.lidar_projection,
            split=ds.Split.TRAIN,
            num_proc=cfg.num_workers,
        ).with_format("torch")

        dataset2 = ds.load_dataset(
            path=f"data/kitti_raw",
            name=cfg.lidar_projection,
            split=ds.Split.TRAIN,
            num_proc=cfg.num_workers,
        ).with_format("torch")

        dataset3 = ds.load_dataset(
            path=f"data/kitti_360",
            name=cfg.lidar_projection,
            split=ds.Split.TRAIN,
            num_proc=cfg.num_workers,
        ).with_format("torch")

        dataset = ds.concatenate_datasets([dataset1, dataset2, dataset3])
        dataset = dataset.shuffle(seed=cfg.seed)

    else: # Designated dataset only
        dataset = ds.load_dataset(
            path=f"data/{cfg.dataset}",
            name=cfg.lidar_projection,
            split=ds.Split.TRAIN,
            num_proc=cfg.num_workers,
        ).with_format("torch")

    print(len(dataset))

    if accelerator.is_main_process:
        print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    lr_scheduler = utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.num_steps * cfg.gradient_accumulation_steps,
    )

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utility
    # =================================================================================

    def add_complex_sparsity(
        img,
        sparsity_level=0.1,
        start_row=0,
        stop_row=None,  # None mean we use the full length N
        start_line=0,
        stop_line=None,  # None mean we use the full length M
        up_prob=0.1,
        down_prob=0.1,
        thickness=1,
    ):
        """
        Adds noicy sparsity mask lines to an image

        Args:
            img (np.array) : image
            sparsity_levels (float): Percentage of sparsity level
            start_row (int): start row of mask
            stop_row (int): stop row of mask
            start_line (int): start line of mask
            stop_line (int): stop line of mask
            up_prob (float): Probability of line inclining
            down_prob (float): Probability of line declining
            thickness (int): thickness of mask line

        Return:
            mask (numpy.array): binary masked area
        """

        image = img[0, :, :]
        N, M = image.shape
        mask = np.zeros((N, M))

        # Define row stopping point for noise
        stop_row = N if stop_row == None else stop_row
        # Define line stopping point for noise
        stop_line = M if stop_line == None else stop_line
        # Make sure the thickness of the line can't be smaller than 1
        thickness = 1 if thickness <= 0 else thickness

        for i in range(start_row, stop_row):
            # Probability of adding sparsity
            if rnd.random() < sparsity_level:
                # Counter to controll incline and decline of the lines
                height_shift = 0
                changed_counter = 0

                for j in range(start_line, stop_line):
                    """
                    We use the formula 1/(log(|height_shift|)+1),
                    in order to controll the amount incline and decline.
                    The probability of continous incline and decline quickly declines,
                    making it unprobable for a line to continue in one direction.
                    """
                    # The probability of change decreases for each change
                    if rnd.random() < 1 / (abs(changed_counter) + 1):
                        if changed_counter < 5:
                            changed_counter += 1
                        else:
                            changed_counter = 0

                        if height_shift < 0:  # If the line is declining
                            if rnd.random() > 10 * up_prob / (
                                np.log(abs(height_shift)) + 1
                            ):
                                height_shift += 1
                            else:
                                height_shift -= 1

                        if height_shift > 0:  # If the line is inclining
                            if rnd.random() > 10 * down_prob / (
                                np.log(abs(height_shift)) + 1
                            ):
                                height_shift -= 1
                            else:
                                height_shift += 1

                    # If the coordinate is still inbounds we mask the pixel
                    if (
                        i + height_shift + thickness < N
                        and i + height_shift - thickness >= 0
                    ):
                        mask[
                            i + height_shift - thickness : i + height_shift + thickness,
                            j,
                        ] = 1

                    # 1/3 chance of going straight, incline or decline
                    if height_shift == 0:
                        randnum = rnd.random()
                        if randnum < 0.3:  # Declining
                            height_shift -= 1
                        elif randnum > 0.7:  # Inclining
                            height_shift += 1

        return mask

    def add_sparsity(img, sparsity_level=0.1):
        """
            Generates simple lines mask
        """
        image = img[0, :, :]
        N, M = image.shape
        mask = np.zeros((N, M))

        for i in range(N):
            mask[i, :] = rnd.random() < sparsity_level

        return mask

    def add_pepper(img, sparsity_level=0.1):
        """
            Generates pepper noise mask
        """
        image = img[0, :, :]
        N, M = image.shape
        mask = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                mask[i, j] = rnd.random() < sparsity_level

        return mask

    def upsampling(img, loss_level=2):
        """
            Generates upsampling mask
        """
        image = img[0, :, :]
        N, M = image.shape
        mask = np.zeros((N, M))

        for i in range(N):
            if i % loss_level != 0:
                mask[i, :] = 1

        return mask

    def preprocess(
        batch,
        mode="simple",
        sparsity=None,
        options=["simple", "complex", "pepper", "upsample"],
    ):
        """
            Preprocessess the data, and generates the designated conditional mask for training
        """
        x = []
        if cfg.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.resolution,
            mode="nearest-exact",
        )

        if cfg.diffusion_task == "generate": # Generate 3D point cloud unconditional
            return x, torch.ones_like(x, device=device)

        if sparsity is None: # Sets sparsity level to 10%-30% if not defined
            sparsity = rnd.uniform(0.1, 0.3)

        if mode == "simple": # Simple lines mask
            mask = torch.stack(
                [
                    torch.tensor(add_sparsity(element, sparsity_level=sparsity))
                    for element in x
                ]
            )

        elif mode == "complex": # Complex lines mask
            mask = torch.stack(
                [
                    torch.tensor(add_complex_sparsity(element, sparsity_level=sparsity))
                    for element in x
                ]
            )
        elif mode == "pepper": # Pepper noise mask
            mask = torch.stack(
                [
                    torch.tensor(add_pepper(element, sparsity_level=sparsity))
                    for element in x
                ]
            )
        elif mode == "upsample": # Upsampling mask
            mask = torch.stack(
                # Various upsampling rate from 2-8
                # [torch.tensor(upsampling(element, rnd.randint(2, 8))) for element in x]
                # 4x upsampling rate
                [torch.tensor(upsampling(element, 4)) for element in x]
            )
        elif mode == "mixed": # Mixed training task
            random_select = rnd.choice(options)

            if random_select == "complex":
                N, M = x[0, 0, :, :].shape

                mask = torch.stack(
                    [
                        torch.tensor(
                            add_complex_sparsity(
                                element,
                                sparsity_level=rnd.uniform(0.1, 0.4),
                                start_row=rnd.randint(0, N // 2),
                                stop_row=rnd.randint(N // 2, N - 1),
                                start_line=rnd.randint(0, M // 2),
                                stop_line=rnd.randint(M // 2, M - 1),
                                up_prob=rnd.uniform(0.1, 0.2),
                                down_prob=rnd.uniform(0.1, 0.2),
                                thickness=rnd.randint(1, 2),
                            )
                        )
                        for element in x
                    ]
                )
            elif random_select == "upsample":
                mask = torch.stack(
                    # [
                    #     torch.tensor(upsampling(element, rnd.randint(2, 8)))
                    #     for element in x
                    # ]
                    [torch.tensor(upsampling(element, 4)) for element in x]
                )
            elif random_select == "pepper":
                mask = torch.stack(
                    [
                        torch.tensor(
                            add_pepper(element, sparsity_level=rnd.uniform(0.1, 0.8))
                        )
                        for element in x
                    ]
                )
            else:  ### THIS NEEDS TO BE CHANGED IF WE WANT TO TEST TRAININ WITHOUT SIMPLE LINES ###
                mask = torch.stack(
                    [
                        torch.tensor(
                            add_sparsity(element, sparsity_level=rnd.uniform(0.1, 0.8))
                        )
                        for element in x
                    ]
                )

        return x, mask.unsqueeze(1).to(device)

    def split_channels(image: torch.Tensor):
        """
            Split depth and reflectance/intensity channels
        """
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        """
            Creates images and for tensorboard for visual purposes
        """
        image = lidar_utils.denormalize(image)
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f"{tag}/depth"] = utils.render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = utils.render.colorize(
                metric / lidar_utils.max_depth
            )
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -utils.render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            bev = utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
                t=torch.tensor([0, 0, 1.0]).to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f"{tag}/reflectance"] = utils.render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================

    progress_bar = tqdm(
        range(cfg.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    while global_step < cfg.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_0, masks = preprocess(
                batch,
                cfg.diffusion_mask,
                sparsity=cfg.sparsity_level,
                options=cfg.mixed_tasks,
            )

            with accelerator.accumulate(ddpm):
                loss = ddpm(x_0=x_0, mask=masks)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                if global_step == 1:
                    log_images(x_0, "image", global_step)

                # Conditional sample image
                if global_step % cfg.save_image_steps == 0:
                    ddpm_ema.ema_model.eval()
                    sample = ddpm_ema.ema_model.conditional_sample(
                        batch_size=masks.shape[0],
                        num_steps=cfg.diffusion_num_sampling_steps,
                        rng=torch.Generator(device=device).manual_seed(0),
                        mask=masks.float(),
                        x_0=x_0,
                    )
                    log_images(sample, "sample", global_step)

                if global_step % cfg.save_model_steps == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(utils.training.TrainingConfig, dest="cfg")
    cfg: utils.training.TrainingConfig = parser.parse_args().cfg
    train(cfg)
