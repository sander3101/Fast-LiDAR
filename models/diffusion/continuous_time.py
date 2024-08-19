import math
from functools import partial
from typing import List, Literal

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm

from . import base


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_snr_schedule_linear(t: torch.Tensor) -> torch.Tensor:
    return -log(expm1(1e-4 + 10 * (t**2)))


def log_snr_schedule_cosine(
    t: torch.Tensor,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def log_snr_schedule_cosine_shifted(
    t: torch.Tensor,
    image_d: float,
    noise_d: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    log_snr = log_snr_schedule_cosine(t, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    shift = 2 * math.log(noise_d / image_d)
    return log_snr + shift


def log_snr_schedule_cosine_interpolated(
    t: torch.Tensor,
    image_d: float,
    noise_d_low: float,
    noise_d_high: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    logsnr_low = log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_low, logsnr_min, logsnr_max
    )
    logsnr_high = log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_high, logsnr_min, logsnr_max
    )
    return t * logsnr_low + (1 - t) * logsnr_high


class ContinuousTimeGaussianDiffusion(base.GaussianDiffusion):
    """
    Continuous-time Gaussian diffusion
    https://arxiv.org/pdf/2107.00630.pdf
    """

    def __init__(
        self,
        denoiser: nn.Module,
        criterion: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        objective: Literal["eps", "v", "x0"] = "eps",
        beta_schedule: Literal[
            "linear", "cosine", "cosine_shifted", "cosine_interpolated"
        ] = "cosine",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
        image_d: float = None,
        noise_d_low: float = None,
        noise_d_high: float = None,
    ):
        super().__init__(
            denoiser=denoiser,
            sampling="ddpm",
            criterion=criterion,
            num_training_steps=None,
            objective=objective,
            beta_schedule=beta_schedule,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            sampling_resolution=sampling_resolution,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        self.image_d = image_d
        self.noise_d_low = noise_d_low
        self.noise_d_high = noise_d_high

    def setup_parameters(self) -> None:
        if self.beta_schedule == "linear":
            self.log_snr = log_snr_schedule_linear
        elif self.beta_schedule == "cosine":
            self.log_snr = log_snr_schedule_cosine
        elif self.beta_schedule == "cosine_shifted":
            assert self.image_d is not None and self.noise_d_low is not None
            self.log_snr = partial(
                log_snr_schedule_cosine_shifted,
                image_d=self.image_d,
                noise_d=self.noise_d_low,
            )
        elif self.beta_schedule == "cosine_interpolated":
            assert (
                self.image_d is not None
                and self.noise_d_low is not None
                and self.noise_d_high is not None
            )
            self.log_snr = partial(
                log_snr_schedule_cosine_interpolated,
                image_d=self.image_d,
                noise_d_low=self.noise_d_low,
                noise_d_high=self.noise_d_high,
            )
        else:
            raise ValueError(f"invalid beta schedule: {self.beta_schedule}")

    @staticmethod
    def log_snr_to_alpha_sigma(log_snr):
        alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
        return alpha, sigma

    def get_target(self, x_0, step_t, noise):
        if self.objective == "eps":
            target = noise
        elif self.objective == "x0":
            target = x_0
        elif self.objective == "v":
            log_snr = self.log_snr(step_t)[:, None, None, None]
            alpha, sigma = self.log_snr_to_alpha_sigma(log_snr)
            target = alpha * noise - sigma * x_0
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return target

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # continuous timesteps
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_denoiser_condition(self, steps):
        return self.log_snr(steps)

    @autocast(enabled=False)
    def q_sample(self, x_0, step_t, noise):
        # forward diffusion process q(zt|x0) where 0<t<1
        log_snr = self.log_snr(step_t)[:, None, None, None]
        alpha, sigma = self.log_snr_to_alpha_sigma(log_snr)
        x_t = x_0 * alpha + noise * sigma
        return x_t

    def get_loss_weight(self, steps):
        log_snr = self.log_snr(steps)[:, None, None, None]
        snr = log_snr.exp()
        clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            clipped_snr.clamp_(max=self.min_snr_gamma)
        if self.objective == "eps":
            loss_weight = clipped_snr / snr
        elif self.objective == "v":
            loss_weight = clipped_snr / (snr + 1)
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return loss_weight

    @torch.inference_mode()
    def p_sample(
        self,
        x_t: torch.Tensor,
        step_t: torch.Tensor,
        step_s: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
        mode: Literal["ddpm", "ddim"] = "ddpm",
    ) -> torch.Tensor:
        # reverse diffusion process p(zs|zt) where 0<s<t<1
        log_snr_t = self.log_snr(step_t)[:, None, None, None]
        log_snr_s = self.log_snr(step_s)[:, None, None, None]
        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = self.log_snr_to_alpha_sigma(log_snr_s)
        prediction = self.denoiser(x_t, log_snr_t[:, 0, 0, 0])
        if self.objective == "eps":
            x_0 = (x_t - sigma_t * prediction) / alpha_t
        elif self.objective == "v":
            x_0 = alpha_t * x_t - sigma_t * prediction
        elif self.objective == "x0":
            x_0 = prediction
        else:
            raise ValueError(f"invalid objective {self.objective}")
        if self.clip_sample:
            x_0.clamp_(-self.clip_sample_range, self.clip_sample_range)
        if mode == "ddpm":
            c = -expm1(log_snr_t - log_snr_s)
            mean = alpha_s * (x_t * (1 - c) / alpha_t + c * x_0)
            var = sigma_s.pow(2) * c
            var_noise = self.randn_like(x_t, rng=rng)
            var_noise[step_t == 0] = 0
            x_s = mean + var.sqrt() * var_noise
        elif mode == "ddim":
            noise = (x_t - alpha_t * x_0) / sigma_t.clamp(min=1e-8)
            x_s = alpha_s * x_0 + sigma_s * noise
        else:
            raise ValueError(f"invalid mode {mode}")
        return x_s

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
    ):
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        for i in tqdm(range(num_steps), **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            x = self.p_sample(x, step_t, step_s, rng=rng, mode=mode)
            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x

    torch.inference_mode()

    def conditional_sample(
        self,
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        mask: torch.Tensor = None,
        x_0: torch.Tensor = None,
    ):
        """
            Sampling method that takes in a conditonal mask and generates a sample with a condition for a given timestep
        """
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]

        ### Mask x to match sparse data ###
        if mask is not None:
            x = x * mask + (1 - mask) * x_0

        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)

        for i in tqdm(range(num_steps), **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            x = self.p_sample(x, step_t, step_s, rng=rng, mode=mode)
            
            ### Fill old area with original image and only keep generated part ###
            if not mask is None:
                x = x * mask + (1 - mask) * x_0

            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x

    def q_step_back(self, x_s, step_t, step_s, rng=None):
        # q(zt|zs) where 0<s<t<1
        # cf. Appendix A of https://arxiv.org/pdf/2107.00630.pdf
        log_snr_t = self.log_snr(step_t)[:, None, None, None]
        log_snr_s = self.log_snr(step_s)[:, None, None, None]
        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = self.log_snr_to_alpha_sigma(log_snr_s)
        alpha_ts = alpha_t / alpha_s
        var_noise = self.randn_like(x_s, rng=rng)
        mean = x_s * alpha_ts
        var = sigma_t.pow(2) - alpha_ts.pow(2) * sigma_s.pow(2)
        x_t = mean + var.sqrt() * var_noise
        return x_t
