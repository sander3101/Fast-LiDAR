import math
from typing import List, Literal

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from . import base


def linear_beta_schedule(steps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)


def cosine_beta_schedule(steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(steps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_bar = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 0, 0.999)


class DiscreteTimeGaussianDiffusion(base.GaussianDiffusion):
    """
    Discrete-time Gaussian diffusion
    https://arxiv.org/abs/2006.11239
    """

    def setup_parameters(self) -> None:
        if self.beta_schedule == "linear":
            beta = linear_beta_schedule(self.num_training_steps)
        elif self.beta_schedule == "cosine":
            beta = cosine_beta_schedule(self.num_training_steps)
        elif self.beta_schedule == "sigmoid":
            beta = sigmoid_beta_schedule(self.num_training_steps)
        else:
            raise ValueError(f"invalid beta schedule {self.beta_schedule}")

        beta = beta[:, None, None, None]  # 4D-tensor for images
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (0,) * 6 + (1, 0), value=1.0)
        snr = alpha_bar / (1 - alpha_bar)

        clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            clipped_snr = clipped_snr.clamp(max=self.min_snr_gamma)
        if self.objective == "eps":
            loss_weight = clipped_snr / snr
        elif self.objective == "x0":
            loss_weight = clipped_snr
        elif self.objective == "v":
            loss_weight = clipped_snr / (snr + 1)
        else:
            raise ValueError(f"invalid objective {self.objective}")

        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("alpha_bar_prev", alpha_bar_prev.float())
        self.register_buffer("loss_weight", loss_weight.float())

    def get_target(self, x_0, steps, noise):
        if self.objective == "eps":
            return noise
        elif self.objective == "x0":
            return x_0
        elif self.objective == "v":
            alpha_bar = self.alpha_bar[steps]
            return alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * x_0
        else:
            raise ValueError(f"invalid objective {self.objective}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # discrete timesteps
        return torch.randint(
            low=0,
            high=self.num_training_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def get_denoiser_condition(self, steps: torch.Tensor) -> torch.Tensor:
        return steps

    @autocast(enabled=False)
    def q_sample(self, x0, steps, noise):
        alpha_bar = self.alpha_bar[steps]
        xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
        return xt

    def get_loss_weight(self, timesteps):
        loss_weight_t = self.loss_weight[timesteps, None]
        return loss_weight_t

    @torch.inference_mode()
    def p_sample(
        self,
        x_t: torch.Tensor,
        steps: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
        mode: Literal["ddpm", "ddim"] = "ddim",
        eta: float = 0.0,
    ):
        beta = self.beta[steps]
        alpha = 1 - beta
        alpha_bar = self.alpha_bar[steps]
        alpha_bar_prev = self.alpha_bar_prev[steps]
        prediction = self.denoiser(x_t, steps)
        if self.objective == "eps":
            eps = prediction
            x_0 = alpha_bar.rsqrt() * x_t - (alpha_bar.reciprocal() - 1).sqrt() * eps
            if self.clip_sample:
                noise = alpha_bar.rsqrt() * x_t - x_0
                noise = noise / (alpha_bar.reciprocal() - 1).sqrt()
        elif self.objective == "x0":
            x_0 = prediction
        elif self.objective == "v":
            v = prediction
            x_0 = alpha_bar.sqrt() * x_t - (1 - alpha_bar).sqrt() * v
        else:
            raise ValueError(f"invalid objective {self.objective}")
        if self.clip_sample:
            x_0.clamp_(-self.clip_sample_range, self.clip_sample_range)
        if mode == "ddpm":
            x_0_coef = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
            x_t_coef = (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar)
            mean = x_0_coef * x_0 + x_t_coef * x_t
            var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            var = var.clamp(min=1e-20)  # to avoid log(0)
            var_noise = self.randn_like(x_t, rng=rng)
            var_noise[steps == 0] *= 0
            x_s = mean + (0.5 * var.log()).exp() * var_noise
        elif mode == "ddim":
            var = (
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )
            std_dev = eta * torch.sqrt(var)
            eps = (x_t - alpha_bar.sqrt() * x_0) / (1 - alpha_bar).sqrt()  # glide
            x_s_dir = (1 - alpha_bar_prev - std_dev**2).sqrt() * eps
            x_s = alpha_bar_prev.sqrt() * x_0 + x_s_dir
            if eta > 0:
                var_noise = self.randn_like(x_t, rng=rng)
                var_noise[steps == 0] *= 0
                x_s = x_s + std_dev * var_noise
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
    ):
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        for timestep in tqdm(list(reversed(range(num_steps))), **tqdm_kwargs):
            timesteps = torch.full((batch_size,), timestep, device=self.device).long()
            x = self.p_sample(x, timesteps)
            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x

    @torch.inference_mode()
    def conditional_sample(
        self,
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mask: torch.Tensor = None,
        x_0: torch.Tensor = None,
    ):
        """
            Sampling method that takes in a conditonal mask and generates a sample with a condition for a given timestep
        """
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)

        ### Mask x to match sparse data ###
        if mask is not None:
            x = x * mask + (1 - mask) * x_0

        for timestep in tqdm(list(reversed(range(num_steps))), **tqdm_kwargs):
            timesteps = torch.full((batch_size,), timestep, device=self.device).long()

            x = self.p_sample(x, timesteps)

            ### Fill old are with original image and only keep generated part ###
            if not mask is None:
                x = x * mask + (1 - mask) * x_0

            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x
