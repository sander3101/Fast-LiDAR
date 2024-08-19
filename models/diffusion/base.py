from typing import List, Literal

import einops
import torch
from torch import nn
from torch.cuda.amp import autocast


class GaussianDiffusion(nn.Module):
    """
    Base class for continuous/discrete Gaussian diffusion models
    """

    def __init__(
        self,
        denoiser: nn.Module,
        sampling: Literal["ddpm", "ddim"] = "ddpm",
        criterion: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        num_training_steps: int = 1000,
        objective: Literal["eps", "v", "x0"] = "eps",
        beta_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.sampling = sampling
        self.num_training_steps = num_training_steps
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if criterion == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif criterion == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif criterion == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif isinstance(criterion, nn.Module):
            self.criterion = criterion
        else:
            raise ValueError(f"invalid criterion: {criterion}")
        if hasattr(self.criterion, "reduction"):
            assert self.criterion.reduction == "none"

        if sampling_resolution is None:
            assert hasattr(self.denoiser, "resolution")
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *self.denoiser.resolution)
        else:
            assert len(sampling_resolution) == 2
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *sampling_resolution)

        self.setup_parameters()
        self.register_buffer("_dummy", torch.tensor([]))

    @property
    def device(self):
        return self._dummy.device

    def randn(
        self,
        *shape,
        rng: List[torch.Generator] | torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
            )
        else:
            raise ValueError(f"invalid rng: {rng}")

    def randn_like(
        self,
        x: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        raise NotImplementedError

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def p_sample(self, *args, **kwargs):
        raise NotImplementedError

    @autocast(enabled=False)
    def q_sample(self, x_0, steps, noise):
        raise NotImplementedError

    def get_denoiser_condition(self, steps: torch.Tensor):
        raise NotImplementedError

    def get_target(self, x_0, steps, noise):
        raise NotImplementedError

    def get_loss_weight(self, steps):
        raise NotImplementedError

    def p_loss(
        self,
        x_0: torch.Tensor,
        steps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
            Calculates the loss during training
        """
        mask = torch.ones_like(x_0) if mask is None else mask
        noise = self.randn_like(x_0)
        xt = self.q_sample(x_0, steps, noise)

        # Conditional masking as explained in thesis
        xt = xt * mask.float() + (1 - mask.float()) * x_0

        condition = self.get_denoiser_condition(steps)
        prediction = self.denoiser(xt, condition)

        target = self.get_target(x_0, steps, noise)
        loss = self.criterion(prediction, target)  # (B,C,H,W)
        # Loss is limited to only be calculated for masked area
        loss = einops.reduce(loss * mask, "B ... -> B ()", "sum")
        mask = einops.reduce(mask, "B ... -> B ()", "sum")
        # To get consistent proper progress the loss is calculated from average over only masked pixels.
        loss = loss / mask.add(1e-8)  # (B,)
        loss = (loss * self.get_loss_weight(steps)).mean()
        return loss

    def forward(
        self, x_0: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
            Forward method
        """
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        loss = self.p_loss(x_0, steps, mask)
        return loss

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        raise NotImplementedError
