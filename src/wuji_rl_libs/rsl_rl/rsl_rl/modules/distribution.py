# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Distribution(nn.Module):
    """Base class for distribution modules.

    Distribution modules encapsulate the stochastic output of a neural model. They define the output structure expected
    from the MLP, manage learnable distribution parameters, and provide methods for sampling, log probability
    computation, and entropy calculation.

    Subclasses must implement all abstract methods and properties to define a specific distribution type.
    """

    def __init__(self, output_dim: int) -> None:
        """Initialize the distribution module.

        Args:
            output_dim: Dimension of the action/output space.
        """
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution parameters given the MLP output.

        Args:
            mlp_output: Raw output from the MLP.
        """
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Sample from the distribution.

        Returns:
            Sampled values.
        """
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the deterministic (mean) output from the raw MLP output.

        Args:
            mlp_output: Raw output from the MLP.

        Returns:
            The deterministic output (typically the distribution mean).
        """
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the deterministic output from the MLP output."""
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Return the input dimension required by the distribution."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation (or spread measure) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the distribution, summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the distribution parameters as a tuple of tensors.

        These are the distribution-specific parameters needed to reconstruct the distribution (e.g., mean and std for
        Gaussian, alpha and beta for Beta). They are stored during rollouts and used for KL divergence computation.
        """
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given outputs, summed over the last dimension.

        Args:
            outputs: Values to compute the log probability for.

        Returns:
            Log probability summed over the last dimension.
        """
        raise NotImplementedError

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the KL divergence KL(old || new) between two distributions of this type.

        The KL divergence measures how the old distribution diverges from the new distribution.
        This is used for adaptive learning rate scheduling in policy optimization.

        Args:
            old_params: Parameters of the old distribution (as returned by :attr:`params`).
            new_params: Parameters of the new distribution (as returned by :attr:`params`).

        Returns:
            KL divergence summed over the last dimension.
        """
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize distribution-specific weights in the MLP.

        This is called after MLP creation to set up any special weight initialization
        required by the distribution (e.g., initializing std head weights).

        Args:
            mlp: The MLP module whose weights may need initialization.
        """
        pass


class GaussianDistribution(Distribution):
    """Gaussian (Normal) distribution module with state-independent standard deviation.

    This distribution parameterizes actions using a multivariate Gaussian with diagonal covariance. The standard
    deviation is a learnable parameter that is independent of the model input. It can be parameterized in either
    "scalar" space (directly) or "log" space.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        super().__init__(output_dim)
        self.std_type = std_type

        # Learnable std parameters
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the mean from the MLP output."""
        return _IdentityDeterministicOutput()

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Gaussian distribution."""
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Gaussian distribution."""
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Gaussian distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current Gaussian distribution."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Gaussian, summed over the last dimension."""
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Gaussian distributions using torch.distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian (Normal) distribution module with state-dependent standard deviation.

    This distribution parameterizes actions using a multivariate Gaussian with diagonal covariance. The standard
    deviation is output by the MLP alongside the mean, making it state-dependent (heteroscedastic). It can be
    parameterized in either "scalar" space (directly) or "log" space.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (used to initialize MLP std head bias).
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that extracts the mean from the MLP output."""
        return _MeanSliceDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the mean and the second is the standard deviation (or log standard deviation).
        """
        return [2, self.output_dim]

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the std head weights in the MLP."""
        # Initialize weights and biases for the std portion of the last layer
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)  # type: ignore
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_std_log)  # type: ignore


class SoftplusGaussianDistribution(HeteroscedasticGaussianDistribution):
    """State-dependent std with softplus activation (Brax-style).

    MLP outputs ``[..., 2, output_dim]``. The first slice is the mean, the second
    is passed through ``softplus`` to produce a positive std:
    ``std = softplus(scale_raw) + min_std``.
    """

    def __init__(self, output_dim: int, init_std: float = 0.5, min_std: float = 0.01) -> None:
        # Reverse-compute bias so that softplus(bias) + min_std = init_std
        target_softplus = max(init_std - min_std, 1e-6)
        init_bias = math.log(math.exp(target_softplus) - 1.0)
        super().__init__(output_dim, init_bias, std_type="scalar")
        self._min_std = min_std

    def update(self, mlp_output: torch.Tensor) -> None:
        mean, scale_raw = torch.unbind(mlp_output, dim=-2)
        std = F.softplus(scale_raw) + self._min_std
        self._distribution = Normal(mean, std)


class ColoredNoiseGaussianDistribution(GaussianDistribution):
    """Gaussian policy whose EXPLORATION noise is temporally CORRELATED (colored) rather than white.

    The rollout action is ``a_t = mean + std * eps_t`` where ``eps_t`` is drawn from a per-env colored-noise
    sequence with power spectrum ``S(f) ~ 1/f^beta`` (beta=0 white/uncorrelated, 0.5 recommended for PPO,
    1.0 pink). Because each ``eps_t`` is normalized to be marginally UNIT-VARIANCE Gaussian, the per-step
    policy remains exactly ``N(mean, std)``: log_prob / entropy / kl / params / deterministic_output are all
    inherited from GaussianDistribution UNCHANGED, so the PPO objective is computed the ordinary way and
    stays "asymptotically on-policy" ( Eberhard-style colored action noise for PPO, arXiv 2312.11091). The
    ONLY thing colored is the cross-step correlation of the sampling.

    Only ``sample()`` (the stochastic rollout path) is colored; deterministic evaluation (the mean) is
    untouched, so paired baseline/deterministic eval is unaffected.

    Noise is generated per-env in chunks of length ``horizon`` and advanced ONE step per ``sample()`` call.
    ``sample()`` fires exactly once per env-step during rollout collection (the PPO update path uses
    log_prob, not sample), so with the Clean structure's SYNCHRONOUS episodes (all envs reset together every
    ``rl_steps``; pass ``horizon=rl_steps``) the chunk boundary coincides with the episode boundary and the
    chunk-local index equals the per-env episode step. If that alignment were ever broken it would only
    soften the correlation reset at episode edges — it can NOT bias the gradient, since the marginal is
    always unit-variance Gaussian regardless of alignment.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        beta: float = 0.5,
        horizon: int = 256,
    ) -> None:
        """Initialize the colored-noise Gaussian distribution.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (same learnable std as GaussianDistribution).
            std_type: Parameterization of the standard deviation: "scalar" or "log".
            beta: Colored-noise exponent (S(f) ~ 1/f^beta). 0 = white, 0.5 = PPO default, 1 = pink.
            horizon: Length of each per-env colored-noise chunk; pass rl_steps to align chunks to episodes.
        """
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.beta = float(beta)
        self.horizon = int(horizon)
        self._noise: torch.Tensor | None = None  # (n_env, horizon, output_dim), lazily built on first sample()
        self._t = 0

    def _regenerate_noise(self, n_env: int, device: torch.device, dtype: torch.dtype) -> None:
        """Draw fresh unit-variance colored-noise chunks for every env (Timmer & Koenig 1995, via rFFT)."""
        T = self.horizon
        freqs = torch.fft.rfftfreq(T, device=device)  # (T//2+1,), freqs[0] == 0 (DC)
        scale = torch.zeros_like(freqs)
        scale[1:] = freqs[1:] ** (-self.beta / 2.0)  # spectral amplitude ~ f^(-beta/2); DC set to 0 (zero mean)
        # Independent complex Gaussian spectrum per (env, action_dim), colored by `scale`.
        real = torch.randn(n_env, self.output_dim, freqs.numel(), device=device)
        imag = torch.randn(n_env, self.output_dim, freqs.numel(), device=device)
        spectrum = torch.complex(real, imag) * scale
        x = torch.fft.irfft(spectrum, n=T, dim=-1)  # (n_env, output_dim, T) real colored sequence
        # Normalize each sequence to EXACT unit empirical std so std*eps has the intended variance (log_prob-correct).
        x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-8)
        self._noise = x.permute(0, 2, 1).to(dtype)  # (n_env, horizon, output_dim)
        self._t = 0

    def sample(self) -> torch.Tensor:
        """Sample an action using colored (temporally correlated) exploration noise."""
        mean = self._distribution.mean  # type: ignore
        std = self._distribution.stddev  # type: ignore
        n_env = mean.shape[0]
        if self._noise is None or self._noise.shape[0] != n_env or self._t >= self.horizon:
            self._regenerate_noise(n_env, mean.device, mean.dtype)
        eps = self._noise[:, self._t, :]  # type: ignore  # (n_env, output_dim), marginally unit-variance Gaussian
        self._t += 1
        return mean + std * eps


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _MeanSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]
