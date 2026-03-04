from collections.abc import Sequence
import functools
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def to_jax_norm_stats(norm_stats: dict[str, _transforms.NormStats] | None, *, target_dim: int | None = None,) -> dict[str, dict[str, jnp.ndarray]] | None:
    """Convert norm stats to JAX arrays and optionally pad to a fixed last dim.

    target_dim should be the model action/state dim (constant) so padding is static for jit.
    """
    if norm_stats is None:
        return None

    def _pad(arr, value: float = 0.0) -> jnp.ndarray:
        if target_dim is None or arr.shape[-1] >= target_dim:
            return jnp.asarray(arr)
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target_dim - arr.shape[-1])
        return jnp.pad(jnp.asarray(arr), pad_width, mode="constant", constant_values=value)

    def _convert(stats: _transforms.NormStats) -> dict[str, jnp.ndarray]:
        return {
            "mean": _pad(stats.mean, value=0.0),
            "std": _pad(stats.std, value=1.0),
            # Pad q01 with -1.0 and q99 with 1.0 so that quantile unnorm acts as identity
            # on padded dims: (x + 1) / 2 * (1 - (-1)) + (-1) = x
            "q01": None if stats.q01 is None else _pad(stats.q01, value=-1.0),
            "q99": None if stats.q99 is None else _pad(stats.q99, value=1.0),
        }

    return {k: _convert(v) for k, v in norm_stats.items()}


def batch_list_of_dicts(samples):
    """Stack a list of observation dicts into a batched dict."""
    return jax.device_put(jax.tree.map(lambda *xs: np.stack(xs, axis=0), *samples))


@functools.partial(jax.jit, static_argnames=("use_quantiles",))
def unnormalize_outputs(outputs, norm_stats, *, use_quantiles: bool = False):
    """JIT-compiled output unnormalization."""
    def _unnorm(x, s):
        return x * (s["std"] + 1e-6) + s["mean"]

    def _unnorm_q(x, s):
        q01, q99 = s["q01"], s["q99"]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    fn = _unnorm_q if use_quantiles else _unnorm
    return jax.tree.map(fn, outputs, norm_stats)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        batch_output_transforms: Sequence[_transforms.DataTransformFn] = (),
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantile_norm: bool = False,
        sharding: jax.sharding.Sharding | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._batch_output_transforms = _transforms.compose(batch_output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._sharding = sharding
        self._data_sharding = None
        self._kv_cache_sharding = None
        self.action_horizon = model.action_horizon
        self.action_dim = model.action_dim
        self.use_quantile_norm = use_quantile_norm
        self.norm_stats = None
        if hasattr(model, "get_prefix_rep"):
            self._get_prefix_rep = nnx_utils.module_jit(
                model.get_prefix_rep, static_argnames=("return_kv_cache",)
            )

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            self.action_dim = model.action_dim
            self.norm_stats = to_jax_norm_stats(norm_stats, target_dim=self.action_dim)
            # Sharding for data-parallel batched inference.
            # _sharding uses PartitionSpec() (replicated) for params/rng.
            # _data_sharding uses PartitionSpec("x") to shard the batch dim across devices.
            if sharding is not None and not is_pytorch:
                self._data_sharding = jax.sharding.NamedSharding(
                    sharding.mesh, jax.sharding.PartitionSpec("x")
                )
                # kv_cache shape: (18, batch, 968, 1, 256) – batch is axis 1
                self._kv_cache_sharding = jax.sharding.NamedSharding(
                    sharding.mesh, jax.sharding.PartitionSpec(None, "x")
                )
                # Replicate norm_stats across mesh devices
                if self.norm_stats is not None:
                    self.norm_stats = jax.device_put(self.norm_stats, sharding)
            else:
                self._data_sharding = None
                self._kv_cache_sharding = None
            # JAX model setup
            # Set static args so regardless of num_steps in sample_kwargs, speed is consistent.
            self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, kv_cache: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Handle batched or single observation
            if inputs["state"].ndim > 1:
                batch_size = inputs["state"].shape[0]
                def _add_batch_dim(x):
                    return jnp.broadcast_to(
                        x[jnp.newaxis, ...],
                        (batch_size,) + x.shape
                    )

                inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
                for key in inputs:
                    if key not in ["image", "state"]:
                        inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
            else:
                batch_size = 1
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if kv_cache is not None:
            if self._is_pytorch_model:
                raise NotImplementedError(
                    "PyTorch infer does not support kv_cache yet."
                )
            kv_cache = jnp.asarray(kv_cache)
            sample_kwargs["kv_cache"] = kv_cache

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        elif batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _prepare_prefix_inputs(self, obs: dict) -> _model.Observation:
        """Shared preprocessing for prefix rep methods: copy, transform, batch."""
        assert not self._is_pytorch_model, "Prefix representations are only supported for JAX models."
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "image" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs)

    @override
    def get_prefix_rep(self, obs: dict) -> dict:  # type: ignore[misc]
        observation = self._prepare_prefix_inputs(obs)
        output = self._get_prefix_rep(observation, return_kv_cache=False)
        output = jax.tree.map(lambda x: np.asarray(x).astype(np.float32), output)
        return {
            "prefix_rep": output,
        }

    @override
    def get_prefix_rep_and_kv_cache(self, obs: dict) -> dict:
        observation = self._prepare_prefix_inputs(obs)
        prefix_rep, kv_cache = self._get_prefix_rep(observation, return_kv_cache=True)
        return {
            "prefix_rep": prefix_rep,
            "kv_cache": kv_cache,
        }

    @override
    def get_prefix_rep_and_kv_cache_batch(self, obs_batch: list[dict]) -> dict:
        assert not self._is_pytorch_model, "Prefix representations are only supported for JAX models."
        inputs = jax.tree.map(lambda x: x, obs_batch)
        inputs = [self._input_transform(i) for i in inputs]
        batched_inputs = batch_list_of_dicts(inputs)
        observation = _model.Observation.from_dict(batched_inputs)
        prefix_rep, kv_cache = self._get_prefix_rep(observation, return_kv_cache=True)
        return {
            "prefix_rep": prefix_rep,
            "kv_cache": kv_cache,
        }

    def infer_batch(
        self, 
        obs_batch: list[dict], 
        *, 
        noise: np.ndarray | None = None,
        kv_cache: np.ndarray | None = None,
    ) -> dict:
        """
        Perform batched inference on multiple observations.
        If model is sharded on multiple devices, the output of this function will be sharded accordingly.
        
        This is significantly faster than calling infer() in a loop when
        processing multiple observations (e.g., from parallel environments).
        
        Args:
            obs_batch: List of observation dicts (one per sample)
            noise: Optional noise array for diffusion (batch_size, action_horizon, action_dim)
        
        Returns:
            dict with:
                - "state": (batch_size, state_dim)
                - "actions": (batch_size, action_horizon, action_dim)  
                - "policy_timing": timing info in milliseconds
        """
        # Transform each observation
        inputs = jax.tree.map(lambda x: x, obs_batch)
        inputs = [self._input_transform(i) for i in inputs]
        if self._is_pytorch_model:
            raise NotImplementedError("PyTorch infer_batch is not supported.")
        else:
            # Stack into batched arrays
            batched_inputs = batch_list_of_dicts(inputs)

            # Shard inputs across devices for data parallelism
            if self._data_sharding is not None:
                batched_inputs = jax.device_put(batched_inputs, self._data_sharding)

            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

            # Replicate rng across mesh devices so every shard sees the same key
            if self._sharding is not None:
                sample_rng_or_pytorch_device = jax.device_put(sample_rng_or_pytorch_device, self._sharding)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            if self._data_sharding is not None:
                noise = jax.device_put(noise, self._data_sharding)
            sample_kwargs["noise"] = noise

        kv_cache_start_time = time.monotonic()
        if kv_cache is not None:
            if self._kv_cache_sharding is not None:
                kv_cache = [jax.device_put(jnp.asarray(c), self._kv_cache_sharding) for c in kv_cache]
            else:
                kv_cache = [jnp.asarray(c) for c in kv_cache]
            sample_kwargs["kv_cache"] = kv_cache
        kv_cache_time = time.monotonic() - kv_cache_start_time

        observation = _model.Observation.from_dict(batched_inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": batched_inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self.norm_stats is not None:
            outputs = unnormalize_outputs(outputs, self.norm_stats, use_quantiles=self.use_quantile_norm)
        outputs = self._batch_output_transforms(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "kv_cache_ms": kv_cache_time * 1000,
        }
        outputs['normalized_actions'] = actions
        return outputs

    def seed(self, seed: int) -> None:
        self._rng = jax.random.key(seed)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
