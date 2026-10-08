"""Compute placement.

1. GPU (PyTorch) for large batched rollouts, e.g. the 303 sigma points of the
   151-parameter neural network, or all trials of an experiment stacked.
2. Multicore CPU (process pool) across independent runs (controllers,
   disturbance cases, BO trials).
3. Single-thread CPU for what is inherently sequential (time stepping, small
   scipy syntheses inside a worker).

``AK_DEVICE=cpu`` disables the GPU, ``AK_WORKERS=n`` caps the process pool.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Callable, Iterable, Sequence

import torch

DTYPE = torch.float64  # identical results on CPU and GPU; problems are launch-bound, not FLOP-bound
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def gpu_available() -> bool:
    return os.environ.get("AK_DEVICE", "").lower() != "cpu" and torch.cuda.is_available()


def device(large_batch: bool = True) -> torch.device:
    """CUDA for large batches when available, CPU otherwise."""
    return torch.device("cuda") if large_batch and gpu_available() else torch.device("cpu")


def n_workers() -> int:
    return max(1, int(os.environ.get("AK_WORKERS", os.cpu_count() or 1)))


def single_thread() -> None:
    """Pin BLAS/torch to one thread (pool workers would otherwise oversubscribe the cores)."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    torch.set_num_threads(1)


def parallel_map(fn: Callable, items: Sequence, workers: int | None = None) -> list:
    """Map ``fn`` over independent ``items`` on a spawn-based process pool.

    Spawn (not fork) keeps CUDA usable inside workers. ``fn`` must be a
    module-level function; falls back to a serial loop for one worker/item.
    """
    items = list(items)
    workers = min(n_workers() if workers is None else workers, len(items))
    if workers <= 1:
        return [fn(it) for it in items]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=single_thread) as ex:
        return list(ex.map(fn, items))


def describe() -> str:
    gpu = torch.cuda.get_device_name(0) if gpu_available() else "none"
    return f"GPU: {gpu} | CPU workers: {n_workers()}"


def to_tensor(x, dev: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE, device=dev)


def chunks(seq: Iterable, sizes: Sequence[int]) -> list:
    """Split a stacked array back into per-run pieces."""
    out, i = [], 0
    for n in sizes:
        out.append(seq[i:i + n])
        i += n
    return out
