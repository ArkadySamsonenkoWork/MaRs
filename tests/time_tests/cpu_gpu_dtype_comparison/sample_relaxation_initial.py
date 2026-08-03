import sys
import os
import math
import random

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, root_dir)

import time
import torch
import typing as tp
from mars import spin_model, mesher, constants, population

basis_list = ["zfs", "zeeman", None, "product"]


def get_populations(N: int, disable_randomness: bool, device, dtype):
    if disable_randomness:
        pops = torch.arange(1, N + 1, device=device, dtype=dtype)
    else:
        pops = torch.rand(N, device=device, dtype=dtype)
    return pops / pops.sum()


def get_rates(N, disable_randomness: bool, device, dtype):
    if disable_randomness:
        rates = torch.triu(
            torch.arange(1, N * N + 1, device=device, dtype=dtype).view(N, N), diagonal=1
        )
    else:
        rates = torch.triu(
            torch.rand(N, N, device=device, dtype=dtype), diagonal=1
        )
    return rates + rates.T


def set_relaxation_and_initial_channels(sample, num_contexts: int = 0, disable_randomness: bool = False):
    device = sample.device
    dtype = sample.dtype
    N = sample.spin_system_dim

    contexts = []
    max_rate = torch.tensor(1e5, dtype=dtype, device=device)
    for _ in range(num_contexts):
        basis = random.choice(basis_list)

        pops = get_populations(N, disable_randomness, device, dtype)
        rates = get_rates(N, disable_randomness, device, dtype) * max_rate

        context = population.Context(
            sample=sample,
            basis=basis,
            init_populations=pops,
            thermal_rates=rates,
            device=device,
            dtype=dtype
        )
        contexts.append(context)

    return population.SummedContext(contexts)


def set_relaxation_and_initial_channels_batches(sample, num_contexts: int = 0, disable_randomness: bool = False):
    device = sample.device
    dtype = sample.dtype
    N = sample.spin_system_dim
    B = sample.config_shape[-2]

    contexts = []
    max_rate = torch.tensor(1e5, dtype=dtype, device=device)
    for _ in range(num_contexts):
        basis = random.choice(basis_list)

        pops = torch.rand(B, 1, 1, N, device=device, dtype=dtype)
        pops = pops / pops.sum(dim=-1, keepdim=True)

        rand_mat = torch.rand(B, 1, 1, N, N, device=device, dtype=dtype)
        upper = max_rate * torch.triu(rand_mat, diagonal=1)
        rates = upper + upper.transpose(-2, -1)

        context = population.Context(
            sample=sample,
            basis=basis,
            init_populations=pops,
            thermal_rates=rates,
            device=device,
            dtype=dtype
        )
        contexts.append(context)
    return population.SummedContext(contexts)
