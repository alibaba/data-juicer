# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import torch
import torch.distributed as dist

_sync_device = None

# ----------------------------------------------------------------------------


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


# ----------------------------------------------------------------------------


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


# ----------------------------------------------------------------------------


def init(temp_dir: str):
    global _sync_device

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(((os.getpid() or 0) % 16384) + 29500)
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    torch.cuda.set_device(get_local_rank())
    torch.multiprocessing.set_start_method("spawn")

    if os.name == "nt":
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        init_method = "file:///" + init_file.replace("\\", "/")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    else:
        dist.init_process_group(backend="nccl", init_method="env://")

    _sync_device = torch.device("cuda") if get_world_size() > 1 else None
