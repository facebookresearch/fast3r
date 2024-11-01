# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes  # noqa
from .arkitscenes_multiview import ARKitScenes_Multiview  # noqa
from .blendedmvs import BlendedMVS  # noqa
from .blendedmvs_multiview import BlendedMVS_Multiview  # noqa
from .co3d import Co3d  # noqa
from .co3d_multiview import Co3d_Multiview  # noqa: F401
from .habitat import Habitat  # noqa
from .habitat_multiview import Habitat_Multiview  # noqa
from .megadepth import MegaDepth  # noqa
from .megadepth_multiview import MegaDepth_Multiview  # noqa
from .scannetpp import ScanNetpp  # noqa
from .scannetpp_multiview import ScanNetpp_Multiview  # noqa
from .staticthings3d import StaticThings3D  # noqa
from .waymo import Waymo  # noqa
from .wildrgbd import WildRGBD  # noqa
from src.data.components.spann3r_datasets import *  # noqa

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader
