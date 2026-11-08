from typing import Dict
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from UniT.dataset.real_data_conversion import real_data_to_replay_buffer
from UniT.dataset.color_augmentation import ColorAugmentation

class TactileRepresentationDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            resized_image_shape=(128, 160),
            augmentation=None,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        print(dataset_path)
        replay_buffer = _get_replay_buffer(
            dataset_path=dataset_path,
        )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.val_mask = val_mask
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.resized_image_shape = resized_image_shape
        # Aumentado de color solo en la entrada: el objetivo ('target') se
        # mantiene limpio para que el modelo aprenda a mapear cualquier
        # color a una imagen táctil general.
        self.augmentation = (
            ColorAugmentation(**OmegaConf.to_container(augmentation, resolve=True))
            if augmentation is not None else None)
        # Dummy value
        self.horizon = 1
        self.n_obs_steps = 1

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.augmentation = None  # validación siempre sobre imágenes limpias
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        raw_img = data['tactile_image'][0]
        #print(raw_img.shape)
        img_float = raw_img.astype(np.float32) / 255.0
        #img_chw = np.transpose(img_float, (2,0,1))

        del data

        #print(img_chw.shape)
        if self.augmentation is None:
            return {'image': img_float}

        # entrada aumentada, objetivo limpio
        return {'image': self.augmentation(img_float),
                'target': img_float}

def _get_replay_buffer(dataset_path):
    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        print(dataset_path)
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
        )

    return replay_buffer


