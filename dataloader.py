from src.utils.util import instantiate_from_config
import os
import torch
from itertools import cycle
import random
import time
from webdataset.utils import make_seed, pytorch_worker_seed

class MultiLoaderIterator:
    def __init__(
        self, 
        dataloaders, 
        mode='sequential', 
        rank=None,
    ):
        self.dataloaders = [d[0] for d in dataloaders]
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.dataset_idx_list = list(range(len(self.iterators)))
        self.mode = mode
        self.loader_cycle = cycle(self.dataset_idx_list)
        self.rank=rank
        self.rng = None # random number generator
        if mode == 'random':
            self.weights = [d[1] for d in dataloaders]
            print ('Datasets sampling weights: ', self.weights)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.mode == 'sequential':
            loader_idx = next(self.loader_cycle)
        elif self.mode == 'random':
            if self.rng is None:
                seed = make_seed(pytorch_worker_seed(), time.time_ns(), os.getpid(), os.urandom(4))
                self.rng = random.Random(seed)
            loader_idx = self.rng.choices(self.dataset_idx_list, weights=self.weights, k=1)[0]
        else:
            raise ValueError("Mode must be either 'sequential' or 'random'")
        
        try:
            batch = next(self.iterators[loader_idx])
        except StopIteration:
            self.iterators[loader_idx] = iter(self.dataloaders[loader_idx])
            batch = next(self.iterators[loader_idx])
            
        # print(f"Rank: {self.rank}, Selected Dataset Index: {loader_index}, clip ids: {batch['clip_name']}")
        return batch

def create_dataloader_iterators(train_dataloaders, validation_dataloaders, mode, rank):
    train_data_iter = MultiLoaderIterator(train_dataloaders, mode, rank)
    val_data_iter = MultiLoaderIterator(validation_dataloaders, 'sequential', rank) # always samples sequentially during validation to ensure coverage
    return train_data_iter, val_data_iter

def create_dataloader(train_data, logger, global_rank, global_seed, validation_data=None):
    logger.info(f"Building training datasets")
    
    train_dataloaders = []
    for train_data_cfg in train_data['datasets']:
        assert "target" in train_data_cfg
        if 'LightningLoader' in train_data_cfg['target']:
            train_data_instance = instantiate_from_config(train_data_cfg)
            train_dataloaders.append((train_data_instance.train_dataloader(), train_data_instance._dataset.sampling_weight))
        else:
            raise NotImplementedError(f"Dataset {train_data_cfg['target']} does not support LightningLoader")

    validation_dataloaders = []
    if validation_data is not None:
        logger.info(f"Building validation datasets")
        for validation_data_cfg in validation_data['datasets']:
            assert "target" in validation_data_cfg
            validation_data_instance = instantiate_from_config(validation_data_cfg)
            validation_dataloaders.append((validation_data_instance.train_dataloader(), validation_data_instance._dataset.sampling_weight))
    else:
        validation_dataloaders = train_dataloaders

    return train_dataloaders, validation_dataloaders
