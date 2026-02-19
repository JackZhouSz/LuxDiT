import os
import functools
import logging
import sys
import imageio
import atexit
import importlib
import torch
import torchvision
import numpy as np
from termcolor import colored

from einops import rearrange
import mediapy as media

def instantiate_from_config(config, **additional_kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    
    # NOTE Kevin changed this to params from kwargs to align with av-vfm
    additional_kwargs.update(config.get("params", dict()))
    return get_obj_from_str(config["target"])(**additional_kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
