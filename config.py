"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config = ed({
    "class_num": 2388,
    "batch_size": 64,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 80,
    "image_height": 299,
    "image_width": 299,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 3000,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./new_check",
    "warmup_epochs": 0,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0,
    "lr_max": 0.1

})
