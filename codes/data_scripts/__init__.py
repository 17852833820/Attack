"""create dataset and dataloader"""
import logging
import torch


def create_dataset(mode, path, label):
    if mode == 'FD':
        from codes.data_scripts.FingerprintDataset import FingerprintDataset as f
    if mode == 'FD40':
        from codes.data_scripts.FingerprintDataset40 import FingerprintDataset40 as f

    dataset = f(path, label)
    return dataset
