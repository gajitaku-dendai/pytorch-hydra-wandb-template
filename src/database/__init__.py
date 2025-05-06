from conf.config import MyConfig
from database.diabetes import diabetes_dataset
from database.california_housing import california_housing_dataset
from database.mnist import mnist_dataset

import numpy as np

def get_dataset(cfg: MyConfig, now_fold):
    if cfg.data.name == "diabetes":
        train_dataset = diabetes_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = diabetes_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        test_dataset = diabetes_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    elif cfg.data.name == "california_housing":
        train_dataset = california_housing_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = california_housing_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        test_dataset = california_housing_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    elif cfg.data.name == "mnist":
        train_dataset = mnist_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = mnist_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        test_dataset = mnist_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    else:
        print(f'Dataset {cfg.data.name} is not defined !!!!')
