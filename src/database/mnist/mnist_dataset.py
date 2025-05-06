from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from conf.config import MyConfig

class MyDataset(Dataset):
    def __init__(self, cfg: MyConfig, mode: str = "train",
                 means=None, stds=None, now_fold=None, pca: PCA=None):
        self.cfg = cfg
        self.input_type = cfg.model.input_type
        self.means = 0
        self.stds = 0

        print(f"\nloading {mode} data")

        self.pca = pca
        self.now_fold = now_fold

        self.X, self.y = self._load_data(mode, self.input_type)
        self.sample_num = self.X.shape[0]
        self.original_size = self.X.shape[1]

        ### 好きな処理をしてください ###
        #例 PCA#
        if cfg.data.use_pca:
            print("\npca")
            self.X = self._pca(self.X, self.input_type, cfg.data.pca_n_components)

        #例 標準化#
        if cfg.data.standardize_type != "none":
            print("\nstandardization")
            if mode == "train":
                self.means, self.stds = self._calc_means_stds(self.X, self.input_type, cfg.data.standardize_type)
            else:
                self.means, self.stds = means, stds
            self.X = (self.X - self.means) / self.stds


        if cfg.model.output_size == 1 or cfg.data.task_type == "regression":
            self.y = self.y.astype(np.float32)
        elif cfg.data.task_type == "classification":
            self.y = self.y.astype(np.int64)
        self.length = len(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
        if self.input_type == "features":
            self.input_size = self.X.shape[1]
            self.num_channels = 1
        elif self.input_type == "img":
            self.input_size = (self.X.shape[2], self.X.shape[3])
            self.num_channels = self.X.shape[1]
        
        print("\ndata_distribution")
        print("max", "min", "mean", "std")
        print(torch.max(self.X), torch.min(self.X), torch.mean(self.X), torch.std(self.X))

    def _load_data(self, mode, input_type):
        X, y = [], []
        for i in range(self.cfg.kfold_n_splits):
            if (mode == "test" and i != self.now_fold) or (mode != "test" and i == self.now_fold):
                continue
            base_path = f"src/database/mnist/{input_type}/{self.cfg.data.dir_name}"
            X.append(np.load(f"{base_path}/fold_{i}_X.npy"))
            y.append(np.load(f"{base_path}/fold_{i}_y.npy"))
        X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
        if mode in ["train", "valid"]:
            if mode == "train":
                if self.cfg.data.task_type == "classification":
                    indices, _ = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=42)
                else:
                    indices, _ = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
            elif mode == "valid":
                if self.cfg.data.task_type == "classification":
                    _, indices = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=42)
                else:
                    _, indices = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
            X, y = X[indices], y[indices]
        return X, y
    
    def _calc_means_stds(self, X, input_type, axis_type):
        if input_type == "features":
            axis = 0 if axis_type == "per_feature" else None
        elif input_type == "img":
            axis = (0, 2, 3) if axis_type == "per_channel" else None
        means = np.nanmean(X, axis=axis, keepdims=True)
        stds = np.nanstd(X, axis=axis, keepdims=True)
        return means, stds
        
    def _pca(self, X, input_type, n_components):
        if self.pca is None: # train
            pca = PCA(n_components=n_components)
        else: # valid or test
            pca = self.pca
        if input_type == "features":
            if self.pca is None: # train
                pca.fit(X)
            X = pca.transform(X)
        print("pca_n_dim:", pca.n_components_)
        self.pca = pca
        return X

    def _getitem(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.length
    
class TrainDataset(MyDataset):
    def __init__(self, cfg: MyConfig, now_fold=None, pca=None):
        super().__init__(cfg, mode="train", now_fold=now_fold,  pca=pca)
        self.rng = np.random.default_rng(42)

    def _dataAug_for_features(self, X):
        if self.cfg.model.use_noise:
            noise = self.rng.normal(0, 0.1, X.shape)
            X = (X + noise).to(torch.float32)
        return X

    def __getitem__(self, idx):
        X, y = self._getitem(idx)

        if self.input_type == "features":
            X = self._dataAug_for_features(X)
        return X, y

class ValidDataset(MyDataset):
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="valid", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y

class TestDataset(MyDataset):
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="test", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y