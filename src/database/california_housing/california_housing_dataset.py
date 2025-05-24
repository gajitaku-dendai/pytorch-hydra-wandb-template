import torch
import numpy as np
from sklearn.decomposition import PCA

from conf.config import MyConfig
from database.base_dataset import BaseDataset

class MyDataset(BaseDataset):
    """
    データセットの読み込み、前処理をするクラス。
    BaseDatasetを継承している。
    詳しくは、src/database/base_dataset.pyを参照。
    独自の処理を追加したい場合は、ここに書く。
    """
    def __init__(self, cfg: MyConfig, mode: str = "train",
                 means=None, stds=None, now_fold=None, pca: PCA=None):
        super().__init__(cfg, mode=mode, means=means, stds=stds, now_fold=now_fold, pca=pca)
    
class TrainDataset(MyDataset):
    """
    学習用データセットのクラス。
    MyDatasetを継承している。
    学習データにのみ適用したい処理があればここに書く。（データ拡張など）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    now_fold : int
        k-fold cross validationの時に、今のfoldを指定する。(学習検証テストスプリットのときはNone)
    pca : PCA
        PCAのインスタンス。基本はNone.
    
    Returns
    -------
    X : np.ndarray
        データセットの特徴量。
    y : np.ndarray
        データセットのラベル。
    
    Notes
    -----
    self.input_typeに応じてデータ拡張の関数を作ると良い。
    """
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
    """
    検証用データセットのクラス。
    MyDatasetを継承している。
    検証データにのみ適用したい処理があればここに書く。（基本はこのまま）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    means : np.ndarray
        標準化のための平均値。trainで計算したものを渡す。
    stds : np.ndarray
        標準化のための標準偏差。trainで計算したものを渡す。
    now_fold : int
        k-fold cross validationの時に、今のfoldを指定する。(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス。trainで計算したものを渡す。

    Returns
    -------
    X : np.ndarray
        データセットの特徴量。
    y : np.ndarray
        データセットのラベル。
    """
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="valid", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y

class TestDataset(MyDataset):
    """
    テスト用データセットのクラス。
    MyDatasetを継承している。
    テストデータにのみ適用したい処理があればここに書く。（基本はこのまま）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    means : np.ndarray
        標準化のための平均値。trainで計算したものを渡す。
    stds : np.ndarray
        標準化のための標準偏差。trainで計算したものを渡す。
    now_fold : int
        k-fold cross validationの時に、今のfoldを指定する。(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス。trainで計算したものを渡す。
        
    Returns
    -------
    X : np.ndarray
        データセットの特徴量。
    y : np.ndarray
        データセットのラベル。
    """
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="test", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y