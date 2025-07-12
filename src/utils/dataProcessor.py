from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from database import get_dataset

from conf.config import MyConfig

class DataProcessor:
    """
    データ処理クラス。
    前処理が必要な場合はこのクラスに実装してください。
    使わなくてもOK。
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def process_data(self):
        return None
    
def load_data(cfg: MyConfig, fold: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    データセットを読み込む関数。
    cfg.data.nameに応じてデータセットを取得し、DataLoaderに変換する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    fold : int
        交差検証のfold番号。
        交差検証を使わない場合は0。
    
    Returns
    -------
    train_loader : DataLoader
        学習用データローダー。
    valid_loader : DataLoader
        検証用データローダー。
    test_loader : DataLoader
        テスト用データローダー。
    
    See Also
    --------
    get_dataset : データセットを取得する関数。(database/__init__.py)
    dataset_to_dataloader : データセットをDataLoaderに変換する関数。(utils/training_utils.py)
    """
    print("loading data...")
    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, now_fold=fold)
    cfg.model.num_channels = train_dataset.num_channels
    cfg.model.input_size = train_dataset.input_size
    train_shape = print_get_dataset_details(cfg, train_dataset, valid_dataset, test_dataset)
    train_loader = dataset_to_dataloader(train_dataset, batch_size=cfg.model.batch_size)
    valid_loader = dataset_to_dataloader(valid_dataset, batch_size=cfg.model.test_batch_size, isValid=True)
    test_loader = dataset_to_dataloader(test_dataset, batch_size=cfg.model.test_batch_size, isTest=True)
    print("success to load data!\n\n")
    return train_loader, valid_loader, test_loader, train_shape

def print_get_dataset_details(cfg: MyConfig, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset) -> tuple:
    """
    データセットの詳細を表示，データのshapeを返却する関数。
    train_dataset, valid_dataset, test_datasetの各データセットの詳細（クラス分布 or サンプル数）を表示する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    train_dataset : Dataset
        学習用データセット。
    valid_dataset : Dataset
        検証用データセット。
    test_dataset : Dataset
        テスト用データセット。
    """
    if cfg.data.task_type == "classification":
        print("train_details:", *np.unique(train_dataset.y, return_counts=True), "/", len(train_dataset))
        if valid_dataset is not None:
            print("valid_details:", *np.unique(valid_dataset.y, return_counts=True), "/", len(valid_dataset))
        if test_dataset is not None:
            print("test_details:", *np.unique(test_dataset.y, return_counts=True), "/", len(test_dataset))
    else:
        print("train_details (length):", len(train_dataset))
        if valid_dataset is not None:
            print("valid_details (length):", len(valid_dataset))
        if test_dataset is not None:
            print("test_details (length):", len(test_dataset))
    train_shape = train_dataset[0][0].shape
    print("data_shape (except num_sample):", train_shape)
    return train_shape

def dataset_to_dataloader(dataset, batch_size, isValid=False, isTest=False):
    """
    データセットをDataLoaderに変換する関数。
    
    Parameters
    ----------
    dataset : Dataset
        データセット。
    batch_size : int
        バッチサイズ。
    isValid : bool, optional
        検証用データセットの場合はTrue, by default False
    isTest : bool, optional
        テスト用データセットの場合はTrue, by default False
    
    Returns
    -------
    DataLoader
        データローダー。
    
    Notes
    -----
    num_workersは2に設定している。
    注意：Windowsではnum_workers=0にしないと超絶学習が遅くなる。
    """
    if dataset is None:
        return None

    g = torch.Generator()
    g.manual_seed(42)
    if isValid or isTest:
        return DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            generator=g)
    else:
        return DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True,
                            generator=g)