from conf.config import MyConfig
from database.diabetes import diabetes_dataset
from database.california_housing import california_housing_dataset
from database.mnist import mnist_dataset

def get_dataset(cfg: MyConfig, now_fold):
    """
    database/**.dataset.pyをimportして、train_dataset, valid_dataset, test_datasetを返す関数

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    now_fold : int
        foldの番号
    
    Returns
    -------
    train_dataset : Dataset
        学習用データセット
    valid_dataset : Dataset
        検証用データセット
    test_dataset : Dataset
        テスト用データセット
    
    Notes
    -----
    データセットを追加した場合、ここにif文を追加する必要がある。
    """

    use_valid = (cfg.data.use_kfold and cfg.data.kfold_valid_ratio > 0) or \
                (not cfg.data.use_kfold and "valid" in cfg.data.data_splits)
    
    use_test = cfg.data.use_kfold or \
               (not cfg.data.use_kfold and "test" in cfg.data.data_splits)

    if cfg.data.name == "diabetes":
        train_dataset = diabetes_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = None
        test_dataset = None
        if use_valid:
            valid_dataset = diabetes_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        if use_test:
            test_dataset = diabetes_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    elif cfg.data.name == "california_housing":
        train_dataset = california_housing_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = None
        test_dataset = None
        if use_valid:
            valid_dataset = california_housing_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        if use_test:
            test_dataset = california_housing_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    elif cfg.data.name == "mnist":
        train_dataset = mnist_dataset.TrainDataset(cfg, now_fold=now_fold)
        valid_dataset = None
        test_dataset = None
        if use_valid:
            valid_dataset = mnist_dataset.ValidDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        if use_test:
            test_dataset = mnist_dataset.TestDataset(cfg, train_dataset.means, train_dataset.stds, now_fold=now_fold, pca=train_dataset.pca)
        return train_dataset, valid_dataset, test_dataset
    else:
        raise ValueError(f"Dataset {cfg.data.name} is not defined.")
