from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from conf.config import MyConfig

class MyDataset(Dataset):
    """
    データセットの読み込み，前処理をするクラス．新たにデータセットを作る場合はこのクラスを参考にすると良い．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    mode : str
        train, valid, testのいずれか．
    means : np.ndarray
        標準化のための平均値．trainの時はNone，valid/testの時はtrainで計算したものを渡す．
    stds : np.ndarray
        標準化のための標準偏差．trainの時はNone，valid/testの時はtrainで計算したものを渡す．
    now_fold : int
        k-fold cross validationの時に，今のfoldを指定する．(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス．trainの時はNone，valid/testの時はtrainで計算したものを渡す．

    Attributes
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    input_type : str
        model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る．処理分けするのに自由に使ってください．
    means : np.ndarray
        標準化のための平均値．trainの時はNone，valid/testの時はtrainで計算したものを渡す．
    stds : np.ndarray
        標準化のための標準偏差．trainの時はNone，valid/testの時はtrainで計算したものを渡す．
    pca : PCA
        PCAのインスタンス．trainの時はNone，valid/testの時はtrainで計算したものを渡す．
    now_fold : int
        k-fold cross validationの時に，今のfoldを指定する．(学習検証テストスプリットのときは0)
    X : np.ndarray
        データセットの特徴量．
    y : np.ndarray
        データセットのラベル．
    sample_num : int
        データセットの元のサンプル数．
    original_size : int
        データセットの元の次元数．
    length : int
        データセットの最終的なサンプル数．
    input_size : int
        データセットの最終的な次元数．
    num_channels : int
        データセットの最終的なチャンネル数．

    Notes
    -----
    元のサンプル数と最終的なサンプル数を取得してる理由は，ウィンドウ処理をしたくなったときに必要になるため．
    input_typeに応じて処理を自由に分岐させてください．（今回はfeaturesのみ．）
    """
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
        if self.input_type == "features":
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
        
        print(f"\ndata_distribution ({mode})")
        print("max", "min", "mean", "std")
        print(torch.max(self.X), torch.min(self.X), torch.mean(self.X), torch.std(self.X))

    def _load_data(self, mode, input_type):
        """
        データセットの読み込みを行う関数．
        k-fold cross validationの時は，foldごとに分けて読み込む．
        train/valid/testの時は，0番のfoldを読み込む．

        Parameters
        ----------
        mode : str
            train, valid, testのいずれか．
        input_type : str
            model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る．処理分けするのに自由に使ってください．
        
        Returns
        -------
        X : np.ndarray
            データセットの特徴量．
        y : np.ndarray
            データセットのラベル．

        Notes
        -----
        k-fold cross validationの場合，学習とテストにのみ分ける場合が多い．
        例えば，k=5でfold 0をテストとするなら，fold 1-4を学習とする．
        しかし，これだとモデルの良し悪しの評価がテストに依存するため，あまり良くない．
        そこで，fold 1-4をランダムにtrainとvalidに分けて，validでモデルの良し悪しを評価することにする．
        学習:検証:テスト = 6:2:2の割合で分ける．
        """
        X, y = [], []
        for i in range(self.cfg.kfold_n_splits):
            if (mode == "test" and i != self.now_fold) or (mode != "test" and i == self.now_fold):
                continue
            base_path = f"src/database/california_housing/{input_type}/{self.cfg.data.dir_name}"
            X.append(np.load(f"{base_path}/fold_{i}_X.npy"))
            y.append(np.load(f"{base_path}/fold_{i}_y.npy"))
        X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
        if mode in ["train", "valid"]:
            if mode == "train":
                if self.cfg.data.task_type == "classification":
                    # 分類問題の場合，クラス不均衡を考慮して，stratified samplingを行う
                    indices, _ = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=42)
                else:
                    indices, _ = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
            elif mode == "valid":
                if self.cfg.data.task_type == "classification":
                    # 分類問題の場合，クラス不均衡を考慮して，stratified samplingを行う
                    _, indices = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=42)
                else:
                    _, indices = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
            X, y = X[indices], y[indices]
        return X, y
    
    def _calc_means_stds(self, X, input_type, axis_type):
        """
        標準化のための平均値と標準偏差を計算する関数．

        Parameters
        ----------
        X : np.ndarray
            データセットの特徴量．
        input_type : str
            model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る．処理分けするのに自由に使ってください．
        axis_type : str
            "per_feature" or "per_channel" or "all"など，平均値と標準偏差を計算する軸を指定する．
            "per_feature"は特徴量ごとに計算する．
            "per_channel"はチャンネルごとに計算する．
            "all"は全体で計算する．

        Returns
        -------
        means : np.ndarray
            標準化のための平均値．
        stds : np.ndarray
            標準化のための標準偏差．
        """
        if input_type == "features":
            axis = 0 if axis_type == "per_feature" else None
        means = np.nanmean(X, axis=axis, keepdims=True)
        stds = np.nanstd(X, axis=axis, keepdims=True)
        return means, stds
        
    def _pca(self, X, input_type, n_components):
        """
        PCAを行う関数．
        trainの時はpcaをfitして，valid/testの時はtrainのpcaでtransformする．

        Parameters
        ----------
        X : np.ndarray
            データセットの特徴量．
        input_type : str
            model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る．処理分けするのに自由に使ってください．
        n_components : int or float
            PCAの次元数．(floatの場合は割合)．

        Returns
        -------
        X : np.ndarray
            PCA後のデータセットの特徴量．
        """
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
    """
    学習用データセットのクラス．
    MyDatasetを継承している．
    学習データにのみ適用したい処理があればここに書く．（データ拡張など）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    now_fold : int
        k-fold cross validationの時に，今のfoldを指定する．(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス．基本はNone.
    
    Returns
    -------
    X : np.ndarray
        データセットの特徴量．
    y : np.ndarray
        データセットのラベル．
    
    Notes
    -----
    input_typeに応じてデータ拡張の関数を作ると良い．
    今回はfeaturesのみ．（ノイズを加えるデータ拡張．）
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
    検証用データセットのクラス．
    MyDatasetを継承している．
    検証データにのみ適用したい処理があればここに書く．（基本はこのまま）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    means : np.ndarray
        標準化のための平均値．trainで計算したものを渡す．
    stds : np.ndarray
        標準化のための標準偏差．trainで計算したものを渡す．
    now_fold : int
        k-fold cross validationの時に，今のfoldを指定する．(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス．trainで計算したものを渡す．

    Returns
    -------
    X : np.ndarray
        データセットの特徴量．
    y : np.ndarray
        データセットのラベル．
    """
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="valid", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y

class TestDataset(MyDataset):
    """
    テスト用データセットのクラス．
    MyDatasetを継承している．
    テストデータにのみ適用したい処理があればここに書く．（基本はこのまま）

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    means : np.ndarray
        標準化のための平均値．trainで計算したものを渡す．
    stds : np.ndarray
        標準化のための標準偏差．trainで計算したものを渡す．
    now_fold : int
        k-fold cross validationの時に，今のfoldを指定する．(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス．trainで計算したものを渡す．
        
    Returns
    -------
    X : np.ndarray
        データセットの特徴量．
    y : np.ndarray
        データセットのラベル．
    """
    def __init__(self, cfg: MyDataset, means=None, stds=None, now_fold=None, pca=None):
        super().__init__(cfg, mode="test", means=means, stds=stds, now_fold=now_fold, pca=pca)

    def __getitem__(self, idx):
        X, y = self._getitem(idx)
        return X, y