import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from torch.utils.data import Dataset
import os
from abc import ABC, abstractmethod

from conf.config import MyConfig
import shutil

class BaseDataset(Dataset):
    """
    データセットの読み込み、前処理をする規定クラス。新たにデータセットを作る場合はこのクラスを継承する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    mode : str
        train, valid, testのいずれか。
    means : np.ndarray
        標準化のための平均値。trainの時はNone、valid/testの時はtrainで計算したものを渡す。
    stds : np.ndarray
        標準化のための標準偏差。trainの時はNone、valid/testの時はtrainで計算したものを渡す。
    now_fold : int
        k-fold cross validationの時に、今のfoldを指定する。(学習検証テストスプリットのときは0)
    pca : PCA
        PCAのインスタンス。trainの時はNone、valid/testの時はtrainで計算したものを渡す。

    Attributes
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    input_type : str
        model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る。処理分けするのに自由に使ってください。
    means : np.ndarray
        標準化のための平均値。trainの時はNone、valid/testの時はtrainで計算したものを渡す。
    stds : np.ndarray
        標準化のための標準偏差。trainの時はNone、valid/testの時はtrainで計算したものを渡す。
    pca : PCA
        PCAのインスタンス。trainの時はNone、valid/testの時はtrainで計算したものを渡す。
    now_fold : int
        k-fold cross validationの時に、今のfoldを指定する。(学習検証テストスプリットのときは0)
    X : np.ndarray
        データセットの特徴量。
    y : np.ndarray
        データセットのラベル。
    sample_num : int
        データセットの元のサンプル数。
    original_size : int or tuple
        データセットの元の次元数。
    length : int
        データセットの最終的なサンプル数。
    input_size : int or tuple
        データセットの最終的な次元数。
    num_channels : int
        データセットの最終的なチャンネル数。

    Notes
    -----
    元のサンプル数と最終的なサンプル数を取得してる理由は、ウィンドウ処理をしたくなったときに必要になるため。
    input_typeに応じて処理を自由に分岐させてください。（今回はimgとfeaturesのみ。）
    """
    def __init__(self, cfg: MyConfig, mode: str = "train",
                 means=None, stds=None, now_fold=None, pca: PCA=None):
        self.cfg = cfg
        self.input_type = cfg.model.input_type
        self.means = means
        self.stds = stds
        self.pca = pca
        self.now_fold = now_fold
        self.mode = mode # 現在のモード（train, valid, test, train_pre_splitなど）を保持

        # データセットの前処理を実行
        self._preprocess()

    def _preprocess(self):

        print(f"\nloading {self.mode} data")

        if not self.cfg.data.memmap:
            print("In-memory loading")
            self.X, self.y = self._load_data()
            self.sample_num, self.original_size = self._get_shape(self.X)
            self.X = self._pca(self.X, self.cfg.data.pca_n_components)
            self.means, self.stds = self._calc_means_stds(self.X, self.cfg.data.standardize_type)
            self.X = self._standardize(self.X, self.means, self.stds)
        else:
            print("Memmap loading")
            self.X, self.y = self._load_data_memmap()
            self.sample_num, self.original_size = self._get_shape(self.X)
            self.X = self._pca_memmap(self.X, self.cfg.data.pca_n_components)
            self.means, self.stds = self._calc_means_stds_memmap(self.X, self.cfg.data.standardize_type)
            self.X = self._standardize_memmap(self.X, self.means, self.stds)

        self.y = self._set_y_dtype(self.y)
        self.length = len(self.X)
        self.num_channels, self.input_size = self._get_num_ch_and_size(self.X)

    def _load_data(self):
        """
        データセットの読み込みを行う関数。
        k-fold cross validationの時は、foldごとに分けて読み込む。
        train/valid/testの時は、train/valid/testのデータを読み込む。

        Parameters
        ----------
        mode : str
            train, valid, testのいずれか。
        input_type : str
            model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る。処理分けするのに自由に使ってください。
        
        Returns
        -------
        X : np.ndarray
            データセットの特徴量。
        y : np.ndarray
            データセットのラベル。

        Notes
        -----
        k-fold cross validationの場合、学習とテストにのみ分ける場合が多い。
        例えば、k=5でfold 0をテストとするなら、fold 1-4を学習とする。
        しかし、これだとモデルの良し悪しの評価がテストに依存するため、あまり良くない。
        そこで、fold 1-4をランダムにtrainとvalidに分けて、validでモデルの良し悪しを評価することにする。
        """
        X, y = [], []
        base_path = f"src/database/{self.cfg.data.name}/{self.cfg.data.dir_name}"
        
        # k-fold cross validationの時は、foldごとに分けて読み込む。
        if self.cfg.data.use_kfold:
            for i in range(self.cfg.data.kfold_n_splits):
                if (self.mode == "test" and i != self.now_fold) or \
                   (self.mode != "test" and i == self.now_fold):
                    continue
                
                # ファイルが存在するか確認
                X_file_path = f"{base_path}/fold_{i}_X.npy"
                y_file_path = f"{base_path}/fold_{i}_y.npy"
                if not os.path.exists(X_file_path):
                    raise FileNotFoundError(f"Missing X data file for fold {i}: {X_file_path}")
                if not os.path.exists(y_file_path):
                    raise FileNotFoundError(f"Missing y data file for fold {i}: {y_file_path}")
                
                X.append(np.load(X_file_path).astype(np.float32))
                y.append(np.load(y_file_path))
            X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)

            if self.mode in ["train", "valid"]:
                if self.cfg.data.task_type == "classification":
                    # 分類問題の場合、クラス不均衡を考慮して、stratified samplingを行う
                    train_indices, valid_indices = train_test_split(np.arange(len(X)), stratify=y, test_size=self.cfg.data.kfold_valid_ratio, random_state=42)
                else:
                    train_indices, valid_indices = train_test_split(np.arange(len(X)), test_size=self.cfg.data.kfold_valid_ratio, random_state=42)

                if self.mode == "train":
                    X, y = X[train_indices], y[train_indices]
                elif self.mode == "valid":
                    X, y = X[valid_indices], y[valid_indices]

        else:
            # ファイルが存在するか確認
            X_file_path = f"{base_path}/{self.mode}_X.npy"
            y_file_path = f"{base_path}/{self.mode}_y.npy"
            if not os.path.exists(X_file_path):
                raise FileNotFoundError(f"Missing X data file: {X_file_path}")
            if not os.path.exists(y_file_path):
                raise FileNotFoundError(f"Missing y data file: {y_file_path}")

            X = np.load(X_file_path).astype(np.float32)
            y = np.load(y_file_path)
        return X, y

    def _load_data_memmap(self):
        """
        データセットの読み込みを行う関数。（メモリマップを使用）
        k-fold cross validationの時は、foldごとに分けて読み込む。
        train/valid/testの時は、train/valid/testのデータを読み込む。

        Parameters
        ----------
        mode : str
            train, valid, testのいずれか。
        input_type : str
            model/**.yamlで指定した"features" or "img" or "text"など入力形式が入る。処理分けするのに自由に使ってください。
        
        Returns
        -------
        X : np.ndarray
            データセットの特徴量。
        y : np.ndarray
            データセットのラベル。

        Notes
        -----
        k-fold cross validationの場合、学習とテストにのみ分ける場合が多い。
        例えば、k=5でfold 0をテストとするなら、fold 1-4を学習とする。
        しかし、これだとモデルの良し悪しの評価がテストに依存するため、あまり良くない。
        そこで、fold 1-4をランダムにtrainとvalidに分けて、validでモデルの良し悪しを評価することにする。
        """
        X, y = [], []
        base_path = f"src/database/{self.cfg.data.name}/{self.cfg.data.dir_name}"
        cache_path = f".cache"
        
        # X_train = np.memmap(f"{cache_path}/X_train.npy", dtype='float32', mode='w+', shape=(0,))  # 空のメモリマップ配列
        # k-fold cross validationの時は、foldごとに分けて読み込む。
        if self.cfg.data.use_kfold:
            len_data = 0
            for i in range(self.cfg.data.kfold_n_splits):
                if (self.mode == "test" and i != self.now_fold) or \
                   (self.mode != "test" and i == self.now_fold):
                    continue
                
                # ファイルが存在するか確認
                X_file_path = f"{base_path}/fold_{i}_X.npy"
                y_file_path = f"{base_path}/fold_{i}_y.npy"
                if not os.path.exists(X_file_path):
                    raise FileNotFoundError(f"Missing X data file for fold {i}: {X_file_path}")
                if not os.path.exists(y_file_path):
                    raise FileNotFoundError(f"Missing y data file for fold {i}: {y_file_path}")
                
                X.append(np.load(X_file_path, mmap_mode='r'))  # メモリマップで読み込み
                y.append(np.load(y_file_path))
                len_data += X[-1].shape[0]  # 各foldのデータ数をカウント
            y = np.concatenate(y, axis=0)

            # メモリマップ配列に書き込む
            data_shape = X[0].shape[1:]  # 特徴量の形状を取得
            X_tmp = np.memmap(f"{cache_path}/{self.mode}_X.mmap", dtype='float32', mode='w+', shape=(len_data, *data_shape))
            current_pos = 0  # memmapファイルへの書き込み開始位置を管理するカーソル
            batch_size = self.cfg.data.mmap_batch_size # 一度にメモリに乗せる行数（PCのメモリに合わせて調整）

            # 【外側のループ】リスト内の各ndarrayを順番に処理
            for x in X:
                # 【内側のループ】一つのndarray `x` を、さらに小さいバatchに分けて処理
                for j in range(0, x.shape[0], batch_size):
                    # 1. バッチを取り出す
                    batch_end = min(j + batch_size, x.shape[0])
                    batch = x[j:batch_end]

                    # 2. memmapファイルの正しい位置にバッチを書き込む
                    dest_start = current_pos + j
                    dest_end = current_pos + batch_end
                    X_tmp[dest_start:dest_end] = batch
                    
                # 一つのndarrayの処理が終わったら、次の配列の書き込み開始位置を更新
                current_pos += x.shape[0]

            # 全ての書き込みが終わったら、変更をディスクに保存
            X_tmp.flush()
            X = X_tmp

            if self.mode in ["train", "valid"]:
                if self.cfg.data.task_type == "classification":
                    # 分類問題の場合、クラス不均衡を考慮して、stratified samplingを行う
                    train_indices, valid_indices = train_test_split(np.arange(len(X)), stratify=y, test_size=self.cfg.data.kfold_valid_ratio, random_state=42)
                else:
                    train_indices, valid_indices = train_test_split(np.arange(len(X)), test_size=self.cfg.data.kfold_valid_ratio, random_state=42)

                if self.mode == "train":
                    indices = train_indices
                elif self.mode == "valid":
                    indices = valid_indices

                X_tmp = np.memmap(f"{cache_path}/{self.mode}_X_2.mmap", dtype='float32', mode='w+', shape=(len(indices), *data_shape))
                for j in range(0, len(indices), batch_size):
                    # 1. バッチを取り出す
                    batch_end = min(j + batch_size, len(indices))
                    batch_indices = indices[j:batch_end]

                    # 2. memmapファイルの正しい位置にバッチを書き込む
                    X_tmp[j:batch_end] = X[batch_indices]
                # 全ての書き込みが終わったら、変更をディスクに保存
                X_tmp.flush()
                X = X_tmp

                # 不要になった元のmemmapファイルを削除
                old_memmap_path = f"{cache_path}/{self.mode}_X.mmap"
                if os.path.exists(old_memmap_path):
                    try:
                        os.remove(old_memmap_path)
                    except Exception as e:
                        print(f"Failed to delete {old_memmap_path}. Reason: {e}")
                
                y = y[indices]  # yも同様にインデックスで絞り込む

        else:
            # ファイルが存在するか確認
            X_file_path = f"{base_path}/{self.mode}_X.npy"
            y_file_path = f"{base_path}/{self.mode}_y.npy"
            if not os.path.exists(X_file_path):
                raise FileNotFoundError(f"Missing X data file: {X_file_path}")
            if not os.path.exists(y_file_path):
                raise FileNotFoundError(f"Missing y data file: {y_file_path}")

            X = np.load(X_file_path, mmap_mode='r')  # メモリマップで読み込み

            # メモリマップ配列に書き込む
            X_tmp = np.memmap(f"{cache_path}/{self.mode}_X.mmap", dtype='float32', mode='w+', shape=X.shape)
            batch_size = self.cfg.data.mmap_batch_size # 一度にメモリに乗せる行数（PCのメモリに合わせて調整）

            # 【内側のループ】一つのndarray `x` を、さらに小さいバatchに分けて処理
            for j in range(0, X.shape[0], batch_size):
                # 1. バッチを取り出す
                batch_end = min(j + batch_size, X.shape[0])
                batch = X[j:batch_end]

                # 2. memmapファイルの正しい位置にバッチを書き込む
                X_tmp[j:batch_end] = batch

            # 全ての書き込みが終わったら、変更をディスクに保存
            X_tmp.flush()
            X = X_tmp

            y = np.load(y_file_path)

        return X, y
    
    def _calc_means_stds(self, X, axis_type):
        """
        標準化のための平均値と標準偏差を計算する関数。
        """
        if self.cfg.data.standardize_type is None or not self.mode == "train":
            return self.means, self.stds

        print("INFO: calculating means and stds... (in-memory mode)")

        if self.input_type == "features":
            axis = 0 if axis_type == "per_feature" else None
        elif self.input_type == "img":
            axis = (0, 2, 3) if axis_type == "per_channel" else None # img (B, C, H, W)
        means = np.nanmean(X, axis=axis, keepdims=True)
        stds = np.nanstd(X, axis=axis, keepdims=True)
        return means, stds

    def _calc_means_stds_memmap(self, X, axis_type):
        """
        標準化のための平均値と標準偏差をオンライン（バッチ）で計算する関数。
        """
        if self.cfg.data.standardize_type is None or not self.mode == "train":
            return self.means, self.stds
        
        print("INFO: calculating means and stds... (memmap mode)")

        if self.input_type == "features":
            # axis=0: 特徴量ごと (B, F) -> (F,)
            axis = 0 if axis_type == "per_feature" else None
        elif self.input_type == "img":
            # axis=(0, 2, 3): チャンネルごと (B, C, H, W) -> (C,)
            axis = (0, 2, 3) if axis_type == "per_channel" else None
        
        batch_size = self.cfg.data.mmap_batch_size
        n_samples = X.shape[0]

        # --- 変数の初期化 ---
        # axisが指定されている場合、合計値はスカラではなく配列になる
        if axis is not None:
            # keepdims=Trueで次元を維持したまま、合計を計算するための形状を取得
            # 例: (B, C, H, W)でaxis=(0, 2, 3)なら、合計のshapeは(1, C, 1, 1)になる
            sum_shape = list(X.shape)
            for ax in (axis if isinstance(axis, tuple) else [axis]):
                sum_shape[ax] = 1
            
            total_sum = np.zeros(sum_shape, dtype=np.float64)
            total_sq_sum = np.zeros(sum_shape, dtype=np.float64)
            total_count = np.zeros(sum_shape, dtype=np.int64)

        else: # axisがNoneの場合（全体統計）
            total_sum = 0.0
            total_sq_sum = 0.0
            total_count = 0

        # --- バッチ処理で合計値などを計算 ---
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X[start:end]
            
            # NaNsを無視して合計を計算
            total_sum += np.nansum(batch, axis=axis, keepdims=True if axis is not None else False)
            total_sq_sum += np.nansum(batch**2, axis=axis, keepdims=True if axis is not None else False)
            # NaNsでない要素の数をカウント
            total_count += np.sum(~np.isnan(batch), axis=axis, keepdims=True if axis is not None else False)

        mean = total_sum / total_count
        # 分散 = E[X^2] - (E[X])^2
        var = total_sq_sum / total_count - mean**2
        std = np.sqrt(var)

        return mean, std

    def _standardize(self, X, means, stds):
        if self.cfg.data.standardize_type is None:
            return X
        else:
            return (X - means) / np.where(stds == 0, 1e-8, stds)  # stdsが0の要素を避けるために小さい値を使用

    def _standardize_memmap(self, X, means, stds):
        batch_size = self.cfg.data.mmap_batch_size
        # 【内側のループ】一つのndarray `x` を、さらに小さいバatchに分けて処理
        for j in range(0, X.shape[0], batch_size):
            # 1. バッチを取り出す
            batch_end = min(j + batch_size, X.shape[0])
            batch = X[j:batch_end]

            # 2. memmapファイルの正しい位置にバッチを書き込む
            X[j:batch_end] = (batch - means) / np.where(stds == 0, 1e-8, stds)

        # 全ての書き込みが終わったら、変更をディスクに保存
        X.flush()
        return X

    def _pca(self, X, n_components):
        """
        PCAを行う関数。
        trainの時はpcaをfitして、valid/testの時はtrainのpcaでtransformする。
        """
        if not self.cfg.data.use_pca:
            return X
        print("INFO: starting PCA transformation... (in-memory mode)")
        # PCAオブジェクトがまだない場合は作成 (主にtrainモードの場合)
        if self.pca is None:
            pca = PCA(n_components=n_components, random_state=42) # random_stateを追加
        else: # valid or testモードの場合
            pca = self.pca

        if self.input_type == "features":
            if self.pca is None: # trainモードの場合のみfit
                pca.fit(X)
            X = pca.transform(X)
        elif self.input_type == "img":
            # 画像データを(Batch * H * W, Channel)の形にreshapeしてPCAを適用
            b, c, h, w = X.shape
            X_reshaped = X.transpose(0, 2, 3, 1).reshape(b * h * w, c)
            if self.pca is None: # trainモードの場合のみfit
                pca.fit(X_reshaped)
            X = pca.transform(X_reshaped)
            # PCA後のデータを元の画像形状に戻す (チャンネル数がn_componentsになる)
            X = X.reshape(b, h, w, -1).transpose(0, 3, 1, 2)

        print("pca_n_dim:", pca.n_components_)
        self.pca = pca # 更新されたPCAオブジェクトを保存
        return X

    def _pca_memmap(self, X, n_components):
        """
        PCAを行う関数。
        trainの時はpcaをfitして、valid/testの時はtrainのpcaでtransformする。
        """
        if not self.cfg.data.use_pca:
            return X
        print("INFO: starting PCA transformation... (memmap mode)")
        batch_size = self.cfg.data.mmap_batch_size
        # PCAオブジェクトがまだない場合は作成 (主にtrainモードの場合)
        if self.pca is None:
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

            # --- 学習フェーズ ---
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                batch = X[start:end]

                if self.input_type == "features":
                    pca.partial_fit(batch)
                elif self.input_type == "img":
                    b, c, h, w = batch.shape
                    batch = batch.transpose(0, 2, 3, 1).reshape(b * h * w, c)
                    pca.partial_fit(batch)
            self.pca = pca  # PCAオブジェクトを保存

        # --- 変換フェーズ ---
        # 変換後のデータを保存する、新しいmemmapファイルを作成
        if self.input_type == "features":
            output_shape = (X.shape[0], self.pca.n_components_)
        elif self.input_type == "img":
            output_shape = (X.shape[0], self.pca.n_components_, X.shape[2], X.shape[3])

        # 元のファイル名にサフィックスを付けて、新しいファイルパスを生成
        output_filepath = f".cache/{self.mode}_X.pca_transformed.mmap"
        X_transformed = np.memmap(output_filepath, dtype='float32', mode='w+', shape=output_shape)

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            batch = X[start:end]

            # バッチごとに変換
            if self.input_type == "features":
                transformed_batch = self.pca.transform(batch)
            elif self.input_type == "img":
                b, c, h, w = batch.shape
                reshaped_batch = batch.transpose(0, 2, 3, 1).reshape(b * h * w, c)
                transformed_reshaped = self.pca.transform(reshaped_batch)
                # 元の画像形式に戻す
                transformed_batch = transformed_reshaped.reshape(b, h, w, -1).transpose(0, 3, 1, 2)
            
            # 新しいmemmapファイルに書き込み
            X_transformed[start:end] = transformed_batch

        X_transformed.flush()
        print("pca_n_dim:", self.pca.n_components_)
        return X_transformed

    def _get_shape(self, X):
        """
        データセットの元の次元数を取得する関数。
        """
        if self.input_type == "img":
            return X.shape[0], (X.shape[2], X.shape[3])
        elif self.input_type == "features":
            return X.shape[0], X.shape[1]
        
    def _get_num_ch_and_size(self, X):
        """
        データセットの元のチャンネル数と次元数を取得する関数。
        """
        if self.input_type == "img":
            return X.shape[1], (X.shape[2], X.shape[3])
        elif self.input_type == "features":
            return 1, X.shape[1]
        
    def _set_y_dtype(self, y):
        """
        yの型を設定する関数。
        回帰問題の場合はfloat32、分類問題の場合はint64に変換する。
        """
        if self.cfg.data.task_type == "regression":
            return y.astype(np.float32)
        elif self.cfg.data.task_type == "classification":
            return y.astype(np.int64)
        else:
            raise ValueError(f"Unknown task type: {self.cfg.data.task_type}")

    def _getitem(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

    def __len__(self):
        return self.length