import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from utils import AvgMeter, calc_loss_pred_y, calc_scores
from conf.config import MyConfig

class Trainer:
    """
    1エポック，モデルの学習を行うクラス．
    学習データに対して学習を行い，検証データに対して評価を行う．

    Parameters
    ----------
    device : torch.device
        使用するデバイス（GPU or CPU）．
    model : nn.Module
        学習するモデル．
    criterion : nn.Module
        損失関数．
    optimizer : optim.Optimizer
        最適化関数．
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    """
    def __init__(self, device:torch.device, model: nn.Module, criterion, optimizer: optim.Optimizer, cfg: MyConfig):
        self.__model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device

    @property
    def model(self):
        return self.__model

    def train_step(self, train_loader: DataLoader) -> tuple[float, list[any]]:
        """
        学習データに対し，1エポックの学習を行う関数．
        ミニバッチ処理を行い，損失と予測値を計算し，逆伝播．
        バッチ処理後に評価指標を計算．

        Parameters
        ----------
        train_loader : DataLoader
            学習用データローダー．
        
        Returns
        -------
        avg_loss : float
            1エポックの平均損失．
        metrics : list[any]
            評価指標のリスト．
            cfg.data.metricsに指定された評価指標を計算する．
        """
        # 1エポックの平均損失を計算する用
        avg_loss = AvgMeter()

        # 正解値と予測値を格納するリスト
        true_list, pred_list = [], []
        
        # --- バッチ処理 ---
        for train_X, train_y in tqdm(train_loader, desc="Epoch", leave=False, total=len(train_loader)):
            train_X = train_X.to(self.device)
            train_y = train_y.to(self.device)
            # --- 順伝播 ---
            output = self.__model(train_X)
            
            # --- 損失計算，予測 ---
            loss, pred_y = calc_loss_pred_y(self.cfg, output, train_y, self.criterion, self.device)
            
            # --- 逆伝播 ---
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # --- 各値の格納 ---
            avg_loss.update(loss.item())
            pred_list.extend(pred_y.tolist())
            true_list.extend(train_y.detach().to(torch.device("cpu")).numpy().tolist())

        pred_list = np.array(pred_list).flatten()
        true_list = np.array(true_list).flatten()

        # --- 評価指標の計算 ---
        metrics: list[float] = calc_scores(true_list, pred_list, self.cfg.data.metrics, calc_cm=False)
        
        return avg_loss.avg, metrics
    
    def valid_step(self, valid_loader: DataLoader) -> tuple[float, list[any]]:
        """
        検証データ（テストデータ）に対し，1エポックの検証を行う関数．
        ミニバッチ処理を行い，損失と予測値を計算．
        バッチ処理後に評価指標を計算．
        
        Parameters
        ----------
        valid_loader : DataLoader
            検証用（テスト用）データローダー．
        
        Returns
        -------
        avg_loss : float
            1エポックの平均損失．
        metrics : list[any]
            評価指標のリスト．
            cfg.data.metricsに指定された評価指標を計算する．
        """
        # 1エポックの平均損失を計算する用
        avg_loss = AvgMeter()

        # 正解値と予測値を格納するリスト
        true_list, pred_list = [], []

        # --- バッチ処理 ---        
        for test_X, test_y in tqdm(valid_loader, desc="Testing", leave=False, total=len(valid_loader)):
            test_X = test_X.to(self.device)
            test_y = test_y.to(self.device)
            # --- 順伝播 ---
            output = self.__model(test_X)
            
            # --- 損失計算，予測 ---
            with torch.no_grad():
                loss, pred_y = calc_loss_pred_y(self.cfg, output, test_y, self.criterion, self.device)            

            # --- 各値の格納 ---
            avg_loss.update(loss.item())
            pred_list.extend(pred_y.tolist())
            true_list.extend(test_y.detach().to(torch.device("cpu")).numpy().tolist())
        
        pred_list = np.array(pred_list).flatten()
        true_list = np.array(true_list).flatten()
        
        # --- 評価指標の計算 ---
        metrics: list[float] = calc_scores(true_list, pred_list, self.cfg.data.metrics, calc_cm=False)

        return avg_loss.avg, metrics