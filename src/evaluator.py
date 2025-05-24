import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import AvgMeter, calc_loss_pred_y, calc_scores
from conf.config import MyConfig

class Evaluator:
    """
    学習済みモデルの評価を行うクラス。
    与えられたデータローダー（学習 or 検証 or テスト）に対して評価を行い、損失と評価指標を計算する。

    Parameters
    ----------
    device : torch.device
        使用するデバイス（GPU or CPU）。
    model : nn.Module
        学習済みモデル。
    criterion : nn.Module
        損失関数。
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    """
    def __init__(self, device:torch.device, model: nn.Module, criterion, cfg: MyConfig):
        self.__model = model
        self.criterion = criterion
        self.cfg = cfg
        self.device = device

    @property
    def model(self):
        return self.__model
    
    def test_step(self, test_loader: DataLoader) -> tuple[float, list[any]]:
        """
        与えられたデータローダー（学習 or 検証 or テスト）に対して評価を行い、損失と評価指標を計算する関数。

        Parameters
        ----------
        test_loader : DataLoader
            評価用データローダー。

        Returns
        -------
        avg_loss : float
            平均損失。
        metrics : list[any]
            評価指標のリスト。
            cfg.data.metricsに指定された評価指標を計算する。
        """
        # 1エポックの平均損失を計算する用
        avg_loss = AvgMeter()

        # 正解値と予測値を格納するリスト
        true_list, pred_list = [], []
        
        # --- バッチ処理 ---
        for test_X, test_y in tqdm(test_loader, desc="Testing", leave=False, total=len(test_loader)):
            test_X = test_X.to(self.device)
            test_y = test_y.to(self.device)
            # --- 順伝播 ---
            output = self.__model(test_X)
            
            # --- 損失計算、予測 ---
            with torch.no_grad():
                loss, pred_y = calc_loss_pred_y(self.cfg, output, test_y, self.criterion, self.device)

            # --- 各値の格納 ---            
            avg_loss.update(loss.item())
            pred_list.extend(pred_y.tolist())
            true_list.extend(test_y.detach().to(torch.device("cpu")).numpy().tolist())
        
        pred_list = np.array(pred_list).flatten().tolist()
        true_list = np.array(true_list).flatten().tolist()
        
        # --- 評価指標の計算 ---
        metrics: list[float] = calc_scores(true_list, pred_list, self.cfg.data.metrics, calc_cm=self.cfg.data.calc_cm)
        
        return avg_loss.avg, metrics, true_list, pred_list
