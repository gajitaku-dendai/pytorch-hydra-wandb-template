import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import *
from conf.config import MyConfig

class Trainer:
    def __init__(self, device:torch.device, model: nn.Module, criterion, optimizer: optim.Optimizer, cfg: MyConfig):
        self.__model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device

    @property
    def model(self):
        return self.__model

    def train_step(self, train_loader: DataLoader) -> tuple[float, float, float, np.ndarray, np.ndarray]:
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
    
    def valid_step(self, valid_loader: DataLoader) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        avg_loss = AvgMeter()
        true_list, pred_list = [], []
        
        for test_X, test_y in tqdm(valid_loader, desc="Testing", leave=False, total=len(valid_loader)):
            test_X = test_X.to(self.device)
            test_y = test_y.to(self.device)
            output = self.__model(test_X)
            
            with torch.no_grad():
                loss, pred_y = calc_loss_pred_y(self.cfg, output, test_y, self.criterion, self.device)            
            avg_loss.update(loss.item())
            pred_list.extend(pred_y.tolist())
            true_list.extend(test_y.detach().to(torch.device("cpu")).numpy().tolist())
        
        pred_list, true_list = np.array(pred_list).flatten(), np.array(true_list).flatten()
        metrics: list[float] = calc_scores(true_list, pred_list, self.cfg.data.metrics, calc_cm=False)

        return avg_loss.avg, metrics