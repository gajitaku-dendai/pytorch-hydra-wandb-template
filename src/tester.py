import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import *
from conf.config import MyConfig

class Tester:
    def __init__(self, device:torch.device, model: nn.Module, criterion, cfg: MyConfig):
        self.__model = model
        self.criterion = criterion
        self.cfg = cfg
        self.device = device

    @property
    def model(self):
        return self.__model
    
    def test_step(self, test_loader: DataLoader) -> tuple[float, float, float, np.ndarray]:
        avg_loss = AvgMeter()
        true_list, pred_list = [], []
        
        for test_X, test_y in tqdm(test_loader, desc="Testing", leave=False, total=len(test_loader)):
            test_X = test_X.to(self.device)
            test_y = test_y.to(self.device)
            output = self.__model(test_X)
            
            with torch.no_grad():
                loss, pred_y = calc_loss_pred_y(self.cfg, output, test_y, self.criterion, self.device)            
            avg_loss.update(loss.item())
            pred_list.extend(pred_y.tolist())
            true_list.extend(test_y.detach().to(torch.device("cpu")).numpy().tolist())
        
        pred_list, true_list = np.array(pred_list).flatten(), np.array(true_list).flatten()
        metrics: list[float] = calc_scores(true_list, pred_list, self.cfg.data.metrics, calc_cm=True)
        
        return avg_loss.avg, metrics
