# 必要なモジュールをimport
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作クラス
from utils import *
from tester import *
from conf.config import MyConfig

def test(cfg: MyConfig,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        test_loader: DataLoader) -> nn.Module:
    
    tester = Tester(device, model, criterion, cfg)       
    tester.model.eval()
    loss, metrics = tester.test_step(test_loader) 

    cm = None
    for metric, value in zip(cfg.data.metrics, metrics):
        if metric != "cm":
            print(f"{metric}: {value:.5f}", end=", ")
        if metric == "cm":
            cm = value
    print("\n")

    if cm is not None:
        print("confusion matrix")
        print(cm)
        print()  

    return loss, metrics