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
    """
    モデルの評価を行う関数．
    与えられたデータローダーに対して評価を行い，損失と評価指標を計算する．
    得られた評価指標をプリントする．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    device : torch.device
        使用するデバイス（GPU or CPU）．
    model : nn.Module
        学習済みモデル．
    criterion : nn.Module
        損失関数．
    test_loader : DataLoader
        評価用データローダー．
    
    Returns
    -------
    loss : float
        平均損失．
    metrics : list[any]
        評価指標のリスト．
        cfg.data.metricsに指定された評価指標を計算する．
    """
    
    tester = Tester(device, model, criterion, cfg)       
    tester.model.eval()
    loss, metrics, true_list, pred_list = tester.test_step(test_loader) 

    cm = None
    if cfg.data.calc_cm:
        cm = metrics.pop()
        print("confusion matrix")
        print(cm)
        print()  
    for metric, value in zip(cfg.data.metrics, metrics):
        print(f"{metric}: {value:.5f}", end=", ")
    print("\n")

    return loss, metrics, true_list, pred_list

def evaluate_model(cfg: MyConfig, device: torch.device, model: nn.Module, criterion: nn.Module, train_loader: DataLoader,
                   valid_loader: DataLoader, test_loader: DataLoader, metrics: dict[str, list[any]]) -> dict[str, list[any]]:
    """
    モデルの評価を行う関数．
    学習用データ，検証用データ，テスト用データそれぞれに対して評価を行い，metricsに結果を追加する．
    
    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    device : torch.device
        使用するデバイス（GPU or CPU）．
    model : nn.Module
        学習したモデル．
    criterion : nn.Module
        損失関数．
    train_loader : DataLoader
        学習用データローダー．
    valid_loader : DataLoader
        検証用データローダー．
    test_loader : DataLoader
        テスト用データローダー．
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
    
    Returns
    -------
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
        各データセット（train, valid, test）ごとに評価指標が追加される．
    """
    print("\n####################")
    print("result")
    print("####################\n")
    metrics = evaluate(cfg, device, model, criterion, train_loader, metrics, "train")
    metrics = evaluate(cfg, device, model, criterion, valid_loader, metrics, "valid")
    metrics = evaluate(cfg, device, model, criterion, test_loader, metrics, "test")
    return metrics

def evaluate(cfg: MyConfig, device: torch.device, model: nn.Module, criterion: nn.Module, loader: DataLoader,
             metrics: dict[str, list[any]], mode: str) -> dict[str, list[any]]:
    """
    与えられたデータローダーに対するモデルの評価を行う関数．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    device : torch.device
        使用するデバイス（GPU or CPU）．
    model : nn.Module
        学習したモデル．
    criterion : nn.Module
        損失関数．
    loader : DataLoader
        データローダー（学習or検証orテスト）．
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
    mode : str
        評価するデータセットの種類（train, valid, test）．
    
    Returns
    -------
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
        データローダー（学習 or 検証 or テスト）の評価指標が追加される．
    """
    print(mode)
    loss, metrics_value, true_list, pred_list = test(cfg, device, model, criterion, loader)
    metrics[f"{mode}_loss"].append(loss)
    metrics[f"{mode}_true"].append(true_list)
    metrics[f"{mode}_pred"].append(pred_list)
    for metric, value in zip(cfg.data.metrics, metrics_value):
        metrics[f"{mode}_{metric}"].append(value)
    return metrics