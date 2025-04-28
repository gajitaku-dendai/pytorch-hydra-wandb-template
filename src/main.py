import os
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import torch.nn as nn
import random
import wandb
import yaml
from conf.config import MyConfig
from utils import *
from database import get_dataset
from train import train
from test_ import test
from architecture import get_model
import multiprocessing

os.environ["OMP_NUM_THREADS"] = str(max(1, int(multiprocessing.cpu_count() * 0.8)))

GPU = torch.device("cuda")
CPU = torch.device("cpu")
device = GPU if torch.cuda.is_available() else CPU

class DictDotNotation(dict):
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictDotNotation(value)
        self.__dict__ = self

    def __getattr__(self, key: str) -> any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DictDotNotation' object has no attribute '{key}'")

def torch_fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def print_config(cfg: MyConfig) -> None:
    print("####################")
    print("Config")
    print("====================")
    print(cfg)
    print("####################\n\n")

def print_mode(use_kfold: bool) -> None:
    mode = "kFold_Mode" if use_kfold else "Train_Valid_Test_Mode"
    print("####################")
    print(mode)
    print("####################\n\n")

def initialize_metrics(cfg: MyConfig):
    # --- 評価指標 ---
    metrics = {"train_loss": [], "valid_loss": [], "test_loss": []}
    for m in cfg.data.metrics:
        metrics[f"train_{m}"] = []
        metrics[f"valid_{m}"] = []
        metrics[f"test_{m}"] = []
    return metrics

def load_data(cfg: MyConfig, fold: int) -> tuple[DataProcessor, DataLoader, DataLoader, DataLoader]:
    print("loading data...")
    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, now_fold=fold)
    cfg.model.num_channels = train_dataset.num_channels
    cfg.model.input_size = train_dataset.input_size
    print_dataset_details(cfg, train_dataset, valid_dataset, test_dataset)
    train_loader = dataset_to_dataloader(train_dataset, batch_size=cfg.model.batch_size)
    valid_loader = dataset_to_dataloader(valid_dataset, batch_size=cfg.model.test_batch_size, isValid=True)
    test_loader = dataset_to_dataloader(test_dataset, batch_size=cfg.model.test_batch_size, isTest=True)
    print("success to load data!\n\n")
    return train_loader, valid_loader, test_loader
    
def print_dataset_details(cfg: MyConfig, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset) -> None:
    if cfg.data.task_type == "classification":
        print("train_details:", *np.unique(train_dataset.y, return_counts=True), "/", len(train_dataset))
        print("valid_details:", *np.unique(valid_dataset.y, return_counts=True), "/", len(valid_dataset))
        print("test_details:", *np.unique(test_dataset.y, return_counts=True), "/", len(test_dataset))
    else:
        print("train_details (length):", len(train_dataset))
        print("valid_details (length):", len(valid_dataset))
        print("test_details (length):", len(test_dataset))
    print("data_shape (except num_sample):", train_dataset[0][0].shape)

def load_model(cfg: MyConfig, device: torch.device, train_loader: DataLoader) -> tuple[nn.Module, Optimizer, nn.Module]:
    print("loading model...")
    model = get_model(cfg).to(device)
    optimizer = get_optimizer(cfg, model)
    if cfg.data.task_type == "classification":
        count_label = np.unique(train_loader.dataset.y, return_counts=True)[1] 
    else:
        count_label = None
    criterion = get_criterion(cfg, count_label, device)
    print(f"success to load {cfg.model.name}\n\n")
    return model, optimizer, criterion

def save_model(cfg: MyConfig, fold: int, model: nn.Module) -> nn.Module:
    # 交差検証のfoldごとにベストモデルとラストモデルの保存
    os.rename(f"{cfg.output_dir}/best.pth", f"{cfg.output_dir}/best_{fold}.pth")
    os.rename(f"{cfg.output_dir}/last.pth", f"{cfg.output_dir}/last_{fold}.pth")
    if cfg.model.which_model != "last":
        model.load_state_dict(torch.load(f'{cfg.output_dir}/best_{fold}.pth', weights_only=True))
    return model

def evaluate_model(cfg: MyConfig, device: torch.device, model: nn.Module, criterion: nn.Module, train_loader: DataLoader,
                   valid_loader: DataLoader, test_loader: DataLoader, metrics: dict[str, list[any]], fold: int) -> dict[str, list[any]]:
    print("\n####################")
    print("result")
    print("####################\n")
    metrics = evaluate(cfg, device, model, criterion, train_loader, metrics, "train", fold)
    metrics = evaluate(cfg, device, model, criterion, valid_loader, metrics, "valid", fold)
    metrics = evaluate(cfg, device, model, criterion, test_loader, metrics, "test", fold)
    return metrics

def evaluate(cfg: MyConfig, device: torch.device, model: nn.Module, criterion: nn.Module, loader: DataLoader,
             metrics: dict[str, list[any]], mode: str, fold: int) -> dict[str, list[any]]:
    print(mode)
    loss, metrics_value = test(cfg, device, model, criterion, loader)
    metrics[f"{mode}_loss"].append(loss)
    for metric, value in zip(cfg.data.metrics, metrics_value):
        metrics[f"{mode}_{metric}"].append(value)
    return metrics

def log_metrics_to_wandb(cfg: MyConfig, metrics: dict[str, list[any]], num: int, path: str) -> None:
    for i in range(num):
        log_data = {f"{metric_name}_fold_{i}": metrics[f"{metric_name}"][i] for metric_name in [
            f"{mode}_{m}" for mode in ["train", "valid", "test"] for m in cfg.data.metrics
        ]}
        wandb.run.log(log_data)

    # 平均と標準偏差をまとめて記録
    summary_data = {}
    for mode in ["train", "valid", "test"]:
        for metric_name in cfg.data.metrics:
            key = f"{mode}_{metric_name}"
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            summary_data[mean_key] = np.mean(metrics[key])
            summary_data[std_key] = np.std(metrics[key])

    # 最後にpathも記録しておく
    summary_data["path"] = path
    wandb.run.log(summary_data)

def print_final_results(cfg, metrics: dict[str, list[any]], num: int) -> None:
    print("\n####################")
    print("Final Result")
    print("####################\n")
    print_metrics(cfg, metrics, "train", num)
    print_metrics(cfg, metrics, "valid", num)
    print_metrics(cfg, metrics, "test", num)

def print_metrics(cfg: MyConfig, metrics: dict[str, list[any]], mode: str, num: int) -> None:
    print(f"{mode}_loss:", [f"{x:.3f}" for x in metrics[f'{mode}_loss']])
    print("mean:", f"{np.mean(metrics[f'{mode}_loss']):.3f}", "std:", f"{np.std(metrics[f'{mode}_loss']):.3f}\n")
    
    for metric_name in cfg.data.metrics:
        print(f"{mode}_{metric_name}:", [f"{x:.3f}" for x in metrics[f'{mode}_{metric_name}']])
        print("mean:", f"{np.mean(metrics[f'{mode}_{metric_name}']):.3f}", 
              "std:", f"{np.std(metrics[f'{mode}_{metric_name}']):.3f}\n")
    
    if metrics.get(f'{mode}_cm') and metrics[f'{mode}_cm']:  # confusion matrixがあるときだけ出す
        for i in range(num):
            print(metrics[f'{mode}_cm'][i])
            print()

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: MyConfig) -> None:
    torch_fix_seed() # 乱数シードを固定

    # --- プロジェクト名を作成or取得 ---
    if cfg.wandb_project_name is None:
        project_name = f"{cfg.model.name}.{cfg.model.input_type}.{cfg.data.dir_name}"
    else:
        project_name = cfg.wandb_project_name

    # --- configをドットアクセス可能にする ---
    if cfg.use_wandb:
        wandb.init(project=project_name, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        cfg = DictDotNotation(wandb.config)
        cfg.output_dir = wandb.run.dir
    else:
        cfg = DictDotNotation(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # --- プロジェクト名を出力ディレクトリに保存 ---
    with open(f'{cfg.output_dir}/{project_name}.txt', 'w'):
        pass

    # --- configを表示 ---
    print_config(cfg)

    # --- 交差検証使うか否か ---
    num = cfg.kfold_n_splits if cfg.use_kfold else 1
    print_mode(cfg.use_kfold)

    # --- metrics（評価指標）（辞書型）の初期化．wandb保存用 ---
    metrics = initialize_metrics(cfg)

    # --- 交差検証（num > 1） or train/valid/test（num = 1）---
    for i in range(num):
        cfg.now_fold = i
        print(f"{cfg.now_fold + 1}/{num}")
        # --- データの読み込み（DataLoader型） ---
        train_loader, valid_loader, test_loader = load_data(cfg, cfg.now_fold)
        
        # --- モデルと最適化関数，損失関数の読み込み ---
        model, optimizer, criterion = load_model(cfg, device, train_loader)

        # --- 学習 ---
        model = train(cfg, device, model, optimizer, criterion, train_loader, valid_loader, test_loader, cfg.now_fold)
        model = save_model(cfg, cfg.now_fold, model)

        # --- モデルの評価 ---
        metrics = evaluate_model(cfg, device, model, criterion, train_loader, valid_loader, test_loader, metrics, cfg.now_fold)

        del train_loader, valid_loader, test_loader, model, optimizer, criterion
        torch.cuda.empty_cache()

    # --- 最終結果の表示 ---
    print_final_results(cfg, metrics, num)

    if cfg.use_wandb:
        # --- wandbに結果を記録 ---
        log_metrics_to_wandb(cfg, metrics, num, cfg.output_dir)
        wandb.run.finish()

if __name__ == "__main__":
    main()
