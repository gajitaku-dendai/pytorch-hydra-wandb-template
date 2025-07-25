import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import wandb


from utils import EarlyStopping, History, get_scheduler
from trainer import Trainer
from conf.config import MyConfig

def train(cfg: MyConfig,
          device: torch.device,
          model: nn.Module,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          test_loader: DataLoader,
          now_fold: int) -> nn.Module:
    """
    モデルの学習を行う関数。
    学習用データに対して学習を行い、検証用データとテスト用データに対して評価を行う。
    学習率スケジューラーを使用して、学習率を調整する。
    Early Stoppingを使用して、過学習を防ぐ+最良モデルの保存。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    device : torch.device
        使用するデバイス（GPU or CPU）。
    model : nn.Module
        学習するモデル。
    optimizer : optim.Optimizer
        最適化関数。
    criterion : nn.Module
        損失関数。
    train_loader : DataLoader
        学習用データローダー。
    valid_loader : DataLoader
        検証用データローダー。
    test_loader : DataLoader
        テスト用データローダー。
    now_fold : int
        交差検証のfold番号。
        交差検証を使わない場合は0。

    Returns
    -------
    model : nn.Module
        学習したモデル。
    """
    
    # --- Early stopping ---
    # cfg.data.data_splitsのリストに'train'しかない場合はEarlyStoppingを無効化
    use_early_stopping = not (cfg.data.data_splits == ["train"])
    if use_early_stopping:
        monitor = cfg.model.monitor
        if cfg.model.which_model == "valid_best":
            if valid_loader is None:
                raise ValueError("valid_loader is None. Cannot use *valid_best* for early stopping.")
            scheduler_metric = f"valid_{monitor}"
            early_stopping = EarlyStopping(path=f"{cfg.output_dir}/best.pth", patience=cfg.model.early, verbose=True,
                                        metric=scheduler_metric)
        elif cfg.model.which_model == "test_best":
            if test_loader is None:
                raise ValueError("test_loader is None. Cannot use *test_best* for early stopping.")
            scheduler_metric = f"test_{monitor}"
            early_stopping = EarlyStopping(path=f"{cfg.output_dir}/best.pth", patience=cfg.model.early, verbose=True,
                                        metric=scheduler_metric)
        elif cfg.model.which_model == "last": 
            if test_loader is None:
                scheduler_metric = f"valid_{monitor}"
            else:
                scheduler_metric = f"test_{monitor}"
            early_stopping = EarlyStopping(path=f"{cfg.output_dir}/best.pth", patience=cfg.model.early, verbose=True,
                                        metric=scheduler_metric)
        
    # --- 1epochの学習バッチ処理させるクラス ---
    trainer = Trainer(device, model, criterion, optimizer, cfg)

    # --- 評価指標のヒストリー ---
    history: dict[str, History] = {}
    for phase in cfg.data.data_splits:
        history[f"{phase}_loss"] = History()
        for metric in cfg.data.metrics:
            history[f"{phase}_{metric}"] = History()

    # --- 学習率スケジューラー ---
    scheduler, sch_type = get_scheduler(optimizer, cfg)

    # --- 学習 ---
    try:
        for epoch in tqdm(range(cfg.model.epochs), desc="Epochs", leave=True):
            trainer.model.train()
            # --- 学習データの学習 ---
            train_loss, train_metrics = trainer.train_step(train_loader)
            # --- 指標の履歴を更新 ---
            history['train_loss'].update(train_loss)
            for metric, value in zip(cfg.data.metrics, train_metrics):
                history[f"train_{metric}"].update(value)

            trainer.model.eval()

            if valid_loader is not None:
                # --- 検証データの検証 ---
                valid_loss, valid_metrics = trainer.valid_step(valid_loader)
                history['valid_loss'].update(valid_loss)
                for metric, value in zip(cfg.data.metrics, valid_metrics):
                    history[f"valid_{metric}"].update(value)

            if test_loader is not None:
                # --- テストデータの検証 ---
                test_loss, test_metrics = trainer.valid_step(test_loader)
                history['test_loss'].update(test_loss)
                for metric, value in zip(cfg.data.metrics, test_metrics):
                    history[f"test_{metric}"].update(value)

            # --- ログ出力 ---
            print("\n\n", end="")
            for phase in cfg.data.data_splits:
                loss = history[f"{phase}_loss"].hist[-1]
                metrics = [history[f"{phase}_{metric}"].hist[-1] for metric in cfg.data.metrics]
                print(f"{phase}_loss: {loss:.5f}", end=", ")
                for metric_name, metric_value in zip(cfg.data.metrics, metrics):
                    print(f"{phase}_{metric_name}: {metric_value:.5f}", end=", ")
            print()

            # --- Early Stopping ---
            if use_early_stopping:
                early_stopping(history[scheduler_metric].hist[-1], model)

            # --- wandbへのログ出力 ---
            if cfg.use_wandb:
                step = epoch + int(cfg.model.epochs * now_fold * 1.5)
                log_dict = {}
                for phase in cfg.data.data_splits:
                    log_dict[f"{phase}_loss"] = history[f"{phase}_loss"].hist[-1]
                    for metric in cfg.data.metrics:
                        log_dict[f"{phase}_{metric}"] = history[f"{phase}_{metric}"].hist[-1]
                log_dict.update({
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })
                wandb.log(log_dict, step=step)
            
            # --- 定期的なモデルの保存 ---
            if (epoch == 0) or ((epoch + 1) % 10 == 0):
                torch.save(model.state_dict(), f"{cfg.output_dir}/last.pth")

            # --- Early Stoppingフラグ ---
            if use_early_stopping:
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

            # --- 学習率スケジューラー更新 ---
            if sch_type == "torch":
                scheduler.step()
            elif sch_type == "timm":
                scheduler.step(epoch)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    print("Finished Training")
    torch.save(model.state_dict(), f"{cfg.output_dir}/last.pth")
    return trainer.model
