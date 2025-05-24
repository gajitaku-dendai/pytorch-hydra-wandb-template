import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, \
    confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from conf.config import MyConfig
from timm.scheduler import CosineLRScheduler
import numpy as np
from torch.optim import Optimizer
import os

from architecture import get_model


def load_model(cfg: MyConfig, device: torch.device, train_loader: DataLoader) -> tuple[nn.Module, Optimizer, nn.Module]:
    """
    モデルと最適化関数、損失関数を読み込む関数。
    cfg.model.nameに応じてモデル、cfg.model.optimizerに応じて最適化関数、cfg.model.criterionに応じて損失関数を設定する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    device : torch.device
        使用するデバイス（GPU or CPU）。
    train_loader : DataLoader
        学習用データローダー。

    Returns
    -------
    model : nn.Module
        学習するモデル。
    optimizer : Optimizer
        最適化関数。
    criterion : nn.Module
        損失関数。

    See Also
    --------
    get_model : モデルを取得する関数。(architecture/__init__.py)
    get_optimizer : 最適化関数を取得する関数。(utils/training_utils.py)
    get_criterion : 損失関数を取得する関数。(utils/training_utils.py)
    """
    print("loading model...")
    model = get_model(cfg).to(device)
    optimizer = get_optimizer(cfg, model)
    # --- classificationの時は損失関数を重み付きに設定可能（cfg.model.use_weighted_loss） ---
    if cfg.data.task_type == "classification":
        count_label = np.unique(train_loader.dataset.y, return_counts=True)[1] 
    else:
        count_label = None
    criterion = get_criterion(cfg, count_label, device)
    print(f"success to load {cfg.model.name}\n\n")
    return model, optimizer, criterion

def save_model(cfg: MyConfig, fold: int, model: nn.Module) -> nn.Module:
    """
    モデルを保存する関数。
    交差検証の場合、foldごとにベストモデルと最終エポックモデルを保存する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    fold : int
        交差検証のfold番号。
        交差検証を使わない場合は0。
    model : nn.Module
        学習したモデル。
    
    Returns
    -------
    model : nn.Module
        学習したモデル（cfg.model.which_modelで指定した段階のモデル）。
    """
    best_model_path = f"{cfg.output_dir}/best.pth"
    best_model_fold_path = f"{cfg.output_dir}/best_{fold}.pth"
    best_exists = os.path.exists(best_model_path)
    if best_exists:
        os.rename(best_model_path, best_model_fold_path)
    os.rename(f"{cfg.output_dir}/last.pth", f"{cfg.output_dir}/last_{fold}.pth")
    if cfg.model.which_model != "last" and best_exists:
        model.load_state_dict(torch.load(f'{cfg.output_dir}/best_{fold}.pth', weights_only=True))
    return model

def get_criterion(cfg: MyConfig, counts, device) -> nn.Module:
    """
    損失関数を取得する関数。
    cfg.model.criterionに応じて損失関数を設定する。
    cfg.model.use_weighted_lossがTrueの場合は、重み付き損失関数を設定する（分類問題のみ）。
    
    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    counts : np.ndarray
        クラス分布（分類問題の場合）。
        0番目の要素がクラス0のサンプル数、1番目の要素がクラス1のサンプル数、...
    
    Returns
    -------
    criterion : nn.Module
        損失関数。
        cfg.model.criterionに応じた損失関数が設定される。
    
    Notes
    -----
    新たに損失関数を追加する場合は、この関数に追加する。
    elif cfg.model.criterion == "損失関数"
        return {損失関数クラス}
    のように追加する。
    """
    if cfg.model.criterion == "CrossEntropy":
        if cfg.model.use_weighted_loss:
            weight = torch.tensor(sum(counts)/(2*counts), dtype=torch.float32).to(device)
            return nn.CrossEntropyLoss(weight=weight)
        else:
            return nn.CrossEntropyLoss()
    elif cfg.model.criterion == "BinaryCrossEntropy":
        if cfg.model.use_weighted_loss:
            weight = torch.tensor(counts[0]/counts[1]).to(device)
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            return nn.BCEWithLogitsLoss()
    elif cfg.model.criterion == "MSE":
        return nn.MSELoss()
    
    else:
        raise ValueError(f"Criterion {cfg.model.criterion} is not defined.")
    
def get_optimizer(cfg: MyConfig, model: nn.Module) -> optim.Optimizer:
    """
    最適化関数を取得する関数。
    cfg.model.optimizerに応じて最適化関数を設定する。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    model : nn.Module
        学習するモデル。
    
    Returns
    -------
    optimizer : Optimizer
        最適化関数。
        cfg.model.optimizerに応じた最適化関数が設定される。
    
    Notes
    -----
    新たに最適化関数を追加する場合は、この関数に追加する。
    elif cfg.model.optimizer == "最適化関数"
        return {最適化関数クラス}
    のように追加する。
    """
    if cfg.model.optimizer == "Adam":
        return optim.Adam(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)
    
    elif cfg.model.optimizer == "SGD":
        return optim.SGD(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)
    
    elif cfg.model.optimizer == "RAdam":
        return optim.RAdam(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)
    
    elif cfg.model.optimizer == "AdamW":
        return optim.AdamW(
            model.parameters(),   
            lr=cfg.model.learning_rate,         
            weight_decay=cfg.model.l2_rate)
    
    else:
        raise ValueError(f"Optimizer {cfg.model.optimizer} is not defined.")
    
def get_scheduler(optimizer: optim.Optimizer, cfg: MyConfig) -> tuple[optim.lr_scheduler._LRScheduler, str]:
    """
    学習率スケジューラを取得する関数。
    cfg.model.schedulerに応じて学習率スケジューラを設定する。

    Parameters
    ----------
    optimizer : Optimizer
        最適化関数。
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。

    Returns
    -------
    scheduler : _LRScheduler
        学習率スケジューラ。
        cfg.model.schedulerに応じた学習率スケジューラが設定される。
    scheduler_type : str
        学習率スケジューラのタイプ。PyTorchのスケジューラは"torch"、timmのスケジューラは"timm"。
    
    Notes
    -----
    新たに学習率スケジューラを追加する場合は、この関数に追加する。
    elif cfg.model.scheduler == "学習率スケジューラ"
        return {学習率スケジューラクラス}, "torch" or "timm"
    のように追加する。
    また、必要な場合はcfg.modelにパラメータ管理用変数を追加してください。
    """
    if cfg.model.scheduler == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.model.scheduler_cycle, eta_min=cfg.model.lr_min), "torch"
    elif cfg.model.scheduler == "CosineLRScheduler":
        return CosineLRScheduler(optimizer,
                                  t_initial=cfg.model.epochs,
                                  lr_min=cfg.model.lr_min,
                                  warmup_t=cfg.model.warmup_t,
                                  warmup_lr_init=cfg.model.lr_min,
                                  warmup_prefix=True), "timm"

    else:
        raise ValueError(f"Scheduler {cfg.model.scheduler} is not defined.")

def calc_loss_pred_y(cfg: MyConfig, output: torch.Tensor, y: torch.Tensor, criterion: nn.Module,
                     device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """
    損失と予測値を計算する関数。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
    output : torch.Tensor
        モデルの出力。
    y : torch.Tensor
        正解値。
    criterion : nn.Module
        損失関数。
    device : torch.device
        使用するデバイス（GPU or CPU）。
    
    Returns
    -------
    loss : torch.Tensor
        損失値。
    pred_y : np.ndarray
        予測値。
        cfg.data.task_typeに応じて、分類問題の場合はクラスラベル、回帰問題の場合は数値が格納される。
    """
    loss = criterion(output, y)
    with torch.no_grad():
        output_cpu = output.to(torch.device("cpu") if device.type == "cuda" else device)
        output_np = output_cpu.detach().numpy()
        if cfg.data.task_type == "classification":
            if cfg.model.output_size == 1:
                # BCEWithLogitsLossを使う場合、モデル出力はSigmoidを通していないlogits。
                # Sigmoid関数は0で0.5なので、logitsが正なら1、負なら0と解釈できる。
                pred_y = np.where((output_np >= 0.0), 1, 0)
            else:
                pred_y = np.argmax(output_np, axis=1)
                pred_y = np.expand_dims(pred_y, axis=1)
        else:
            pred_y = output_np
        return loss, pred_y
    
def calc_scores(y: np.ndarray, pred_y: np.ndarray, metric_names: list[str], calc_cm: bool=False) -> list[any]:
    """
    評価指標を計算する関数。
    metric_namesに応じて、accuracy, f1, confusion_matrix, mae, r2, rmse, mseを計算する。
    calc_cmがTrueの場合は混同行列を計算する。

    Parameters
    ----------
    y : np.ndarray
        正解値。
    pred_y : np.ndarray
        予測値。
    metric_names : list[str]
        評価指標の名前。
        "acc", "f1", "mae", "r2", "rmse", "mse"...
    calc_cm : bool, default False
        混同行列を計算するかどうか。
        Trueの場合は混同行列を計算する。

    Returns
    -------
    metrics : list[any]
        評価指標の値。
        metric_namesに応じて、accuracy, f1, confusion_matrix, mae, r2, rmse, mse, ...が格納される。
    
    Notes
    -----
    新たに評価指標を追加する場合は、この関数に追加する。
    elif name == "評価指標名":
        metrics.append({評価指標名}(y, pred_y))
    """
    with torch.no_grad():
        metrics = []
        for name in metric_names:
            if name == "acc":
                metrics.append(accuracy_score(y, pred_y))
            elif name == "f1":
                metrics.append(f1_score(y, pred_y, average="macro"))
            elif name == "mae":
                metrics.append(mean_absolute_error(y, pred_y))
            elif name == "r2":
                metrics.append(r2_score(y, pred_y))
            elif name == "rmse":
                metrics.append(root_mean_squared_error(y, pred_y))
            elif name == "mse":
                metrics.append(mean_squared_error(y, pred_y))
            
            else:
                raise ValueError(f"Metric {name} is not defined.")
        if calc_cm:
            # 混同行列を計算する場合は必ず最後に追加する。後にmetrcis[-1]で参照しているため
            metrics.append(confusion_matrix(y, pred_y))
        return metrics

class AvgMeter:
    """
    平均値を計算するクラス。
    updateメソッドで値を追加し、avgプロパティで平均値を取得する。
    resetメソッドで初期化する。

    Attributes
    ----------
    _avg : float
        平均値。
    _sum : float
        合計値。
    _count : int
        値の個数。

    Methods
    -------
    reset() -> None
        初期化する。
    update(value: float) -> None
        値を追加する。
    avg() -> float
        平均値を取得する。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._avg = 0.0
        self._sum = 0.0
        self._count = 0

    def update(self, value):
        self._count += 1
        self._sum += value
        self._avg = self._sum / self._count
    
    @property
    def avg(self):
        return self._avg
    
class History:
    """
    学習履歴を保存するクラス。
    updateメソッドで値を追加し、histプロパティで履歴を取得する。
    resetメソッドで初期化する。

    Attributes
    ----------
    _hist : list[float]
        学習履歴。

    Methods
    -------
    reset() -> None
        初期化する。
    update(value: float) -> None
        値を追加する。
    hist() -> list[float]
        学習履歴を取得する。
    """
    def __init__(self):
        self._hist = []

    def update(self, value):
        self._hist.append(value)

    @property
    def hist(self):
        return self._hist

class EarlyStopping:
    """
    EarlyStoppingクラス。
    指定したメトリックが改善されない場合に学習を停止する。
    指定したパスに最良モデルを保存する。

    Attributes
    ----------
    metric : str
        モデルの評価指標。
        "acc", "f1", "mae", "r2", "rmse", "mse"...
    patience : int
        何エポック改善されなかったら学習を停止するか。
    verbose : bool
        学習の進捗を表示するかどうか。
    path : str
        モデルを保存するパス。
    best_score : float
        最良のスコア。
    counter : int
        改善されなかったエポック数。
    early_stop : bool
        学習を停止するかどうか。
    direction : int
        スコアの改善方向。
        1ならスコアが大きくなる方向、-1ならスコアが小さくなる方向。
    
    Methods
    -------
    __call__(score: float, model: nn.Module) -> None
        学習を停止するかどうかを判定する。
    _save_checkpoint(score: float, model: nn.Module) -> None
        最良モデルを保存する。
    
    Notes
    -----
    新たに評価指標を追加した場合、__init__メソッドのdirectionのif文を修正する必要がある。
    もし、新たな評価指標がスコアが大きいほど良い場合は、in ["acc", "f1", "r2"]に追加する。
    """
    def __init__(self, path, patience, verbose=False, metric=None):
        self.metric = metric
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.direction = 1 if metric.split("_")[1] in ["acc", "f1", "r2"] else -1
        self.path = path

    def __call__(self, score, model):
        if self.best_score is None or (score - self.best_score) * self.direction > 0:
            self._save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f"Best {self.metric}: {self.best_score:.5f}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, score, model):
        if self.best_score is None:
            self.best_score = score
        if self.verbose:
            print(f'{self.metric} improved ({self.best_score:.5f} -> {score:.5f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
