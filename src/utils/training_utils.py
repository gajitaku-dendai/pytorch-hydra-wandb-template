import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, \
    confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from conf.config import MyConfig
from timm.scheduler import CosineLRScheduler
import numpy as np

def dataset_to_dataloader(dataset, batch_size, isValid=False, isTest=False):
    g = torch.Generator()
    g.manual_seed(42)
    if isValid or isTest:
        return DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            generator=g)
    else:
        return DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True,
                            generator=g)
    
def get_criterion(cfg: MyConfig, counts, device) -> nn.Module:
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
    
def get_optimizer(cfg: MyConfig, model: nn.Module) -> optim.Optimizer:
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
    
def get_scheduler(optimizer: optim.Optimizer, cfg: MyConfig) -> tuple[optim.lr_scheduler._LRScheduler, str]:
    if cfg.model.scheduler == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.model.scheduler_cycle, eta_min=cfg.model.lr_min), "torch"
    elif cfg.model.scheduler == "CosineLRScheduler":
        return CosineLRScheduler(optimizer,
                                  t_initial=cfg.model.epochs,
                                  lr_min=cfg.model.lr_min,
                                  warmup_t=cfg.model.warmup_t,
                                  warmup_lr_init=cfg.model.lr_min,
                                  warmup_prefix=True), "timm"

def calc_loss_pred_y(cfg: MyConfig, output, y, criterion, device:torch.device) -> tuple[torch.Tensor, np.ndarray]:
    loss = criterion(output, y)
    with torch.no_grad():
        output_cpu = output.to(torch.device("cpu") if device.type == "cuda" else device)
        output_np = output_cpu.detach().numpy()
        if cfg.data.task_type == "classification":
            if cfg.model.output_size == 1:
                pred_y = np.where((output_np >= 0.0), 1, 0)
            else:
                pred_y = np.argmax(output_np, axis=1)
                pred_y = np.expand_dims(pred_y, axis=1)
        else:
            pred_y = output_np
            return loss, pred_y
    
def calc_scores(y, pred_y, metric_names, calc_cm=False):
    with torch.no_grad():
        metrics = []
        for name in metric_names:
            if name == "acc":
                metrics.append(accuracy_score(y, pred_y))
            elif name == "f1":
                metrics.append(f1_score(y, pred_y, average="macro"))
            elif name == "cm" and calc_cm:
                metrics.append(confusion_matrix(y, pred_y))
            elif name == "mae":
                metrics.append(mean_absolute_error(y, pred_y))
            elif name == "r2":
                metrics.append(r2_score(y, pred_y))
            elif name == "rmse":
                metrics.append(root_mean_squared_error(y, pred_y))
            elif name == "mse":
                metrics.append(mean_squared_error(y, pred_y))
        return metrics
        
class AvgMeter:
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
    def __init__(self):
        self._hist = []

    def update(self, value):
        self._hist.append(value)

    @property
    def hist(self):
        return self._hist

class EarlyStopping:
    def __init__(self, path, patience=5, verbose=False, metric=None):
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
