import numpy as np
import torch
import random
import wandb
import yaml

from conf.config import MyConfig

class DictDotNotation(dict):
    """
    ドットアクセス可能にした辞書クラス．辞書のキーを属性としてアクセスできるようにする．

    Examples
    --------
    >>> data = DictDotNotation({"key1": "value1", "nested": {"key2": "value2"}})
    >>> data.key1
    'value1'
    >>> data.nested.key2
    'value2'
    >>> data.nonexistent
    AttributeError: 'DictDotNotation' object has no attribute 'nonexistent'
    """
    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictDotNotation(value)

    def __getattr__(self, key: str) -> any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DictDotNotation' object has no attribute '{key}'")

def torch_fix_seed(seed: int = 42) -> None:
    """
    Pytorchで使われる乱数シードを固定する関数．

    Parameters
    ----------
    seed : int, default 42
        乱数シード．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def print_config(cfg: MyConfig) -> None:
    """
    configの内容を表示する関数．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    """
    print("####################")
    print("Config")
    print("====================")
    print(cfg)
    print("####################\n\n")

def print_mode(use_kfold: bool) -> None:
    """
    交差検証を使うかどうかのモードを表示する関数．

    Parameters
    ----------
    use_kfold : bool
        交差検証を使うかどうかのフラグ．Trueなら交差検証，Falseならtrain/valid/test分割．
    """
    mode = "kFold_Mode" if use_kfold else "Train_Valid_Test_Mode"
    print("####################")
    print(mode)
    print("####################\n\n")

def log_metrics_to_wandb(cfg: MyConfig, metrics: dict[str, list[any]], num: int, path: str) -> None:
    """
    wandbに評価指標を記録する関数．
    交差検証の場合，foldごとに評価指標を記録する．
    なお，交差検証の場合，fold全体の平均と標準偏差も計算して記録する．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
    num : int
        交差検証のfold数．
    path : str
        wandbのログ保存先パス．
    """
    cfg.data.metrics.append("loss")
    for i in range(num):
        log_data = {f"{metric_name}_fold_{i}": metrics[f"{metric_name}"][i] for metric_name in [
            f"{mode}_{m}" for mode in ["train", "valid", "test"] for m in cfg.data.metrics
        ]}
        wandb.run.log(log_data)

    # confusion matrixをfoldごとに記録
    if cfg.data.calc_cm:
        # クラス名を追加したい場合，cfg.data.class_namesに追加すること．nullにしておけば，インデックスを使う．
        if cfg.data.class_names is not None:
            class_names = cfg.data.class_names
        else:
            class_names = [str(i) for i in np.arange(2 if cfg.model.output_size == 1 else cfg.model.output_size).tolist()]
        for i in range(num):
            for mode in ["train", "valid", "test"]:
                log_data = {f"{mode}_cm_fold_{i}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=metrics[f"{mode}_true"][i],
                    preds=metrics[f"{mode}_pred"][i],
                    class_names=class_names
                )}
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
    """
    最終結果（すべての評価指標）を表示する関数．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
    num : int
        交差検証のfold数．
    """
    print("\n####################")
    print("Final Result")
    print("####################\n")
    print_metrics(cfg, metrics, "train", num)
    print_metrics(cfg, metrics, "valid", num)
    print_metrics(cfg, metrics, "test", num)

def initialize_metrics(cfg: MyConfig) -> None:
    """
    metrics（評価指標）を初期化する関数．
    cfg.data.metricsに指定された評価指標を初期化し，wandb保存用（またはログ表示用）の辞書を作成する．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    """
    metrics = {"train_loss": [], "valid_loss": [], "test_loss": [],
               "train_cm": [], "valid_cm": [], "test_cm": [],
               "train_true": [], "valid_true": [], "test_true": [],
               "train_pred": [], "valid_pred": [], "test_pred": []}
    for m in cfg.data.metrics:
        metrics[f"train_{m}"] = []
        metrics[f"valid_{m}"] = []
        metrics[f"test_{m}"] = []
    return metrics

def print_metrics(cfg: MyConfig, metrics: dict[str, list[any]], mode: str, num: int) -> None:
    """
    指定されたmode（train or valid or test）に対する評価指標を表示する関数．
    各評価指標の平均と標準偏差も表示する．
    混同行列がある場合は表示する．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    metrics : dict[str, list[any]]
        評価指標を格納する辞書．
    mode : str
        評価するデータセットの種類（train, valid, test）．
    num : int
        交差検証のfold数．
    """
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

def load_config(file: str) -> dict[str, any]:
    """
    YAMLファイルを読み込む関数．
    YAMLファイルの内容を辞書型で返す．

    Parameters
    ----------
    file : str
        読み込むYAMLファイルのパス．
    
    Returns
    -------
    dict[str, any]
        YAMLファイルの内容を辞書型で返す．
    """
    with open(file, 'r') as yml:
        return yaml.safe_load(yml)