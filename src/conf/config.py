from dataclasses import dataclass, field

# Hydraは、config.yamlの内容に従ってドットアクセス可能な辞書型のオブジェクトを作成する。
# しかし、型ヒントや補完が効かない。
# そこで、dataclassを使って、yamlファイルの内容をクラスとして定義する。
# これにより、型ヒントや補完が効くようになる。
# 例えば、cfg.model.まで打つと、cfg.model.の中身が表示される。
# これにより、cfg.model.の中身を確認しながらコーディングできる。
# さらに型ヒントにより、例えばcfg.model.nameはstr型であることがわかる。

# 上記メリットを使用したいのであれば、Hydra用のyamlファイルにパラメータを追加した場合、このクラスにも追加する必要がある。

# 注意：実際にこのクラスをインスタンス化することはない。ただ補完や型ヒントのためだけに使う。
# そのため、ここで指定したデフォルト値は意味がないし、指定した型以外の値が入力されたとしてもエラーにはならない。
# ただし、yamlファイルに存在しないパラメータを指定した場合は、存在しないのに補完が効いてしまうので注意。

# 第二層にあるconf/data/**.yaml用
@dataclass
class Data():
    name: str = ""
    dir_name: str = ""
    use_pca: bool = False
    pca_n_components: float = 0.0
    standardize_type: str = ""
    metrics: list[str] = field(default_factory=list)
    calc_cm: bool = False
    class_names: list[str] = field(default_factory=list)
    task_type: str = ""
    valid_ratio: float = 0.0
    data_splits: list[str] = field(default_factory=list)

    use_kfold: bool = False
    kfold_n_splits: int = 0
    now_fold: int = 0
    kfold_valid_ratio: float = 0.0

# 第二層にあるconf/model/**.yaml用
@dataclass
class Model():
    name: str = ""
    input_type: str = ""
    pretrained_path: str = ""

    epochs: int = 0
    early: int = 0
    learning_rate: float = 0.0
    l2_rate: float = 0.0
    batch_size: int = 0
    test_batch_size: int = 0

    optimizer: str = ""
    criterion: str = ""
    use_weighted_loss: bool = False

    which_model: str = ""
    monitor: str = ""
    output_size: int = 0

    scheduler: str = ""
    lr_min: float = 0.0
    warmup_t: int = 0
    scheduler_cycle: int = 0

    use_noise: bool = False

    num_channels: int = 0
    input_size: int|list[int] = 0


# 第一層にあるconf/config.yaml用
@dataclass
class MyConfig():
    # 二層目に設定ファイルを増やす場合、以下のように記述する。
    model: Model = field(default_factory=Model())
    data: Data = field(default_factory=Data())
    # train: Train = field(default_factory=Train()) みたいに

    use_wandb: bool = False
    wandb_project_name: str|None = ""

    use_sweep: bool = False
    sweep_id: str = ""
    sweep_name: str = ""

    device: str = ""

    output_dir: str = ""
    note: str = ""