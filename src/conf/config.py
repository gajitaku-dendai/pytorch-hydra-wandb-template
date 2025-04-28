from dataclasses import dataclass, field

@dataclass
class Data():
    name: str = ""
    dir_name: str = ""
    use_pca: bool = False
    pca_n_components: float = 0.0
    standardize_type: str = ""
    metrics: list[str] = field(default_factory=list)
    task_type: str = ""

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
    input_size: int = 0


@dataclass
class MyConfig():
    model: Model = field(default_factory=Model())
    data: Data = field(default_factory=Data())

    use_wandb: bool = False
    wandb_project_name: str|None = ""

    use_kfold: bool = False
    kfold_n_splits: int = 0
    now_fold: int = 0

    output_dir: str = ""
    note: str = ""