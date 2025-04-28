from conf.config import MyConfig
from architecture import mlp

def get_model(cfg: MyConfig):
    model = None
    if cfg.model.name == "mlp":
        model = mlp.get_model(cfg)
    else:
        print(f'Model {cfg.model.name} is not defined.')
    return model