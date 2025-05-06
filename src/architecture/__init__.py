# モデルを追加した場合，__init__.pyに追加する
# from architecture import "model_name"

from conf.config import MyConfig
from architecture import mlp
from architecture import cnn_2d

def get_model(cfg: MyConfig):
    """
    モデルを取得する関数．
    cfg.model.nameに応じてモデルを取得する．
    cfg.model.nameに指定されたモデルが存在しない場合はエラー．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．

    Notes
    -----
    モデルを追加した場合，この関数に追加する必要がある．
    例：cfg.model.name = "transformer"の場合，transformer.pyにget_model関数を実装する．
    その後，この関数にimportして追加する．
    例：from architecture import transformer
        ~~
        elif cfg.model.name == "transformer":
            model = transformer.get_model(cfg)
    """
    model = None
    if cfg.model.name == "mlp":
        model = mlp.get_model(cfg)
    elif cfg.model.name == "cnn_2d":
        model = cnn_2d.get_model(cfg)
    # elif cfg.model.name == "transformer":
    #     model = transformer.get_model(cfg)
    else:
        raise ValueError(f"Model '{cfg.model.name}' is not defined.")
    return model