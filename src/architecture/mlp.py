import torch
import torch.nn as nn
from conf.config import MyConfig

def get_model(cfg: MyConfig=None):
    """
    自作MLPモデルを取得する関数．
    
    Parameters
    ----------
    cfg : MyConfig 
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    
    Returns
    -------
    model : nn.Module
        自作MLPモデル．
    """
    # 分類問題の場合はクラス数，回帰問題の場合は１．（他モデルを作成するときも以下のコード使うと楽）
    num_classes = cfg.model.output_size #number of classes 

    # 入力サイズは特徴量の次元数．
    # （他モデルを作成するときは，cfg.model.input_sizeに入力サイズをdataset.py内であらかじめ指定しておくと楽．）
    input_size = int(cfg.model.input_size)

    # もしチャンネル数が必要な場合は，cfg.model.num_channelsに指定しておく．
    num_channels = cfg.model.num_channels

    # モデルの初期化
    model = MLP(num_classes, input_size, num_channels, cfg)

    # 学習済みモデルの重みを読み込む場合
    # cfg.model.pretrained_pathに学習済みモデルのパスを指定しておく．
    # 例．"src/models/**.pth" に保存すると良い
    path = cfg.model.pretrained_path
    if path is not None:
        weights=torch.load(path, weights_only=True)
        model.load_state_dict(weights)

    return model

class MLP(nn.Module):
    """
    自作MLPモデルクラス．
    nn.Moduleを継承しており，基本的なPyTorchのモデルの実装に従っている．
    __init__メソッドでモデルの構造を定義し，forwardメソッドで順伝播を実装している．
    それさえ満たせば，PyTorchのモデルは自由に実装できる．

    Parameters
    ----------
    num_classes : int
        分類問題の場合はクラス数，回帰問題の場合は1．
    input_size : int
        入力サイズ．特徴量の次元数．
    num_channels : int
        特徴量にチャンネルが存在する場合は使用してください．
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    """
    def __init__(self, num_classes: int, input_size: int, num_channels: int, cfg: MyConfig):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        # 線形層．input_size → 32
        self.fc1 = nn.Linear(input_size, 32)

        # 線形層．32 → 32
        self.fc2 = nn.Linear(32, 32)

        # 線形層．32 → num_classes
        self.fc3 = nn.Linear(32, num_classes)

        # Dropout層を追加することで，過学習を防ぐ．
        # Dropoutは，学習時にランダムに一部のユニット(以下の場合25%)を無効化することで，過学習を防ぐ．
        self.dropout = nn.Dropout(0.25)

        # ReLU活性化関数を追加することで，非線形性を持たせる．
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, input_size)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        # x: (batch_size, num_classes)
        # num_classesが1の場合は，(batch_size,)にする
        x = x.squeeze(-1)
        return x