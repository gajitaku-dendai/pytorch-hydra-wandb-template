import torch
import torch.nn as nn
from conf.config import MyConfig

def get_model(cfg:MyConfig=None):
    """
    自作2D_CNNモデルを取得する関数．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    
    Returns
    -------
    model : nn.Module
        自作2D_CNNモデル．
    """
    # 分類問題の場合はクラス数，回帰問題の場合は１．（他モデルを作成するときも以下のコード使うと楽）
    num_classes = cfg.model.output_size

    # 入力サイズは画像サイズ．
    # （他モデルを作成するときは，cfg.model.input_sizeに入力サイズをdataset.py内であらかじめ指定しておくと楽．）
    # 画像の場合，cfg.model.input_sizeは(高さ, 幅)のタプル． 
    input_size = cfg.model.input_size

    # 画像のチャンネル数は，グレースケール画像の場合は1，RGB画像の場合は3．
    # 画像のチャンネル数はcfg.model.num_channelsに指定しておく．
    num_channels = cfg.model.num_channels

    # モデルの初期化
    model = CNN_2D(num_classes, input_size, num_channels, cfg)

    # 学習済みモデルの重みを読み込む場合
    # cfg.model.pretrained_pathに学習済みモデルのパスを指定しておく．
    # 例．"src/models/**.pth" に保存すると良い
    path = cfg.model.pretrained_path
    if path is not None:
        weights=torch.load(path, weights_only=True)
        model.load_state_dict(weights)

    return model

class CNN_2D(nn.Module):
    """
    自作2D_CNNモデルクラス．
    nn.Moduleを継承しており，基本的なPyTorchのモデルの実装に従っている．
    __init__メソッドでモデルの構造を定義し，forwardメソッドで順伝播を実装している．
    それさえ満たせば，PyTorchのモデルは自由に実装できる．

    Parameters
    ----------
    num_classes : int
        分類問題の場合はクラス数，回帰問題の場合は1．
    input_size : tuple[int,int]
        入力サイズ．（高さ, 幅）のタプル．
    num_channels : int
        画像のチャンネル数．グレースケール画像の場合は1，RGB画像の場合は3．
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
    """
    def __init__(self, num_classes: int, input_size: tuple[int,int], num_channels: int, cfg: MyConfig):
        super(CNN_2D, self).__init__()

        # フィルタ数がnum_channels→16，両側に1ピクセル追加(padding)して，3x3(kernel_size)のカーネルを1ピクセル(stride)ずつスライドさせる．
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalizationを追加することで，学習を安定させる．
        # Batch Normalizationは，学習時にミニバッチの平均と分散を計算して，その値を使って正規化する．
        self.bn1 = nn.BatchNorm2d(16)
        
        # フィルタ数が16→32，両側に1ピクセル追加(padding)して，3x3(kernel_size)のカーネルを1ピクセル(stride)ずつスライドさせる．
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(32)
        
        # フィルタ数が32→64，両側に1ピクセル追加(padding)して，3x3(kernel_size)のカーネルを1ピクセル(stride)ずつスライドさせる．
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(64)

        # プーリング層を追加することで，特徴マップのサイズを小さくする．
        # この場合は，特徴マップのサイズを半分にするために，2x2のカーネルを2ピクセル(stride)ずつスライドさせる．
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # (batch_size, num_channels, height, width)を(batch_size, num_channels*height*width)に変換するためのFlatten層
        self.flatten = nn.Flatten()

        # ReLU活性化関数を追加することで，非線形性を持たせる．
        self.relu = nn.ReLU()

        # fc1への入力サイズを計算するために，ダミーの入力を通して出力サイズを計算
        conv_output_size = self._get_conv_output_size(num_channels, input_size)

        # 線形層．conv層の出力を128次元に変換する．
        self.fc1 = nn.Linear(conv_output_size, 128)

        self.bn_fc1 = nn.BatchNorm1d(128)
        
        # 線形層．128次元をnum_classes次元に変換する．
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout層を追加することで，過学習を防ぐ．
        # Dropoutは，学習時にランダムに一部のユニット(以下の場合25%)を無効化することで，過学習を防ぐ．
        self.dropout = nn.Dropout(0.25)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    def _get_conv_output_size(self, num_channels, input_size):
        # fc1への入力サイズを計算するために，ダミーの入力を通して出力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, input_size[0], input_size[1])
            x = self.features(dummy_input)
            return x.numel()

    def forward(self, x):
        # x: (batch_size, num_channels, height, width)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # x: (batch_size, num_classes)
        # num_classesが1の場合は，(batch_size,)にする
        x = x.squeeze(-1)
        return x