# 可視化＆ログ＆パラメータ管理に特化した深層学習フレームワーク

## はじめに

このリポジトリは、  
**Python × pyenv × poetry × Hydra × W&B × PyTorch** を使って  
**「これから機械学習を始めたい人」** や  
**「再現性の高い開発環境を構築したい人」** のためのテンプレート（フレームワーク）です。

✅ 初心者でもわかりやすく  
✅ Pythonの環境構築からモデル実装、可視化・ログ管理までカバー  
✅ すぐに開発を始められる構成

---

## 特徴

- `pyenv` × `Poetry` によるシンプルなPython環境管理
- `Hydra` で柔軟な設定ファイル管理
  - 設定ファイルからモデルとデータセットを柔軟に組み合わせ可能
- `Weights & Biases` による実験ログ・可視化
- リポジトリ全体が再利用可能なテンプレート

---

## 使用技術・ツール

| ツール | 役割 |
|--------|------|
| [pyenv](https://github.com/pyenv/pyenv) | Pythonバージョン管理 |
| [poetry](https://python-poetry.org/) | パッケージ・仮想環境管理 |
| [Hydra](https://github.com/facebookresearch/hydra) | 設定ファイルの管理 |
| [Weights & Biases](https://wandb.ai/site) | ログ・可視化ツール |
| [PyTorch](https://pytorch.org/) | 機械学習ライブラリ |

---

## 動作確認済み環境

- Python 3.11.9（※ pyenv で事前インストール必須（後述））
- CUDA 12.4（GPU利用時、NVIDIA公式の対応ドライバが必要）
- OS：Windows 11 + WSL2 / Ubuntu 22.04 など

---

## ディレクトリ構造

```bash
.
├── outputs  # Hydraのログ出力
├── wandb # W&Bのログ出力
├── pyproject.toml # Poetryのパッケージ管理用ファイル
└── src
    ├── architecture # モデルクラス保存ディレクトリ
    │   ├── __init__.py
    │   ├── cnn_2d.py # モデルの例: 2D-CNN
    │   └── mlp.py # モデルの例: MLP
    │
    ├── conf # Hydra用Configディレクトリ
    │   ├── __init__.py
    │   ├── config.py # 型定義ファイル
    │   ├── config.yaml # Hydraパラメータ管理（全体）
    │   ├── data # Hydraパラメータ管理（データセットごと）
    │   │   ├── california_housing.yaml
    │   │   ├── diabetes.yaml
    │   │   └── mnist.yaml
    │   ├── model # Hydraパラメータ管理（モデルごと）
    │   │   ├── cnn_2d.yaml
    │   │   └── mlp.yaml
    │   └── sweep.yaml
    │
    ├── database # データセット保存用ディレクトリ
    │   ├── __init__.py
    │   ├── california_housing # データセットの例: California Housing
    │   │   ├── california_housing_dataset.py # データをまとめあげるファイル
    │   │   ├── features # データの実態が存在するフォルダ
    │   │   └── load_data.ipynb # データの読み込みをするノートブック
    │   ├── diabetes # データセットの例: Diabetes
    │   │   ├── diabetes_dataset.py # データをまとめあげるファイル
    │   │   ├── features # データの実態が存在するフォルダ
    │   │   └── load_data.ipynb # データの読み込みをするノートブック
    │   │── mnist # データセットの例: MNIST
    │   │   ├── img # データの実態が存在するフォルダ
    │   │   ├── load_data.ipynb # データの読み込みをするノートブック
    │   │   └── mnist_dataset.py # データをまとめあげるファイル
    │   └── base_dataset.py # データをまとめ上げるファイルの既定ファイル
    │
    ├── main.py # 学習実行ファイル (コマンドラインから実行するのはこのファイル)
    ├── models # 学習済モデルの重み保存ディレクトリ
    ├── notebook # .ipynbファイル（ノートブック）を保存するフォルダ
    │   └── check_model.ipynb # モデルの構造をチェックするノートブック
    ├── evaluate.py # モデルの評価をするファイル
    ├── evaluator.py # 実際にモデルの評価（epoch単位）をするファイル
    ├── train.py # モデルの学習をするファイル
    ├── trainer.py # 実際にモデルの学習（epoch単位）をするファイル
    │
    └── utils # 様々な関数を保存
        ├── __init__.py
        ├── dataProcessor.py # 主にデータ関連の関数が存在
        ├── main_utils.py # 主にmain.pyで活用する関数が存在
        └── training_utils.py # 主に学習に関する関数が存在
```

---

## 事前準備  

1. `pyenv` と `Poetry` のインストール
インストール方法については、筆者が作成した以下の記事を読んでください。
[pyenv × PoetryでPythonのパッケージ管理を効率化！（Windows / Linux 両対応・初心者歓迎）](https://qiita.com/gajitaku/items/c50b945725fcf8d75bb8)

2. W&Bへの新規登録
[W&B公式サイト](https://wandb.ai/site/ja/)からアカウントを登録。
設定画面からAPI keyを取得してコピーしておきましょう。

## セットアップ方法

```bash
# リポジトリをクローン
git clone https://github.com/gajitaku-dendai/pytorch-hydra-wandb-template.git
cd ml-template

# Python環境のセットアップ（pyenv と poetry インストール前提）
pyenv install 3.11.9
pyenv local 3.11.9
poetry install

# wandbの設定
wandb login #APIキーの入力

# 学習の実行（例）
poetry run python src/main.py
```

---

## 使い方

- このフレームワークは、`src/main.py` がすべての処理をまとめる核になっています
- 基本的な学習や評価のフローは、この `main.py` を読めば全体の流れが把握できます！
- すべてのファイルには詳細なコメントを書いておいたので、各ファイルの中身を読み進めることで、詳細な挙動や設定方法は深く理解できるはずです
- 文字が多くて読み飛ばしたくなる気持ちは痛いほどわかりますが、**読み飛ばした瞬間理解できなくなります** のでちゃんと読もう

### 1. データの準備

まず、学習につかうデータを準備する必要があります。サンプルのデータセットを用意してありますので、対応する `load_data.ipynb` を実行してデータをダウンロードしてください。

- MNISTデータセット
  - 手書き数字の分類問題で使われるデータセット
  - 対応ファイル: `src/database/mnist/load_data.ipynb`
  - 実行すると、`src/database/mnist/img/raw/` ディレクトリに画像データが保存される

- California Housingデータセット
  - カリフォルニア州の住宅価格予測で使われる回帰データセット
  - 対応ファイル: `src/database/california_housing/load_data.ipynb`
  - 実行すると、`src/database/california_housing/features/raw/` ディレクトリに特徴量データが保存される

- Diabetesデータセット
  - 糖尿病の進行度予測で使われる回帰データセット
  - 対応ファイル: `src/database/diabetes/load_data.ipynb`
  - 実行すると、`src/database/diabetes/features/raw/` ディレクトリに特徴量データが保存される

実際に、自分でデータセットを用意する場合は、各サンプルデータセットを参考にファイルを作成してください。
例えば、`abc` というデータセットを用意する場合は以下の手順でファイルを追加・変更します。

1. `src/database/abc` というフォルダを作成する
2. `src/dtabase/abc/load_data.ipynb` を各サンプルデータセットの該当ファイルを参考に作成し、データを `src/database/abc/**/**/**.npy` として保存する
3. `src/database/abc/abc_dataset.py` を各サンプルデータセットの該当ファイルを参考に作成する
`MyDataset(BaseDataset)` と `TrainDataset(MyDataset)` と `ValidDataset(MyDataset)` と `TestDataset(MyDataset)` クラスを作成しよう（基本サンプルファイルをコピペして、`TrainDataset`内の`input_type`に依存しているところを変更すればOK）
4. `src/database/__init__.py` の `get_dataset()` に `abc_dataset` の分岐を追加する
5. `src/conf/data/abc.yaml` を各サンプルデータセットの該当yamlファイルを参考に作成する
yamlファイル内の `name` は `"abc"` とすること

### 2. モデルを新規で追加する場合

このフレームワークではサンプルのモデルを2つ用意してあります。

- MLP
  - 線形層と活性化関数のみのニューラルネットワークモデル
  - 対応ファイル: `src/architecture/mlp.py`

- 2D-CNN
  - 画像を扱う畳み込みニューラルネットワークモデル
  - 対応ファイル: `src/architecture/cnn_2d.py`

実際に、自分でモデルを新たに用意する場合は、各サンプルモデルを参考にファイルを作成してください。
例えば、`abc` というモデルを用意する場合は以下の手順でファイルを追加・変更します。

1. `src/architecture/abc.py` を各サンプルモデルのファイルを参考に作成する
各サンプルモデルと同様に `ABC` クラスと `get_model()` 関数を作成しましょう
2. `src/architecture/__init__.py` の `get_model()` に `abc` の分岐を追加する
3. `src/conf/model/abc.yaml` を各サンプルモデルの該当yamlファイルを参考に作成する
yamlファイル内の `name` は `"abc"` とすること

### 3. 使うモデルやデータセット、パラメータを変更する方法

- 学習のモデルやデータセット、ハイパーパラメータの設定は、`src/conf/config.yaml` を編集して行う
- このファイルは **Hydra** によって管理されていて、 `model` や `data` といったサブ設定ファイルを読み込むようになっている。

#### 例1. MNISTデータセットで2D-CNNモデルを動かす場合

1. `src/conf/config.yaml` を以下のように編集する

    ```yaml:config.yaml
    ### src/conf/config.yaml

    defaults:
    - _self_
    - model: cnn_2d # CNN_2Dモデルを使用
    - data: mnist # MNISTデータセットを使用

    # ... その他の設定はデフォルトでOK
    ```

2. 必要であれば、`src/conf/model/cnn_2d.yaml` や `src/conf/data/mnist.yaml` 内のハイパーパラメータやデータ処理の設定を調整してください

#### 例2. California HousingデータセットでMLPモデルを動かす場合

1. `src/conf/config.yaml` を以下のように編集する

    ```yaml:config.yaml
    ### src/conf/config.yaml

    defaults:
    - _self_
    - model: mlp # MLPモデルを使用
    - data: california_housing # California Housingデータセットを使用

    # ... その他の設定はデフォルトでOK
    ```

2. 必要であれば、`src/conf/model/mlp.yaml` や `src/conf/data/california_housing.yaml` 内のハイパーパラメータやデータ処理の設定を調整してください

### 4. 学習の実行

- 設定が終わったら、`src/main.py` を実行するだけ

```bash:Bash
poetry run python src/main.py
```

- これで、指定したモデルとデータセットで学習が始まり、Weights & Biasesへのログ記録や、設定された評価指標の表示、Early Stoppingなどの機能が自動的に実行されます！

### 5. Hydraならではの簡単な設定変更方法

- 例えば、MNISTデータセットを2D-CNNモデルに入力して、モデルのエポック数を50にしたい！と思ったとき、yamlファイルを変更しないで、コマンドライン上で指定することもできます。

```bash:Bash
poetry run python src/main.py data=mnist model=cnn_2d model.epochs=50
```

`src/main.py` の後に、順不同で設定したい変数を上記のように指定するだけです。
このように指定した場合、yamlファイル上の設定を上書きして実行することができます。

---

## ライセンス

このリポジトリは MIT ライセンスのもとで公開されています。  
ご自由にご利用・改変・再配布いただけますが、再利用時は著作権表示を残してください。  

詳しくは [LICENSE](./LICENSE) ファイルをご確認ください。  

---

## 参考になるサイト

- [5行でカッコいい可視化を「WandB」入門](https://qiita.com/Yu_Mochi/items/4fc283ebc31225d4e106)
- [5分でできるHydraによるパラメーター管理](https://qiita.com/Isaka-code/items/3a0671306629756895a6)
- [Hydra 公式ドキュメント](https://hydra.cc/docs/intro/)
- [Weights & Biases 公式サイト](https://wandb.ai/)
- [Poetry 公式](https://python-poetry.org/)  
