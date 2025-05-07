# 個人的ベストな機械学習（深層学習）テンプレート 🔧🔥  

## はじめに 🌱

このリポジトリは、  
**Python × WSL2 × pyenv × poetry × Hydra × W&B × PyTorch** を使って  
**「これから機械学習を始めたい人」** や  
**「再現性の高い環境を構築したい人」** のためのテンプレートです。

✅ 初学者でも安心！  
✅ Pythonの環境構築から可視化・ログ管理までカバー  
✅ すぐに開発を始められる構成になっています！

---

## 特徴 ✨

- Windows / Linux 対応（WSL2前提 or ネイティブ環境OK）
- `pyenv` × `poetry` によるシンプルなPython環境管理
- `Hydra` で柔軟な設定ファイル管理
- `Weights & Biases` による実験ログ・可視化
- リポジトリ全体が再利用可能なテンプレート

---

## 使用技術・ツール 🧰

| ツール | 役割 |
|--------|------|
| [WSL2](https://learn.microsoft.com/ja-jp/windows/wsl/) | WindowsでLinux環境を使うための仕組み |
| [pyenv](https://github.com/pyenv/pyenv) | Pythonバージョン管理 |
| [poetry](https://python-poetry.org/) | パッケージ・仮想環境管理 |
| [Hydra](https://github.com/facebookresearch/hydra) | 設定ファイルの管理 |
| [Weights & Biases](https://wandb.ai/site) | ログ・可視化ツール |
| [PyTorch](https://pytorch.org/) | 機械学習ライブラリ |

---

## 動作確認済み環境 ⚙️

- Python 3.11.9（※ pyenv で事前インストール必須（後述））
- CUDA 12.4（GPU利用時、NVIDIA公式の対応ドライバが必要）
- OS：Windows 11 + WSL2 / Ubuntu 22.04 など

---

## ディレクトリ構造

```bash
.
├── outputs  # Hydraのログ出力
├── wandb # W&Bのログ出力
├── poetry.lock
├── pyproject.toml
└── src
    ├── architecture # モデルクラス保存ディレクトリ
    │   ├── __init__.py
    │   ├── cnn_2d.py
    │   └── mlp.py
    ├── conf # Hydra用Configディレクトリ
    │   ├── __init__.py
    │   ├── config.py
    │   ├── config.yaml
    │   ├── data
    │   │   ├── california_housing.yaml
    │   │   ├── diabetes.yaml
    │   │   └── mnist.yaml
    │   ├── model
    │   │   ├── cnn_2d.yaml
    │   │   └── mlp.yaml
    │   └── sweep.yaml
    ├── database # データセット保存用ディレクトリ
    │   ├── __init__.py
    │   ├── california_housing
    │   │   ├── california_housing_dataset.py
    │   │   ├── features
    │   │   └── load_data.ipynb
    │   ├── diabetes
    │   │   ├── diabetes_dataset.py
    │   │   ├── features
    │   │   │   └── raw
    │   │   └── load_data.ipynb
    │   └── mnist
    │       ├── img
    │       │   └── raw
    │       ├── load_data.ipynb
    │       └── mnist_dataset.py
    ├── main.py # 学習実行ファイル
    ├── models # 学習済モデルの重み保存ディレクトリ
    ├── notebook # .ipynbファイルはこちら
    │   └── check_model.ipynb
    ├── test_.py
    ├── tester.py
    ├── train.py
    ├── trainer.py
    └── utils # 様々な関数を保存
        ├── __init__.py
        ├── dataProcessor.py
        ├── main_utils.py
        └── training_utils.py
```

---

## セットアップ方法 🚀

```bash
# リポジトリをクローン
git clone https://github.com/gajitaku-dendai/ml-template.git
cd ml-template

# Python環境のセットアップ（pyenv と poetry インストール前提）
pyenv install 3.11.9
pyenv local 3.11.9

poetry install

# 学習の実行（例）
poetry run python src/main.py
```

---

## 使い方

すべてのファイルにコメントを記述済み！  
隅々まで読んで理解して、使い倒してください！  
（※解説サイト作成中。しばしお待ち下さい。）

---

## ライセンス 📄

このリポジトリは MIT ライセンスのもとで公開されています。  
ご自由にご利用・改変・再配布いただけますが、再利用時は著作権表示を残してください。

詳しくは [LICENSE](./LICENSE) ファイルをご確認ください。
