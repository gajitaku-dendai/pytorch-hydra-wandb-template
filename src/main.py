import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch
import wandb
from conf.config import MyConfig
from utils import DictDotNotation, print_config, print_mode, print_final_results, \
    log_metrics_to_wandb, initialize_metrics, load_data, load_model, save_model, torch_fix_seed, load_config
from train import train
from test_ import evaluate_model
import multiprocessing

# --- スレッド数をCPUコア数の80%に設定 ---
# この設定は、特にCPUでのデータローディング時にマルチプロセッシングを効率的に利用するためのもの。
os.environ["OMP_NUM_THREADS"] = str(max(1, int(multiprocessing.cpu_count() * 0.8)))

def sweep(id: str = "", project_name: str = "sample") -> None:
    """
    wandbのsweepを実行する関数。
    sweep_idが指定されていない場合は、sweep.yamlを読み込んで新しいsweepを作成する。
    sweep_idが指定されている場合は、そのsweepを実行する。
    sweep.yamlはwandbのsweepを定義するyamlファイル。

    Parameters
    ----------
    id : str, default ""
        wandbのsweep_id。指定しない場合は新しいsweepを作成する.
    project_name : str, default "sample"
        wandbのプロジェクト名。デフォルトは"sample".
    """
    GlobalHydra.instance().clear()
    try:
        # sweep.yamlを読み込む。ファイルが存在しない場合はFileNotFoundErrorを発生させる。
        sweep_config = load_config("src/conf/sweep.yaml")
    except FileNotFoundError:
        print("Error: src/conf/sweep.yaml not found. Please ensure the sweep configuration file exists.")
        return # sweep.yamlがない場合は処理を終了する
    
    sweep_id = wandb.sweep(sweep_config, project=project_name) if id == "" else id
    # wandb.agentは、指定されたsweep_idで定義されたハイパーパラメータ空間を探索し、
    # main関数を繰り返し実行することで、最適なモデルを見つけ出す。
    wandb.agent(sweep_id, main)

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: MyConfig) -> None:
    """
    モデルの学習を行うメイン関数。
    config.yamlはHydraの設定ファイルで、モデルのハイパーパラメータやデータセットの設定を行う。
    config.yamlはsrc/conf/config.yamlに保存されている。

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト。
        実際はDictDotNotation型 or DictConfig型。
        Hydraのオーバーライド機能を使って、自動的にconfig.yamlをcfgに読み込む。
    """
    # --- デバイスのセットアップ ---
    # cfg.deviceで指定されたデバイスを使用。もし 'cuda' が指定されていてGPUが利用できない場合は 'cpu' にフォールバックする。
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- 乱数シードを固定 ---
    # 実験の再現性を確保するために乱数シードを固定する。   
    torch_fix_seed()

    # --- プロジェクト名を作成or取得 ---
    # wandbのプロジェクト名を動的に生成。設定で指定がなければ、モデル名とデータセット名から構成される。
    if cfg.wandb_project_name is None:
        dir_name = cfg.data.dir_name.replace("/", ".")
        project_name = f"{cfg.model.name}.{cfg.data.name}.{dir_name}"
    else:
        project_name = cfg.wandb_project_name

    # --- 出力ディレクトリの設定とwandbの初期化 ---
    # wandbを使うかどうかに応じて出力ディレクトリを設定し、wandbを初期化する。

    # wandbのsweepを使うかどうかの判定をここで行う
    # os.environ.get("WANDB_SWEEP_ID") が設定されている場合、これは wandb agent によって起動された実行であることを示す。
    is_wandb_agent_run = os.environ.get("WANDB_SWEEP_ID") is not None

    # sweepを使っている場合、そしてそれが agent によって起動された実行でない場合のみ、sweep関数を呼び出す
    if cfg.use_sweep and not is_wandb_agent_run:
        sweep(id=cfg.sweep_id, project_name=cfg.sweep_name)
        return # sweepを使う場合は、main関数を終了する

    # agentによって起動された場合、またはsweepを使用しない通常の実行の場合
    # ---------------------------------------------------------------
    if cfg.use_wandb:
        # wandbを初期化し、Hydraのconfigをwandbのconfigとしてログに記録する。
        wandb.init(project=project_name, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        # wandb.configはdot-notationでアクセスできないため、DictDotNotationに変換する。
        # これにより、cfg.model.nameのように直接属性としてアクセスできるようになる。
        cfg = DictDotNotation(wandb.config)
        # wandbの実行ディレクトリを出力ディレクトリとして設定
        cfg.output_dir = wandb.run.dir
    else:
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(f"Output directory: {cfg.output_dir}")

    try:
        # --- プロジェクト名がわかるように出力ディレクトリにファイル名として保存 ---
        # cfg.noteが指定されている場合は、note.txtも作成する。
        with open(f'{cfg.output_dir}/{project_name}.txt', 'w'):
            pass
        if cfg.note:
            with open(f'{cfg.output_dir}/note.txt', 'w') as f:
                pass

        # --- configを表示 ---
        # 現在の設定内容をコンソールに表示する。
        # 交差検証の分割数に基づいて、ループの回数を設定する。
        print_config(cfg)

        # --- 交差検証使うか否か ---
        # 交差検証の分割数に基づいて、ループの回数を設定する。
        # 交差検証を使う場合、あらかじめデータをfoldに分割して保存しておく必要あり。
        num = cfg.data.kfold_n_splits if cfg.data.use_kfold else 1
        print_mode(cfg.data.use_kfold)

        # --- metrics（評価指標）（辞書型）の初期化。wandb保存用 ---
        # 各評価フェーズ（train, valid, test）と各メトリックの履歴を保存するための辞書を初期化する。
        metrics = initialize_metrics(cfg)

        # --- 交差検証（num > 1） or train/valid/test（num = 1）---
        # 各foldに対して、データのロード、モデルの初期化、学習、評価、モデルの保存を行う。
        for i in range(num):
            # 現在のfold番号をcfgに設定。これはデータローディング時にKFoldの分割に利用される。
            cfg.data.now_fold = i
            if cfg.data.use_kfold:
                print(f"--- Fold {cfg.data.now_fold + 1}/{num} ---")
            # --- データの読み込み（DataLoader型） ---
            train_loader, valid_loader, test_loader = load_data(cfg, cfg.data.now_fold)
            
            # --- モデルと最適化関数、損失関数の読み込み ---
            model, optimizer, criterion = load_model(cfg, device, train_loader)

            # --- 学習 ---
            model = train(cfg, device, model, optimizer, criterion, train_loader, valid_loader, test_loader, cfg.data.now_fold)

            # 最良モデルまたは最終モデルを保存し、評価のためにその状態をロードする。
            model = save_model(cfg, cfg.data.now_fold, model)

            # --- モデルの評価 ---
            # 学習済みモデルを評価し、結果をmetrics辞書に追加する。
            metrics = evaluate_model(cfg, device, model, criterion, train_loader, valid_loader, test_loader, metrics)

            # 各foldのメモリを解放
            del train_loader, valid_loader, test_loader, model, optimizer, criterion
            torch.cuda.empty_cache()

        # --- 最終結果の表示 ---
        # 全てのfoldの平均と標準偏差を含む最終的な評価結果を表示する。
        print_final_results(cfg, metrics, num)

        # --- wandbに結果を記録 ---
        if cfg.use_wandb:
            log_metrics_to_wandb(cfg, metrics, num, cfg.output_dir)

    finally:
        if cfg.use_wandb:
            wandb.run.finish()

if __name__ == "__main__":
    main()