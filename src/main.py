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
os.environ["OMP_NUM_THREADS"] = str(max(1, int(multiprocessing.cpu_count() * 0.8)))

def sweep(id: str = "", project_name: str = "sample") -> None:
    """
    wandbのsweepを実行する関数．
    sweep_idが指定されていない場合は，sweep.yamlを読み込んで新しいsweepを作成する．
    sweep_idが指定されている場合は，そのsweepを実行する．
    sweep.yamlはwandbのsweepを定義するyamlファイル．

    Parameters
    ----------
    id : str, default ""
        wandbのsweep_id．指定しない場合は新しいsweepを作成する.
    project_name : str, default "sample"
        wandbのプロジェクト名．デフォルトは"sample".
    """
    GlobalHydra.instance().clear()
    sweep_config = load_config("src/conf/sweep.yaml")
    sweep_id = wandb.sweep(sweep_config, project=project_name) if id == "" else id
    wandb.agent(sweep_id, main)

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: MyConfig) -> None:
    """
    モデルの学習を行うメイン関数．
    config.yamlはHydraの設定ファイルで，モデルのハイパーパラメータやデータセットの設定を行う．
    config.yamlはsrc/conf/config.yamlに保存されている．

    Parameters
    ----------
    cfg : MyConfig
        型ヒントとしてMyConfigを使っているHydraの構成オブジェクト．
        実際はDictDotNotation型 or DictConfig型．
        Hydraのオーバーライド機能を使って，自動的にconfig.yamlをcfgに読み込む．
    """
    # --- cudaが使える → device=GPU，使えない → device=CPU ---
    GPU = torch.device("cuda")
    CPU = torch.device("cpu")
    device = GPU if torch.cuda.is_available() else CPU
    torch_fix_seed() # 乱数シードを固定

    # --- プロジェクト名を作成or取得 ---
    if cfg.wandb_project_name is None:
        project_name = f"{cfg.model.name}.{cfg.data.name}.{cfg.data.dir_name}"
    else:
        project_name = cfg.wandb_project_name

    # --- configをドットアクセス可能にする ---
    if cfg.use_wandb:
        wandb.init(project=project_name, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        cfg = DictDotNotation(wandb.config)
        cfg.output_dir = wandb.run.dir
    else:
        cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # --- wandbのsweepを使うか ---
    if cfg.use_sweep and os.environ.get("WANDB_SWEEP_ID") is None:
        sweep(id=cfg.sweep_id, project_name=cfg.sweep_name)

    # --- プロジェクト名を出力ディレクトリに保存 ---
    with open(f'{cfg.output_dir}/{project_name}.txt', 'w'):
        pass

    # --- configを表示 ---
    print_config(cfg)

    # --- 交差検証使うか否か ---
    num = cfg.kfold_n_splits if cfg.use_kfold else 1
    print_mode(cfg.use_kfold)

    # --- metrics（評価指標）（辞書型）の初期化．wandb保存用 ---
    metrics = initialize_metrics(cfg)

    # --- 交差検証（num > 1） or train/valid/test（num = 1）---
    for i in range(num):
        cfg.now_fold = i
        print(f"{cfg.now_fold + 1}/{num}")
        # --- データの読み込み（DataLoader型） ---
        train_loader, valid_loader, test_loader = load_data(cfg, cfg.now_fold)
        
        # --- モデルと最適化関数，損失関数の読み込み ---
        model, optimizer, criterion = load_model(cfg, device, train_loader)

        # --- 学習 ---
        model = train(cfg, device, model, optimizer, criterion, train_loader, valid_loader, test_loader, cfg.now_fold)
        model = save_model(cfg, cfg.now_fold, model)

        # --- モデルの評価 ---
        metrics = evaluate_model(cfg, device, model, criterion, train_loader, valid_loader, test_loader, metrics)

        del train_loader, valid_loader, test_loader, model, optimizer, criterion
        torch.cuda.empty_cache()

    # --- 最終結果の表示 ---
    print_final_results(cfg, metrics, num)

    if cfg.use_wandb:
        # --- wandbに結果を記録 ---
        log_metrics_to_wandb(cfg, metrics, num, cfg.output_dir)
        wandb.run.finish()

if __name__ == "__main__":
    main()