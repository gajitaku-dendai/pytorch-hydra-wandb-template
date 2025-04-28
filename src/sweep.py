import wandb
from src.main import main
import yaml
import argparse

def load_config(file: str) -> dict[str, any]:
    with open(file, 'r') as yml:
        return yaml.safe_load(yml)

def sweep(id: str = "", project_name: str = "sample") -> None:
    sweep_config = load_config("sweep.yaml")
    sweep_id = wandb.sweep(sweep_config, project=project_name) if id == "" else id
    wandb.agent(sweep_id, main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep.")
    parser.add_argument("--id", type=str, default="", help="Sweep ID (optional).")
    parser.add_argument("--project_name", type=str, default="sample", help="W&B project name.")
    args = parser.parse_args()

    sweep(id=args.id, project_name=args.project_name)