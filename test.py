import argparse
import hydra
from utils.test_utils import test_model
from omegaconf import DictConfig


@hydra.main(
    config_path="./confs/training_confs",
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    test_model(cfg.model_dir, batch_size=cfg.get("batch_size", 128), test_data_path=cfg.test_data_path)


if __name__ == "__main__":
    main()
