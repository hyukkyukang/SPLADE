import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from script.evaluate_mteb import run as run_sparse_benchmark


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate_nanobeir")
def main(cfg: DictConfig) -> None:
    run_sparse_benchmark(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
