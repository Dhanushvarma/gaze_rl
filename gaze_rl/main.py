import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gaze_rl.trainers.bc_trainer import BCTrainer
from gaze_rl.trainers.hbc_trainer import HBCTrainer
from gaze_rl.utils.general_utils import omegaconf_to_dict, print_dict
from gaze_rl.utils.logger import log

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    trainer_to_cls = {"bc": BCTrainer, "hbc": HBCTrainer}

    if cfg.name not in trainer_to_cls:
        raise ValueError(f"Invalid trainer name: {cfg.name}")

    log("start")
    trainer = trainer_to_cls[cfg.name](cfg)
    trainer.train()

    # end program
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    main()
