from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from gaze_rl.models.mlp_policy import MLPPolicy
from gaze_rl.trainers.offline_trainer import OfflineTrainer
from gaze_rl.utils.logger import log
from r3m import load_r3m


class BCTrainer(OfflineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

    def setup_model(self):
        # setup vision encoder if needed to extract features from images
        # when we have image observations, this is where we load R3M
        if "feature_extractor" in self.cfg.model and self.cfg.model.feature_extractor:
            if self.cfg.model.feature_extractor == "r3m":
                feature_extractor = load_r3m("resnet50")
                feature_extractor.eval()
                feature_extractor.to(self.device)
            else:
                # TODO
                import ipdb

                ipdb.set_trace()
                # maybe it is one we trained ourselves
                cfg_file = Path(self.cfg.model.feature_extractor_ckpt) / "config.yaml"
                vision_cfg = OmegaConf.load(cfg_file)

            if not self.cfg.model.finetune_feature_extractor:
                log("Freezing feature extractor, not training these weights", "blue")
                for param in feature_extractor.parameters():
                    param.requires_grad = False
        else:
            feature_extractor = None

        model = MLPPolicy(
            self.cfg.model,
            input_dim=self.obs_shape,
            output_dim=self.cfg.env.action_dim,
            feature_extractor=feature_extractor,
        )
        return model

    def compute_loss(self, batch, train: bool = True):
        if self.cfg.env.image_obs:
            action_pred = self.model(batch.observations["front_rgb"])
        else:
            action_pred = self.model(batch.observations)
        loss = self.loss_fn(action_pred, batch.actions)
        loss = loss.mean()
        return {"action_loss": loss.item()}, loss
