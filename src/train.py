from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import torch
import rootutils

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

torch.cuda.manual_seed_all(42975)
torch.set_float32_matmul_precision('medium')

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Initiate Logger
    logger = instantiate(cfg.logger) if not cfg.test else None

    lr_monitor = LearningRateMonitor(logging_interval='step')
    exp_id = logger.experiment.id if 'Wandb' in cfg.logger._target_ else logger.log_dir
    ckpt_callback = instantiate(cfg.checkpoint, dirpath=cfg.checkpoint.dirpath.format(exp_id=exp_id))
    trainer = instantiate(cfg.trainer, callbacks=[ckpt_callback, lr_monitor], logger=logger)
    if trainer.global_rank==0 and 'Wandb' in cfg.logger._target_:
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    train_loader = instantiate(cfg.dataset)
    val_loader = instantiate(cfg.dataset, dataset={'split': 'validation'}, shuffle=False)

    model = instantiate(cfg.model, expname=cfg.experiment.expname, format=cfg.data.format)

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path
    )
   
    return model

if __name__ == "__main__":
    main()