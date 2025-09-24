#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import torch
from transformers import AutoModel

from fim.data.dataloaders import DataLoaderFactory
from fim.models.blocks import ModelFactory
from fim.trainers.trainer import TrainerFactory
from fim.trainers.utils import cleanup, clear_gpu_cache, get_accel_type, setup, setup_environ_flags
from fim.utils.helper import GenericConfig, expand_params, load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging


setup_logging()

warnings.filterwarnings("ignore", module="matplotlib")
logger = RankLoggerAdapter(logging.getLogger(__name__))


@click.command()
@click.option("-c", "--config", "cfg_path", required=True, type=click.Path(exists=True), help="path to config file")
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-r",
    "--resume",
    "resume",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    help="Path to checkpoint directory to resume training from.",
)
@click.option("-e", "--epoch", "epoch", default="last-epoch", help="Epoch to resume training from.")
def main(cfg_path: Path, log_level: int, resume: Path, epoch: str):
    config = load_yaml(cfg_path)
    gs_configs = expand_params(config)
    train(gs_configs, resume, epoch)


def train(configs: List[GenericConfig], resume: Optional[Path] = None, epoch: str = "last-epoch"):
    for config in configs:
        if config.distributed.enabled:
            train_distributed(config, resume, epoch)
        else:
            train_single(config, resume, epoch)


def train_distributed(config: GenericConfig, resume: Optional[Path] = None, epoch: str = "last-epoch"):
    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(rank, world_size)

    if rank == 0:
        logger.info("Starting Experiment: %s", config.experiment.name)
        logger.info("World Size: %d", world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        device_map = config.experiment.device_map
        if device_map == "auto":
            device_map = get_accel_type()

        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
        if config.model.get("base_model", None) is None:
            model = ModelFactory.create(config.model.to_dict())
        else:
            model = AutoModel.from_pretrained(config.model.base_model)
        trainer = TrainerFactory.create(
            config.trainer.name, model=model, dataloader=dataloader, config=config, resume=resume, resume_epoch=epoch
        )
        trainer.train()
    # dist.barrier()
    cleanup()


def train_single(config: GenericConfig, resume: Optional[Path] = None, epoch: str = "last-epoch"):
    logger.info("Starting Experiment: %s", config.experiment.name)

    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    torch.cuda.empty_cache()

    # device_map = config.experiment.device_map

    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    if config.model.get("base_model", None) is None:
        logger.info("Creating model from scratch")
        model = ModelFactory.create(config.model.to_dict())
    else:
        logger.info(f"Loading model from {config.model.base_model}")
        model = AutoModel.from_pretrained(config.model.base_model)

    trainer = TrainerFactory.create(
        config.trainer.name, model=model, dataloader=dataloader, config=config, resume=resume, resume_epoch=epoch
    )
    trainer.train()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
