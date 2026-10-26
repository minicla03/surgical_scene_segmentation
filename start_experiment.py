from __future__ import annotations

import hydra
import torch

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

from src.utils.repro import set_seed
from src.utils.wandb_logging import WandbLogger

from src.datasets.transformation.factory import TransformFactory
from src.datasets.transformation.transform_dataset import TransformDataset


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    # setup
    set_seed(cfg.experiment.seed)

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    experiment_name = cfg.logging.experiment_name

    if experiment_name is None:
        experiment_name = (
            f"{cfg.data.name}_"
            f"{cfg.model.name}_"
            f"seed{cfg.experiment.seed}"
        )

    logger = WandbLogger(
        enabled=cfg.logging.wandb.enabled,
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=experiment_name,
        config=dict(cfg),
        tags=list(cfg.logging.tags),
        mode=cfg.logging.wandb.mode,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT")
    print("=" * 60)
    print(f"Dataset : {cfg.data.name}")
    print(f"Model   : {cfg.model.name}")
    print(f"Device  : {device}")
    print("=" * 60)

    try:

        # Dataset

        dataset = instantiate(cfg.data.dataset)

        print("\nDataset loaded.")


        train_dataset, val_dataset, test_dataset =  dataset.create_official_split()

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples  : {len(val_dataset)}")
        print(f"Test samples : {len(test_dataset)}")

        # Transforms
        transform_factory = TransformFactory(cfg)

        train_transform = transform_factory.build_train_transform()
        val_transform = transform_factory.build_val_transform()
        test_transform = transform_factory.build_test_transform()


        train_dataset = TransformDataset(
            train_dataset,
            transform=train_transform,
        )

        val_dataset = TransformDataset(
            val_dataset,
            transform=val_transform,
        )

        test_dataset = TransformDataset(
            test_dataset,
            transform=test_transform,
        )

        # DataLoaders
        train_loader = None
        val_loader = None
        test_loader = None

        if train_dataset is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_memory,
            )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_memory,
            )

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_memory,
            )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches  : {len(val_loader)}")
        print(f"Test batches : {len(test_loader)}")

        # Model
        model = instantiate(cfg.model.architecture).to(device)

        # Train config
        optimizer = instantiate(
            cfg.trainer.optimizer,
            params=model.parameters(),
        )

        scheduler = instantiate(
            cfg.trainer.scheduler,
            optimizer=optimizer,
        )

        loss_fn = instantiate(
            cfg.trainer.loss
        )

        scaler = instantiate(
            cfg.trainer.scaler
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )

        # =====================================================
        # Training/validation
        # =====================================================

        train_results = trainer.fit()


        checkpoint = torch.load(
            train_results["best_checkpoint"],
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # =====================================================
        # Test
        # =====================================================

        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )

        test_results = evaluator.evaluate()

        # =====================================================
        # Logging
        # =====================================================

        logger.log_final_results(
            train_results,
            test_results,
        )

    finally:

        logger.finish()


    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()