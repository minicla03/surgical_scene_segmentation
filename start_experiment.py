from __future__ import annotations

import time
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.engine.trainer import Trainer
#from src.engine.evaluator import Evaluator
from src.utils.repro import set_seed, seed_worker
from src.utils.wandb_logging import WandbLogger
from src.datasets.transformation.factory import TransformFactory
from src.datasets.transformation.transform_dataset import TransformDataset
from src.metrics.metric_manager import MetricManager

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    experiment_start = time.time()

    def log_step(step, message):
        elapsed = time.time() - experiment_start
        print(f"[{elapsed:8.2f}s] [STEP {step}] {message}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset : {cfg.data.name}", flush=True)
    print(f"Model   : {cfg.model.name}", flush=True)
    print(f"Device  : {cfg.device}", flush=True)
    print("=" * 60, flush=True)

    try:
        log_step(1, "Setting random seed...")
        set_seed(cfg.experiment.seed)
        generator = torch.Generator()
        generator.manual_seed(cfg.experiment.seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        log_step(1, f"Setup completed. Device: {device}")

        log_step(2, "Initializing WandB logger...")

        logger = WandbLogger(
            enabled=cfg.logging.wandb.enabled,
            project=cfg.logging.wandb.project,
            name=cfg.logging.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.logging.tags),
            mode=cfg.logging.wandb.mode,
        )
        log_step(2, "WandB logger initialized.")

        log_step(3, "Instantiating dataset...")
        dataset = instantiate(cfg.data.dataset)

        log_step(3, "Dataset instantiated.")

        log_step(3, "Creating train/validation/test split...")

        split_cfg = cfg.data.split

        split_kwargs = {
            "strategy": split_cfg.strategy,
            "group_by": split_cfg.group_by,
        }

        if split_cfg.strategy == "percentage":
            split_kwargs.update(
                {
                    "train_ratio": split_cfg.train_ratio,
                    "val_ratio": split_cfg.val_ratio,
                    "test_ratio": split_cfg.test_ratio,
                    "seed": split_cfg.get(
                        "seed",
                        cfg.experiment.seed,
                    ),
                    "stratified": split_cfg.get(
                        "stratified",
                        False,
                    ),
                }
            )

        train_dataset, val_dataset, test_dataset = dataset.create_split(**split_kwargs)

        log_step(3, "Dataset split completed.")

        log_step(4, "Building transforms...")
        transform_factory = TransformFactory(cfg)
        train_transform = transform_factory.build_train_transform()
        val_transform = transform_factory.build_val_transform()
        test_transform = transform_factory.build_test_transform() if test_dataset is not None else None
        log_step(4, "Transforms built.")

        log_step(5, "Wrapping datasets with TransformDataset...")
        
        log_step(5, "Wrapping datasets with TransformDataset...")

        if train_dataset is not None:
            train_dataset = TransformDataset(
                train_dataset,
                transform=train_transform,
            )

        if val_dataset is not None:
            val_dataset = TransformDataset(
                val_dataset,
                transform=val_transform,
            )

        if test_dataset is not None:
            test_dataset = TransformDataset(
                test_dataset,
                transform=test_transform,
            )

        log_step(5, "Datasets wrapped.")

        log_step(6, "Creating DataLoaders...")
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
                worker_init_fn=seed_worker,
                generator=generator,
            )

        if val_dataset is not None:

            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_memory,
                worker_init_fn=seed_worker,
                generator=generator,
            )

        if test_dataset is not None:

            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_memory,
                worker_init_fn=seed_worker,
                generator=generator,
            )
        log_step(6, "DataLoaders created.")

        if train_loader is not None:
            print(f"           Train batches: {len(train_loader)}", flush=True)
        if val_loader is not None:
            print(f"           Val batches  : {len(val_loader)}", flush=True)
        if test_loader is not None:
            print(f"           Test batches : {len(test_loader)}", flush=True)

        log_step(7, "Building model...")
        model = instantiate(cfg.model.architecture).to(device)
        log_step(7, "Model initialized.")
        logger.log_model_info(model, model_name=cfg.model.name)

        log_step(8, "Building optimizer, scheduler, loss...")
        optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
        loss_fn = instantiate(cfg.train.loss)
        log_step(8, "Optimization components initialized.")

        log_step(9, "Initializing AMP scaler...")
        scaler = None

        if "scaler" in cfg.train:
            scaler = instantiate(cfg.train.scaler)
        
        log_step(9, "AMP scaler initialized.")

        log_step(10, "Initializing metrics...")
        metrics = {
            name: instantiate(metric_cfg)
            for name, metric_cfg in cfg.evaluation.metrics.items()
        }

        if hasattr(dataset, "CLASS_NAMES"):
            class_names = {i: name for i, name in enumerate(dataset.CLASS_NAMES)}
        else:
            raise ValueError(f"CLASS_NAMES attribute is missing in dataset {cfg.data.name}")

        num_classes = dataset.num_classes
        
        metric_manager = MetricManager(
            metrics=metrics,
            num_classes=num_classes,
            dataset_name=cfg.data.name,
            class_names=class_names,
        )

        log_step(10, "Metrics initialized.")

        log_step(11, "Initializing Trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            metric_manager=metric_manager,
            cfg=cfg,
            device=device,
            logger=logger,
        )
        log_step(11, "Trainer initialized.")

        log_step(12, "Starting training...")
        train_results = trainer.fit()
        log_step(12, "Training completed.")

        log_step(13, "Loading best checkpoint...")
        checkpoint = torch.load(train_results["best_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step(13, "Best checkpoint loaded.")


        #if test_loader is not None:
        #
        #     log_step(14, "Starting final test evaluation...")
        #
        #     test_metrics = MetricManager(
        #         metrics={
        #             name: instantiate(metric_cfg)
        #             for name, metric_cfg
        #             in cfg.evaluation.metrics.items()
        #         },
        #         num_classes=cfg.data.num_classes,
        #         dataset_name=cfg.data.name,
        #         class_names=class_names,
        #     )
        #
        #     evaluator = Evaluator(
        #         model=model,
        #         data_loader=test_loader,
        #         metric_manager=test_metrics,
        #         device=device,
        #     )
        #
        #     test_results = evaluator.evaluate()
        #
        #     logger.log_final_results(
        #         train_results,
        #         test_results,
        #     )
        #
        #     log_step(14, "Final test evaluation completed.")

        log_step(14, "Logging final results...")
        logger.log_final_results(train_results, {})
        log_step(14, "Experiment completed.")

    finally:
        logger.finish()
        elapsed = time.time() - experiment_start
        print(f"\nTotal execution time: {elapsed:.2f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("FINAL TEST RESULTS", flush=True)
    print("=" * 60, flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()