from __future__ import annotations

import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.transformation.factory import TransformFactory
from src.datasets.transformation.transform_dataset import TransformDataset
from src.engine.tester import Tester
from src.engine.trainer import Trainer
from src.metrics.metric_manager import MetricManager
from src.utils.checkpoint import format_lr_str, load_checkpoint_file, resolve_resume_context
from src.utils.repro import seed_worker, set_seed
from src.utils.wandb_logging import WandbLogger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    experiment_start = time.time()
    logger = None

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

    # K-fold: every fold is a separate process/run. Give each fold its own checkpoint
    # directory, W&B name and tag, and a common W&B group, so runs never collide.
    split_cfg = cfg.data.split
    kfold_enabled = split_cfg.strategy == "kfold"
    fold_index = int(split_cfg.get("fold_index", 0)) if kfold_enabled else None
    run_name = cfg.logging.experiment_name
    run_group = cfg.logging.wandb.get("group")
    run_tags = list(cfg.logging.tags)
    if kfold_enabled:
        n_folds = int(split_cfg.get("n_folds", 4))
        if not 0 <= fold_index < n_folds:
            raise ValueError(f"data.split.fold_index must be in [0, {n_folds - 1}], got {fold_index}.")
        cfg.output_dir = f"{cfg.output_dir}/fold{fold_index}"
        # the learning rate is part of the name: LR sweeps must not share a group
        lr_tag = f"lr{format_lr_str(cfg.train.optimizer.lr)}"
        run_group = run_group or f"{run_name}_{lr_tag}_cv"
        run_name = f"{run_name}_{lr_tag}_fold{fold_index}"
        run_tags += ["kfold", f"fold:{fold_index}"]
        print(f"K-fold  : fold {fold_index}/{n_folds - 1} | run: {run_name} | group: {run_group}", flush=True)

    # Seed runs: the official split is fixed and the variance is estimated with several training
    # seeds. Same idea as the folds: own checkpoint directory, W&B name/tag, common group.
    seed_runs = bool(cfg.experiment.get("seed_runs", False))
    if seed_runs:
        if kfold_enabled:
            raise ValueError("experiment.seed_runs cannot be combined with data.split.strategy=kfold.")
        seed = int(cfg.experiment.seed)
        cfg.output_dir = f"{cfg.output_dir}/seed{seed}"
        lr_tag = f"lr{format_lr_str(cfg.train.optimizer.lr)}"
        run_group = run_group or f"{run_name}_{lr_tag}_seeds"
        run_name = f"{run_name}_{lr_tag}_seed{seed}"
        run_tags += ["seeds", f"seed:{seed}"]
        print(f"Seeds   : seed {seed} | run: {run_name} | group: {run_group}", flush=True)

    try:
        # ==============================================================
        # 1. Setup
        # ==============================================================
        log_step(1, "Setting random seed...")
        set_seed(cfg.experiment.seed)

        generator = torch.Generator()
        generator.manual_seed(cfg.experiment.seed)

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        log_step(1, f"Setup completed. Device: {device}")

        # ==============================================================
        # 1.5. Resume resolution (before WandB so the same run can resume)
        # ==============================================================
        resume_ctx = resolve_resume_context(cfg)
        if resume_ctx.checkpoint_path is not None:
            ckpt_epoch = (
                resume_ctx.preloaded_checkpoint.get("epoch")
                if resume_ctx.preloaded_checkpoint is not None
                else "?"
            )
            log_step(
                1,
                f"Resume resolved ({resume_ctx.source}): {resume_ctx.checkpoint_path} "
                f"(epoch={ckpt_epoch}, wandb_run_id={resume_ctx.wandb_run_id})",
            )

        # ==============================================================
        # 2. WandB
        # ==============================================================
        log_step(2, "Initializing WandB logger...")
        wandb_run_id = resume_ctx.wandb_run_id if resume_ctx.resume_wandb else None
        logger = WandbLogger(
            enabled=cfg.logging.wandb.enabled,
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.get("entity"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=run_tags,
            mode=cfg.logging.wandb.mode,
            run_id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
            job_type="train",
            group=run_group,
            notes=cfg.logging.wandb.get("notes"),
        )
        log_step(2, "WandB logger initialized.")

        # ==============================================================
        # 3. Dataset
        # ==============================================================
        log_step(3, "Instantiating dataset...")
        dataset = instantiate(cfg.data.dataset)
        log_step(3, "Dataset instantiated.")

        # --------------------------------------------------------------
        # Dataset split
        # --------------------------------------------------------------
        log_step(3, "Creating train/validation/test split...")
        split_kwargs = {
            "strategy": split_cfg.strategy,
            "group_by": split_cfg.group_by,
        }

        if split_cfg.strategy == "percentage":
            split_kwargs.update({
                "train_ratio": split_cfg.train_ratio,
                "val_ratio": split_cfg.val_ratio,
                "test_ratio": split_cfg.test_ratio,
                "seed": split_cfg.get("seed", cfg.experiment.seed),
                "stratified": split_cfg.get("stratified", False),
            })

        elif split_cfg.strategy == "kfold":
            split_kwargs.update({
                "n_folds": split_cfg.get("n_folds", 4),
                "fold_index": fold_index,
                "test_ratio": split_cfg.get("test_ratio", 0.0),
                # fixed split seed (NOT the training seed): all folds must share the same partition
                "seed": split_cfg.get("seed", 42),
                "balance": split_cfg.get("balance", True),
                "n_restarts": split_cfg.get("n_restarts", 30),
                "balance_min_pixel_pct": split_cfg.get("balance_min_pixel_pct", 1.0),
            })

        train_dataset, val_dataset, test_dataset = dataset.create_split(**split_kwargs)
        log_step(3, "Dataset split completed.")

        kfold_plan = getattr(dataset, "kfold_plan", None) if kfold_enabled else None
        if kfold_plan is not None:
            logger.log_kfold_plan(kfold_plan, fold_index, group=run_group)

        # ==============================================================
        # 4. Transforms
        # ==============================================================
        log_step(4, "Building transforms...")
        transform_factory = TransformFactory(cfg)

        train_transform = transform_factory.build_train_transform()
        val_transform = transform_factory.build_val_transform()
        test_transform = transform_factory.build_test_transform() if test_dataset is not None else None
        log_step(4, "Transforms built.")

        # ==============================================================
        # 5. TransformDataset
        # ==============================================================
        log_step(5, "Wrapping datasets with TransformDataset...")
        if train_dataset is not None:
            train_dataset = TransformDataset(train_dataset, transform=train_transform)
        if val_dataset is not None:
            val_dataset = TransformDataset(val_dataset, transform=val_transform)
        if test_dataset is not None:
            test_dataset = TransformDataset(test_dataset, transform=test_transform)
        log_step(5, "Datasets wrapped.")

        # ==============================================================
        # 6. DataLoaders
        # ==============================================================
        log_step(6, "Creating DataLoaders...")
        train_loader, val_loader, test_loader = None, None, None

        num_workers = int(cfg.train.get("num_workers", 0))
        pin_memory = bool(cfg.train.get("pin_memory", True))
        persistent_workers = bool(cfg.train.get("persistent_workers", num_workers > 0)) if num_workers > 0 else False
        multiprocessing_context = cfg.train.get("multiprocessing_context", "spawn") if num_workers > 0 else None

        dataloader_kwargs = {
            "batch_size": cfg.train.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = persistent_workers
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context

        if train_dataset is not None:
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                **dataloader_kwargs,
            )
            print(f"           Train batches: {len(train_loader)}", flush=True)

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **dataloader_kwargs,
            )
            print(f"           Val batches  : {len(val_loader)}", flush=True)

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                **dataloader_kwargs,
            )
            print(f"           Test batches : {len(test_loader)}", flush=True)
        else:
            print("           Test loader  : disabled", flush=True)
        log_step(6, "DataLoaders created.")

        # ==============================================================
        # 7. Model
        # ==============================================================
        log_step(7, "Building model...")
        model = instantiate(cfg.model.architecture).to(device)
        log_step(7, "Model initialized.")
        
        logger.log_model_info(model, model_name=cfg.model.name)

        # ==============================================================
        # 8. Optimizer, scheduler, loss
        # ==============================================================
        log_step(8, "Building optimizer, scheduler, loss...")
        optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
        loss_fn = instantiate(cfg.train.loss)
        log_step(8, "Optimization components initialized.")

        # ==============================================================
        # 9. AMP scaler
        # ==============================================================
        log_step(9, "Initializing AMP scaler...")
        scaler = instantiate(cfg.train.scaler) if "scaler" in cfg.train else None
        log_step(9, "AMP scaler initialized.")

        # ==============================================================
        # 10. Metrics
        # ==============================================================
        log_step(10, "Initializing metrics...")
        metrics = {name: instantiate(metric_cfg) for name, metric_cfg in cfg.evaluation.metrics.items()}

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
            evaluable_classes=(
                kfold_plan.evaluable_classes(fold_index) if kfold_plan is not None else None
            ),
        )
        log_step(10, "Metrics initialized.")

        # ==============================================================
        # 11. Trainer
        # ==============================================================
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
            dataloader_generator=generator,
        )
        log_step(11, "Trainer initialized.")

        if resume_ctx.checkpoint_path is not None:
            log_step(11, f"Restoring training state from {resume_ctx.checkpoint_path}...")
            trainer.load_resume_checkpoint(
                checkpoint_path=resume_ctx.checkpoint_path,
                preloaded_checkpoint=resume_ctx.preloaded_checkpoint,
            )
            resume_ctx.preloaded_checkpoint = None

        # ==============================================================
        # 12. Training
        # ==============================================================
        log_step(12, "Starting training...")
        train_results = trainer.fit()
        log_step(12, "Training completed.")

        # ==============================================================
        # 13. Final evaluation with the full metric set (incl. HD95) on the validation set:
        #     first the LAST weights, then the BEST checkpoint. These are the values reported
        #     as final/last/* and final/best/* (identical protocol for every model).
        # ==============================================================
        # The test set (when the split has one, e.g. DSAD official) is evaluated with the same
        # weights; keys final/test_last/* and final/test_best/*. Not supported together with the
        # K-fold, where the evaluable classes of the test set would differ from the validation ones.
        evaluate_test = test_loader is not None and kfold_plan is None
        if test_loader is not None and not evaluate_test:
            log_step(13, "Test set not evaluated with K-fold (evaluable classes differ from validation).")

        def final_evaluation(tag: str) -> None:
            targets = [("", "validation", val_loader)]
            if evaluate_test:
                targets.append(("test_", "test", test_loader))
            for prefix, name, loader in targets:
                if loader is None:
                    continue
                log_step(13, f"Final evaluation ({tag} weights, {name} set, all metrics)...")
                loss, metrics_out, per_class_out, _ = trainer.validate(
                    max(trainer.last_epoch, 1), extra=True, loader=loader
                )
                metrics_out["loss"] = loss
                train_results[f"{prefix}{tag}_metrics"] = metrics_out
                train_results[f"{prefix}{tag}_per_class"] = per_class_out

        final_evaluation("last")

        log_step(13, "Loading best checkpoint...")
        best_ckpt_path = Path(train_results["best_checkpoint"])
        if best_ckpt_path.is_file():
            checkpoint = load_checkpoint_file(best_ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            log_step(13, f"Best checkpoint loaded from {best_ckpt_path}.")
            final_evaluation("best")
        else:
            log_step(13, f"Best checkpoint not found at {best_ckpt_path}; keeping current weights.")

        # ==============================================================
        # 14. Test results
        # ==============================================================
        # The test set is evaluated in step 13 (final_evaluation): final/test_last/*, final/test_best/*.
        test_results = {}

        # ==============================================================
        # 15. Final results
        # ==============================================================
        log_step(15, "Logging final results...")
        logger.log_final_results(train_results, test_results)
        log_step(15, "Experiment completed.")

        # ==============================================================
        # Final summary
        # ==============================================================
        print("\n" + "=" * 60, flush=True)

        for tag in ("best", "last"):
            for prefix, title in (("", "VALIDATION"), ("test_", "TEST")):
                final_metrics = train_results.get(f"{prefix}{tag}_metrics")
                if not final_metrics:
                    continue
                print(f"FINAL {title} RESULTS ({tag} weights)", flush=True)
                print("=" * 60, flush=True)
                for name, value in final_metrics.items():
                    print(f"{name:20s}: {value:.4f}", flush=True)
                print("=" * 60, flush=True)

        if not any(train_results.get(f"test_{tag}_metrics") for tag in ("best", "last")):
            print("TEST EVALUATION NOT AVAILABLE", flush=True)

        print("=" * 60, flush=True)

    finally:
        if logger is not None:
            logger.finish()

        elapsed = time.time() - experiment_start
        print(f"\nTotal execution time: {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()