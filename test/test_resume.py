"""
Tests for end-of-epoch checkpointing, resume continuity, legacy compatibility,
and W&B run/artifact resume integration.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.engine.trainer import Trainer
from src.metrics.metric_manager import MetricManager
from src.utils.checkpoint import (
    atomic_torch_save,
    build_checkpoint_paths,
    build_epoch_checkpoint_path,
    load_checkpoint_file,
    resolve_checkpoint_dir,
    resolve_resume_context,
)
from src.utils.repro import seed_worker, set_seed
from src.utils.wandb_logging import WandbLogger


class DeterministicSegDataset(Dataset):
    """Deterministic synthetic dataset for repeatable training/resume tests."""

    def __init__(self, num_samples: int = 6, num_classes: int = 4, size=(16, 16), seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        h, w = size
        self.images = torch.randn(num_samples, 3, h, w, generator=gen, dtype=torch.float32)
        self.masks = torch.randint(0, num_classes, (num_samples, 1, h, w), generator=gen, dtype=torch.long)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx], "mask": self.masks[idx]}


class TinySegNet(torch.nn.Module):
    """Minimal 2D segmentation network."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonImprovingLoss(torch.nn.Module):
    """Loss that makes val loss rise after epoch 1 to exercise early stopping."""

    def __init__(self, trainer_ref: dict[str, Trainer]):
        super().__init__()
        self.trainer_ref = trainer_ref
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        target = masks.squeeze(1)
        base = self.ce(logits, target)
        trainer = self.trainer_ref.get("trainer")
        if trainer is not None and not trainer.model.training:
            # Make validation loss monotonically worse after epoch 1
            step_offset = len(trainer.history) + 1
            if step_offset > 1:
                return base.detach() * 0.0 + float(step_offset)
            return base.detach() * 0.0 + 0.5
        return base


def _build_components(
    tmp_path: Path,
    max_epochs: int = 4,
    early_stopping_enabled: bool = False,
    patience: int = 3,
    save_every_n_epochs: int | None = None,
    seed: int = 42,
    logger=None,
    loss_fn=None,
):
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    num_classes = 4
    train_ds = DeterministicSegDataset(num_samples=6, num_classes=num_classes, seed=10)
    val_ds = DeterministicSegDataset(num_samples=4, num_classes=num_classes, seed=20)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = TinySegNet(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    class SqueezingLoss(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, preds, masks):
            return self.inner(preds, masks.squeeze(1) if masks.ndim == 4 else masks)

    wrapped_loss = loss if isinstance(loss, NonImprovingLoss) else SqueezingLoss(loss)

    metric_manager = MetricManager(
        metrics={},
        num_classes=num_classes,
        class_names={i: f"cls_{i}" for i in range(num_classes)},
    )

    cfg = OmegaConf.create({
        "data": {"name": "sarrarp50", "num_classes": num_classes},
        "model": {"name": "U-Net"},
        "output_dir": str(tmp_path),
        "train": {
            "max_epochs": max_epochs,
            "accumulation_steps": 1,
            "eval_freq": 1,
            "precision": "32",
            "optimizer": {"lr": 1e-3},
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "patience": patience,
                "monitor": "loss",
                "mode": "min",
            },
        },
        "checkpoint": {
            "save_last": True,
            "save_every_n_epochs": save_every_n_epochs,
            "upload_best_to_wandb": False,
            "upload_last_to_wandb": False,
        },
        "logging": {"qualitative": {"enabled": False}},
    })

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=wrapped_loss,
        scaler=None,
        metric_manager=metric_manager,
        cfg=cfg,
        device=torch.device("cpu"),
        logger=logger,
        dataloader_generator=generator,
    )
    return trainer


class TestCheckpointAndResume:
    def test_uninterrupted_vs_resumed_training_matches_exactly(self, tmp_path: Path):
        full_dir = tmp_path / "full"
        resumed_dir = tmp_path / "resumed"

        # 1. Uninterrupted 4-epoch baseline run
        full_trainer = _build_components(full_dir, max_epochs=4)
        full_results = full_trainer.fit()

        # 2. Phase A: run 2 epochs and stop
        part_a = _build_components(resumed_dir, max_epochs=2)
        # Keep T_max=4 on scheduler to match the 4-epoch schedule
        part_a.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(part_a.optimizer, T_max=4)
        part_a.fit()

        paths = build_checkpoint_paths(resumed_dir, "sarrarp50", "U-Net", 1e-3)
        assert paths["last"].is_file()
        assert not any(resumed_dir.glob(".*.tmp"))

        last_ckpt = load_checkpoint_file(paths["last"])
        assert last_ckpt["epoch"] == 2
        assert last_ckpt["scheduler_stepped"] is True
        assert len(last_ckpt["history"]) == 2
        assert "rng_state" in last_ckpt
        assert "epochs_without_improvement" in last_ckpt

        # 3. Phase B: resume from epoch 2 up to epoch 4
        part_b = _build_components(resumed_dir, max_epochs=4)
        next_epoch = part_b.load_resume_checkpoint(paths["last"])
        assert next_epoch == 3
        assert part_b.scheduler.last_epoch == 2

        resumed_results = part_b.fit()

        assert len(resumed_results["history"]) == len(full_results["history"]) == 4
        assert resumed_results["best_epoch"] == full_results["best_epoch"]
        assert resumed_results["best_score"] == pytest.approx(full_results["best_score"], rel=1e-6, abs=1e-7)

        for full_ep, res_ep in zip(full_results["history"], resumed_results["history"]):
            assert full_ep["epoch"] == res_ep["epoch"]
            assert full_ep["train"]["loss"] == pytest.approx(res_ep["train"]["loss"], rel=1e-6, abs=1e-6)
            assert full_ep["val"]["loss"] == pytest.approx(res_ep["val"]["loss"], rel=1e-6, abs=1e-6)

        for name, full_tensor in full_trainer.model.state_dict().items():
            assert torch.allclose(full_tensor, part_b.model.state_dict()[name], atol=1e-6)

        assert part_b.optimizer.param_groups[0]["lr"] == pytest.approx(
            full_trainer.optimizer.param_groups[0]["lr"], rel=1e-9
        )
        assert part_b.scheduler.last_epoch == full_trainer.scheduler.last_epoch == 4

    def test_resume_from_legacy_best_checkpoint_aligns_scheduler(self, tmp_path: Path):
        trainer_src = _build_components(tmp_path / "src", max_epochs=6)
        # Simulate 3 completed epochs saved in legacy _update_best style (before scheduler.step at epoch 3)
        for _ in range(2):
            trainer_src.scheduler.step()
        legacy_path = tmp_path / "legacy_best.pt"
        atomic_torch_save(
            {
                "epoch": 3,
                "model_state_dict": trainer_src.model.state_dict(),
                "optimizer_state_dict": trainer_src.optimizer.state_dict(),
                "scheduler_state_dict": trainer_src.scheduler.state_dict(),
                "best_score": 0.42,
                "best_epoch": 3,
                "best_metrics": {"loss": 0.42},
                "metrics": {"loss": 0.42},
                "config": trainer_src.cfg,
            },
            legacy_path,
        )

        # Reference scheduler stepped 3 times (end of epoch 3)
        ref_trainer = _build_components(tmp_path / "ref", max_epochs=6)
        for _ in range(3):
            ref_trainer.scheduler.step()
        expected_lr = ref_trainer.optimizer.param_groups[0]["lr"]

        resumed_dir = tmp_path / "resumed"
        trainer_dst = _build_components(resumed_dir, max_epochs=4)
        next_epoch = trainer_dst.load_resume_checkpoint(legacy_path)

        assert next_epoch == 4
        assert trainer_dst.scheduler.last_epoch == 3
        assert trainer_dst.epochs_without_improvement == 0
        assert trainer_dst.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-9)
        assert trainer_dst.best_checkpoint_path.is_file()

        results = trainer_dst.fit()
        assert results["history"][-1]["epoch"] == 4
        assert trainer_dst.scheduler.last_epoch == 4

    def test_early_stopping_counter_persists_across_resume(self, tmp_path: Path):
        ref_a: dict[str, Trainer] = {}
        loss_a = NonImprovingLoss(ref_a)
        trainer_a = _build_components(
            tmp_path / "es",
            max_epochs=2,
            early_stopping_enabled=True,
            patience=3,
            loss_fn=loss_a,
        )
        ref_a["trainer"] = trainer_a
        trainer_a.fit()

        # Epoch 1 improved (best=0.5), Epoch 2 worsened -> epochs_without_improvement == 1
        assert trainer_a.epochs_without_improvement == 1

        ref_b: dict[str, Trainer] = {}
        loss_b = NonImprovingLoss(ref_b)
        trainer_b = _build_components(
            tmp_path / "es",
            max_epochs=10,
            early_stopping_enabled=True,
            patience=3,
            loss_fn=loss_b,
        )
        ref_b["trainer"] = trainer_b
        trainer_b.load_resume_checkpoint(trainer_a.last_checkpoint_path)
        assert trainer_b.epochs_without_improvement == 1

        results = trainer_b.fit()
        # Patience=3: epochs 2, 3, 4 fail to improve -> stops at epoch 4
        assert [ep["epoch"] for ep in results["history"]] == [1, 2, 3, 4]
        assert trainer_b.epochs_without_improvement == 3

    def test_periodic_checkpoints_and_already_finished_resume(self, tmp_path: Path):
        trainer = _build_components(tmp_path, max_epochs=4, save_every_n_epochs=2)
        trainer.fit()

        ep2 = build_epoch_checkpoint_path(tmp_path, "sarrarp50", "U-Net", 1e-3, 2)
        ep4 = build_epoch_checkpoint_path(tmp_path, "sarrarp50", "U-Net", 1e-3, 4)
        assert ep2.is_file()
        assert ep4.is_file()

        trainer2 = _build_components(tmp_path, max_epochs=4, save_every_n_epochs=2)
        next_epoch = trainer2.load_resume_checkpoint(ep4)
        assert next_epoch == 5
        results = trainer2.fit()
        assert len(results["history"]) == 4

    def test_resolve_resume_context_local_and_wandb_download(self, tmp_path: Path, monkeypatch):
        base_cfg = OmegaConf.create({
            "data": {"name": "sarrarp50"},
            "model": {"name": "U-Net"},
            "output_dir": str(tmp_path),
            "train": {"optimizer": {"lr": 1e-3}},
            "logging": {"wandb": {"project": "surgical_scene_segmentation", "entity": "my_entity"}},
            "resume": {
                "enabled": False,
                "auto": True,
                "checkpoint_path": None,
                "wandb_run_id": None,
                "wandb_artifact": None,
                "epoch": None,
                "resume_wandb": True,
            },
        })

        # 1. auto=true without local file -> fresh start
        ctx_none = resolve_resume_context(base_cfg)
        assert ctx_none.checkpoint_path is None
        assert ctx_none.source == "none"

        # 2. auto=true with local last file carrying wandb_run_id
        last_path = build_checkpoint_paths(tmp_path, "sarrarp50", "U-Net", 1e-3)["last"]
        atomic_torch_save({"epoch": 5, "model_state_dict": {}, "wandb_run_id": "run_local_1"}, last_path)
        ctx_auto = resolve_resume_context(base_cfg)
        assert ctx_auto.checkpoint_path == last_path
        assert ctx_auto.source == "local_auto"
        assert ctx_auto.wandb_run_id == "run_local_1"

        # 3. Remote W&B download by wandb_run_id & epoch (with legacy metadata-less fallback)
        remote_store = tmp_path / "remote_store"
        remote_store.mkdir()

        class FakeArtifact:
            def __init__(self, name: str, version: str, epoch: int, created_at: str, with_meta: bool):
                self.name = f"{name}:{version}"
                self.version = version
                self.type = "model"
                self.aliases = []
                self.created_at = created_at
                self.metadata = {"epoch": epoch, "kind": "best"} if with_meta else {}
                self._epoch = epoch

            def download(self, root: str) -> str:
                d = Path(root)
                d.mkdir(parents=True, exist_ok=True)
                atomic_torch_save({"epoch": self._epoch, "model_state_dict": {}}, d / f"ckpt_{self._epoch}.pt")
                return str(d)

            def logged_by(self):
                return SimpleNamespace(id="run_remote_99")

        fake_run = SimpleNamespace(
            logged_artifacts=lambda: [
                FakeArtifact("model_lr_1e-03", "v0", epoch=1, created_at="2026-11-10T01:00:00Z", with_meta=False),
                FakeArtifact("model_lr_8e-04", "v1", epoch=17, created_at="2026-11-10T05:00:00Z", with_meta=False),
            ]
        )

        class FakeApi:
            def run(self, path: str):
                assert path == "my_entity/surgical_scene_segmentation/run_remote_99"
                return fake_run

        fake_wandb = SimpleNamespace(Api=FakeApi)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        wandb_cfg = OmegaConf.merge(
            base_cfg,
            {
                "resume": {
                    "auto": False,
                    "wandb_run_id": "run_remote_99",
                    "epoch": 17,
                }
            },
        )
        ctx_wb = resolve_resume_context(wandb_cfg)
        assert ctx_wb.source == "wandb_run"
        assert ctx_wb.wandb_run_id == "run_remote_99"
        assert ctx_wb.preloaded_checkpoint["epoch"] == 17
        assert ctx_wb.checkpoint_path is not None and ctx_wb.checkpoint_path.is_file()

    def test_wandb_logger_resumes_same_run_and_respects_upload_flags(self, tmp_path: Path, monkeypatch):
        calls: dict[str, list] = {"init": [], "config_update": [], "log": [], "log_artifact": []}

        class FakeArtifact:
            def __init__(self, name, type, metadata=None):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files = []

            def add_file(self, path):
                self.files.append(path)

        class FakeWandb:
            def __init__(self):
                self.run = None
                self.config = SimpleNamespace(
                    update=lambda cfg, allow_val_change=False: calls["config_update"].append(
                        (cfg, allow_val_change)
                    )
                )

            def init(self, **kwargs):
                calls["init"].append(kwargs)
                self.run = SimpleNamespace(id=kwargs.get("id", "new_id"), resumed=True)
                return self.run

            def log(self, data, **kwargs):
                calls["log"].append((data, kwargs))

            def Artifact(self, name, type, metadata=None):
                return FakeArtifact(name, type, metadata)

            def log_artifact(self, artifact, aliases=None):
                calls["log_artifact"].append((artifact, aliases))

            def finish(self):
                self.run = None

        fake_wandb = FakeWandb()
        import src.utils.wandb_logging as wl_mod

        monkeypatch.setattr(wl_mod, "wandb", fake_wandb)

        logger = WandbLogger(
            enabled=True,
            project="surgical_scene_segmentation",
            entity="my_entity",
            name="sarrarp50_U-Net",
            config={"train": {"max_epochs": 50}},
            tags=["sarrarp50"],
            mode="online",
            run_id="m0sa4rfx",
            resume="must",
        )

        assert logger.run_id == "m0sa4rfx"
        assert logger.resumed is True
        assert calls["init"][0]["id"] == "m0sa4rfx"
        assert calls["init"][0]["resume"] == "must"
        assert "config" not in calls["init"][0]
        assert calls["config_update"][0] == ({"train": {"max_epochs": 50}}, True)

        # Should skip wandb.log(metrics) without step when resumed=True
        logger.log_model_info(TinySegNet(), model_name="U-Net")
        assert len(calls["log"]) == 0

        # Epoch logging resumes at step=18
        logger.log_epoch(epoch=18, train_metrics={"loss": 0.25}, val_metrics={"loss": 0.33})
        assert calls["log"][0][1]["step"] == 18
        assert calls["log"][0][1]["commit"] is True

        # Checkpoint upload=False saves locally without log_artifact
        local_last = tmp_path / "last.pt"
        logger.log_checkpoint(
            checkpoint={"epoch": 18},
            checkpoint_path=local_last,
            name="sarrarp50_U-Net_lr_1e-3",
            aliases=["epoch-18", "last", "latest"],
            upload_to_wandb=False,
            metadata={"epoch": 18, "kind": "last"},
        )
        assert local_last.is_file()
        assert len(calls["log_artifact"]) == 0

        # Checkpoint upload=True logs artifact with metadata
        local_best = tmp_path / "best.pt"
        logger.log_checkpoint(
            checkpoint={"epoch": 18},
            checkpoint_path=local_best,
            name="sarrarp50_U-Net_lr_1e-3",
            aliases=["epoch-18", "best"],
            upload_to_wandb=True,
            metadata={"epoch": 18, "kind": "best"},
        )
        assert local_best.is_file()
        assert len(calls["log_artifact"]) == 1
        art, aliases = calls["log_artifact"][0]
        assert art.name == "sarrarp50_U-Net_lr_1e-3"
        assert art.metadata == {"epoch": 18, "kind": "best"}
        assert aliases == ["epoch-18", "best"]

    def test_resolve_checkpoint_dir_and_hydra_default(self, tmp_path: Path):
        from hydra import compose, initialize_config_dir

        # Explicit output_dir takes precedence over paths.checkpoint
        cfg_explicit = OmegaConf.create({
            "output_dir": str(tmp_path / "custom"),
            "paths": {"checkpoint": "/work/tesi_cmininno/checkpoint/sarrarp50/U-Net"},
        })
        assert resolve_checkpoint_dir(cfg_explicit) == tmp_path / "custom"

        # Falls back to paths.checkpoint when output_dir is omitted
        cfg_paths_only = OmegaConf.create({
            "paths": {"checkpoint": "/work/tesi_cmininno/checkpoint/sarrarp50/U-Net"},
        })
        assert resolve_checkpoint_dir(cfg_paths_only) == Path("/work/tesi_cmininno/checkpoint/sarrarp50/U-Net")

        # Falls back to ./outputs when neither is set
        assert resolve_checkpoint_dir(OmegaConf.create({})) == Path("./outputs")

        # Hydra project config composes output_dir to /work/tesi_cmininno/checkpoint/<name>/<model.name>
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            hydra_cfg = compose(
                config_name="config",
                overrides=[
                    "dataset=baseline/sarrarp50",
                    "training=baseline/sarrarp50",
                    "model=baseline/unet",
                ],
            )
        assert resolve_checkpoint_dir(hydra_cfg) == Path("/work/tesi_cmininno/checkpoint/sarrarp50/U-Net")

