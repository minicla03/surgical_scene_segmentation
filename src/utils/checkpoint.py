from __future__ import annotations

import inspect
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ResumeContext:
    """Resolved resume state computed prior to WandB and Trainer initialization."""

    checkpoint_path: Path | None = None
    preloaded_checkpoint: dict[str, Any] | None = None
    wandb_run_id: str | None = None
    resume_wandb: bool = True
    source: str = "none"


def format_lr_str(lr: float) -> str:
    """Format learning rate consistently across checkpoint filenames."""
    return f"{float(lr):.0e}".replace("e-0", "e-").replace("e+0", "e+")


def sanitize_artifact_name(name: str) -> str:
    """Sanitize a string so it matches W&B artifact name constraints."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def resolve_checkpoint_dir(cfg: Any) -> Path:
    """Resolve the directory used for saving and resuming checkpoints."""
    output_dir = cfg.get("output_dir") if hasattr(cfg, "get") else getattr(cfg, "output_dir", None)
    if output_dir:
        return Path(str(output_dir))

    paths_cfg = (cfg.get("paths") or {}) if hasattr(cfg, "get") else getattr(cfg, "paths", None)
    ckpt_dir = (
        paths_cfg.get("checkpoint")
        if hasattr(paths_cfg, "get")
        else getattr(paths_cfg, "checkpoint", None)
    )
    if ckpt_dir:
        return Path(str(ckpt_dir))

    return Path("./outputs")


def build_checkpoint_paths(
    output_dir: str | Path,
    dataset_name: str,
    model_name: str,
    lr: float,
) -> dict[str, Path]:
    """Build standard best and last checkpoint paths for an experiment."""
    base_dir = Path(output_dir)
    lr_str = format_lr_str(lr)
    stem = f"{dataset_name}_{model_name}_lr_{lr_str}"
    return {
        "best": base_dir / f"best_{stem}.pt",
        "last": base_dir / f"last_{stem}.pt",
    }


def build_epoch_checkpoint_path(
    output_dir: str | Path,
    dataset_name: str,
    model_name: str,
    lr: float,
    epoch: int,
) -> Path:
    """Build checkpoint path for a specific periodic epoch snapshot."""
    base_dir = Path(output_dir)
    lr_str = format_lr_str(lr)
    return base_dir / f"epoch_{epoch:03d}_{dataset_name}_{model_name}_lr_{lr_str}.pt"


def atomic_torch_save(payload: dict[str, Any], target_path: str | Path) -> Path:
    """Save a checkpoint atomically via a temporary file and os.replace."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint_file(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dictionary onto CPU safely across PyTorch versions."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint at '{path}' to be a dict, got {type(checkpoint).__name__}."
        )
    return checkpoint


def capture_rng_state(
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch, CUDA, and DataLoader RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if dataloader_generator is not None:
        state["dataloader_generator"] = dataloader_generator.get_state()
    return state


def restore_rng_state(
    rng_state: dict[str, Any] | None,
    dataloader_generator: torch.Generator | None = None,
) -> None:
    """Restore saved RNG states if present in a checkpoint."""
    if not rng_state:
        return

    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])

    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])

    if "torch" in rng_state and rng_state["torch"] is not None:
        torch_state = rng_state["torch"].detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(torch_state)

    if "cuda" in rng_state and rng_state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = rng_state["cuda"]
        if len(cuda_states) == torch.cuda.device_count():
            normalized = [s.detach().cpu().to(dtype=torch.uint8) for s in cuda_states]
            torch.cuda.set_rng_state_all(normalized)
        else:
            print(
                f"[Resume] Warning: CUDA device count mismatch "
                f"(checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}); "
                "skipping CUDA RNG restore.",
                flush=True,
            )

    if (
        dataloader_generator is not None
        and "dataloader_generator" in rng_state
        and rng_state["dataloader_generator"] is not None
    ):
        gen_state = rng_state["dataloader_generator"].detach().cpu().to(dtype=torch.uint8)
        dataloader_generator.set_state(gen_state)


def _find_checkpoint_in_dir(directory: Path) -> Path:
    """Locate the checkpoint file inside a downloaded W&B artifact directory."""
    candidates = sorted(
        list(directory.glob("*.pt")) + list(directory.glob("*.pth")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoint file found in '{directory}'.")
    return candidates[0]


def _extract_logged_by_run_id(artifact: Any) -> str | None:
    """Safely extract the source W&B run ID that logged an artifact."""
    try:
        run = artifact.logged_by()
    except Exception:
        return None
    return getattr(run, "id", None) if run is not None else None


def _artifact_sort_key(artifact: Any) -> tuple[int, int, str, int]:
    """Sort key prioritizing higher epoch metadata, kind='last', then creation time/version."""
    metadata = getattr(artifact, "metadata", None) or {}
    raw_epoch = metadata.get("epoch")
    epoch_val = int(raw_epoch) if raw_epoch is not None else -1
    kind = metadata.get("kind")
    aliases = list(getattr(artifact, "aliases", []) or [])
    is_last = 1 if (kind == "last" or "last" in aliases) else 0
    created_at = str(getattr(artifact, "created_at", "") or "")
    version_str = str(getattr(artifact, "version", "") or "")
    version_match = re.search(r"(\d+)$", version_str)
    version_num = int(version_match.group(1)) if version_match else -1
    return (epoch_val, is_last, created_at, version_num)


def _download_artifact_checkpoint(artifact: Any, download_root: Path) -> Path:
    """Download a W&B artifact to a deterministic subfolder and return its .pt path."""
    target_dir = download_root / sanitize_artifact_name(str(artifact.name).replace(":", "_"))
    downloaded_dir = Path(artifact.download(root=str(target_dir)))
    return _find_checkpoint_in_dir(downloaded_dir)


def download_checkpoint_from_wandb(
    download_root: str | Path,
    project: str,
    entity: str | None = None,
    wandb_run_id: str | None = None,
    wandb_artifact: str | None = None,
    target_epoch: int | None = None,
) -> tuple[Path, str | None, dict[str, Any] | None]:
    """Download a checkpoint from W&B via explicit artifact ref or run ID.

    When ``wandb_run_id`` is used, artifacts are inspected directly on the run's
    ``logged_artifacts()`` list so legacy runs whose artifact names collided
    across experiments can still be resolved reliably.
    """
    import wandb

    api = wandb.Api()
    cache_root = Path(download_root)

    if wandb_artifact:
        artifact_ref = wandb_artifact
        if "/" not in artifact_ref:
            prefix = f"{entity}/{project}" if entity else project
            artifact_ref = f"{prefix}/{artifact_ref}"
        if ":" not in artifact_ref:
            suffix = f"epoch-{target_epoch}" if target_epoch is not None else "latest"
            artifact_ref = f"{artifact_ref}:{suffix}"

        artifact = api.artifact(artifact_ref, type="model")
        ckpt_path = _download_artifact_checkpoint(artifact, cache_root)
        source_run_id = _extract_logged_by_run_id(artifact) or wandb_run_id
        return ckpt_path, source_run_id, None

    if not wandb_run_id:
        raise ValueError("Either 'wandb_artifact' or 'wandb_run_id' must be provided.")

    run_path = f"{entity}/{project}/{wandb_run_id}" if entity else f"{project}/{wandb_run_id}"
    run = api.run(run_path)
    model_artifacts = [art for art in run.logged_artifacts() if getattr(art, "type", None) == "model"]

    if not model_artifacts:
        raise RuntimeError(f"No model artifacts found in W&B run '{run_path}'.")

    ordered = sorted(model_artifacts, key=_artifact_sort_key, reverse=True)

    if target_epoch is None:
        chosen = ordered[0]
        ckpt_path = _download_artifact_checkpoint(chosen, cache_root)
        return ckpt_path, wandb_run_id, None

    # 1. Exact match by artifact metadata (written by updated Trainer)
    meta_matches = [
        art
        for art in ordered
        if isinstance(getattr(art, "metadata", None), dict)
        and art.metadata.get("epoch") == target_epoch
    ]
    if meta_matches:
        ckpt_path = _download_artifact_checkpoint(meta_matches[0], cache_root)
        return ckpt_path, wandb_run_id, None

    # 2. Match by artifact alias
    alias_tag = f"epoch-{target_epoch}"
    alias_matches = [
        art for art in ordered if alias_tag in (getattr(art, "aliases", []) or [])
    ]
    if alias_matches:
        ckpt_path = _download_artifact_checkpoint(alias_matches[0], cache_root)
        return ckpt_path, wandb_run_id, None

    # 3. Legacy fallback: inspect downloaded checkpoints in reverse chronological order
    # (necessary when old artifact collections shared names and aliases were overwritten).
    available_epochs: list[int] = []
    for art in ordered:
        ckpt_path = _download_artifact_checkpoint(art, cache_root)
        ckpt = load_checkpoint_file(ckpt_path)
        ckpt_epoch = ckpt.get("epoch")
        if ckpt_epoch is not None:
            available_epochs.append(int(ckpt_epoch))
        if ckpt_epoch == target_epoch:
            return ckpt_path, wandb_run_id, ckpt

    raise ValueError(
        f"Epoch {target_epoch} not found among model artifacts of W&B run '{run_path}'. "
        f"Available epochs: {sorted(set(available_epochs))}"
    )


def resolve_resume_context(cfg: Any) -> ResumeContext:
    """Resolve checkpoint path and W&B run ID to resume before initializing logger."""
    resume_cfg = cfg.get("resume") or cfg.get("train", {}).get("resume") or {}
    enabled = bool(resume_cfg.get("enabled", False))
    auto = bool(resume_cfg.get("auto", False))
    raw_ckpt_path = resume_cfg.get("checkpoint_path")
    wandb_run_id = resume_cfg.get("wandb_run_id")
    wandb_artifact = resume_cfg.get("wandb_artifact")
    target_epoch = resume_cfg.get("epoch")
    target_epoch = int(target_epoch) if target_epoch is not None else None
    resume_wandb = bool(resume_cfg.get("resume_wandb", True))

    if not any([enabled, auto, raw_ckpt_path, wandb_run_id, wandb_artifact]):
        return ResumeContext(resume_wandb=resume_wandb, source="none")

    output_dir = resolve_checkpoint_dir(cfg)
    expected_paths = build_checkpoint_paths(
        output_dir=output_dir,
        dataset_name=str(cfg.data.name),
        model_name=str(cfg.model.name),
        lr=float(cfg.train.optimizer.lr),
    )
    expected_last_path = expected_paths["last"]

    resolved_path: Path | None = None
    preloaded_ckpt: dict[str, Any] | None = None
    source_wandb_run_id: str | None = None
    source = "none"

    if raw_ckpt_path:
        resolved_path = Path(str(raw_ckpt_path))
        preloaded_ckpt = load_checkpoint_file(resolved_path)
        source = "local"
    elif auto and expected_last_path.is_file():
        resolved_path = expected_last_path
        preloaded_ckpt = load_checkpoint_file(resolved_path)
        source = "local_auto"
    elif wandb_artifact or wandb_run_id:
        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        project = wandb_cfg.get("project")
        entity = wandb_cfg.get("entity")
        if not project:
            raise ValueError(
                "logging.wandb.project must be configured to download checkpoints from W&B."
            )
        resolved_path, source_wandb_run_id, preloaded_ckpt = download_checkpoint_from_wandb(
            download_root=output_dir / ".wandb_resume_cache",
            project=str(project),
            entity=str(entity) if entity else None,
            wandb_run_id=str(wandb_run_id) if wandb_run_id else None,
            wandb_artifact=str(wandb_artifact) if wandb_artifact else None,
            target_epoch=target_epoch,
        )
        if preloaded_ckpt is None:
            preloaded_ckpt = load_checkpoint_file(resolved_path)
        source = "wandb_artifact" if wandb_artifact else "wandb_run"
    elif auto:
        # Auto-resume requested without existing local checkpoint: start fresh.
        return ResumeContext(resume_wandb=resume_wandb, source="none")
    elif enabled:
        if expected_last_path.is_file():
            resolved_path = expected_last_path
            preloaded_ckpt = load_checkpoint_file(resolved_path)
            source = "local_last"
        else:
            raise FileNotFoundError(
                "resume.enabled=true was set, but no checkpoint_path, wandb_run_id, "
                f"wandb_artifact, or existing last checkpoint ({expected_last_path}) was found."
            )

    ckpt_wandb_run_id = (
        preloaded_ckpt.get("wandb_run_id") if preloaded_ckpt is not None else None
    )
    final_wandb_run_id: str | None = None
    if resume_wandb:
        raw_id = wandb_run_id or ckpt_wandb_run_id or source_wandb_run_id
        final_wandb_run_id = str(raw_id) if raw_id else None

    return ResumeContext(
        checkpoint_path=resolved_path,
        preloaded_checkpoint=preloaded_ckpt,
        wandb_run_id=final_wandb_run_id,
        resume_wandb=resume_wandb,
        source=source,
    )

