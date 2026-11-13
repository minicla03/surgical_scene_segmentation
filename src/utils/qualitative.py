"""Streaming selection of qualitative samples (worst / best / random by per-image Dice).

The sampler is fed batch by batch during validation/test, so no extra forward
pass is needed and memory stays bounded: only the selected images are copied
to the CPU.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


def to_class_map(predictions: torch.Tensor) -> torch.Tensor:
    """Convert model output to an integer class map [B, H, W]."""
    if predictions.ndim == 4:
        if predictions.shape[1] > 1:
            predictions = predictions.argmax(dim=1)
        else:
            predictions = predictions.squeeze(1)
    if predictions.ndim != 3:
        raise ValueError(f"Cannot convert predictions of shape {tuple(predictions.shape)} to a class map.")
    return predictions.long()


def per_image_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    exclude_classes: Iterable[int] = (),
) -> torch.Tensor:
    """Mean Dice per image over the classes present in target or prediction.

    Args:
        prediction: [B, H, W] integer class map.
        target: [B, H, W] integer class map.

    Returns:
        [B] tensor, NaN for images where no (non-excluded) class is present.
    """
    excluded = {int(c) for c in exclude_classes}
    batch = prediction.shape[0]
    dice_sum = torch.zeros(batch, device=prediction.device)
    valid_count = torch.zeros(batch, device=prediction.device)

    for class_id in range(num_classes):
        if class_id in excluded:
            continue
        pred_c = prediction == class_id
        target_c = target == class_id
        inter = (pred_c & target_c).flatten(1).sum(1).float()
        total = pred_c.flatten(1).sum(1).float() + target_c.flatten(1).sum(1).float()
        present = total > 0
        dice = 2.0 * inter / total.clamp(min=1.0)
        dice_sum += torch.where(present, dice, torch.zeros_like(dice))
        valid_count += present.float()

    score = dice_sum / valid_count.clamp(min=1.0)
    return torch.where(valid_count > 0, score, torch.full_like(score, float("nan")))


@dataclass
class _Candidate:
    score: float
    index: int
    image: torch.Tensor
    mask: torch.Tensor
    prediction: torch.Tensor


class QualitativeSampler:
    """Keeps the K worst, K best and K random samples seen so far.

    ``index`` is the position of the sample in the iteration order of the
    loader (equal to the dataset index when the loader does not shuffle).
    Also accumulates a confusion matrix over all pixels seen.
    """

    def __init__(
        self,
        num_classes: int,
        num_worst: int = 4,
        num_best: int = 2,
        num_random: int = 2,
        exclude_classes: Iterable[int] = (),
        seed: int = 0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.num_worst = max(0, int(num_worst))
        self.num_best = max(0, int(num_best))
        self.num_random = max(0, int(num_random))
        self.exclude_classes = tuple(int(c) for c in exclude_classes)

        self._rng = random.Random(seed)
        self._seen = 0
        self._worst: list[_Candidate] = []
        self._best: list[_Candidate] = []
        self._random: list[_Candidate] = []
        self._confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @classmethod
    def from_config(cls, qualitative_cfg: Any, num_classes: int, seed: int = 0) -> "QualitativeSampler":
        cfg = qualitative_cfg or {}
        return cls(
            num_classes=num_classes,
            num_worst=cfg.get("num_worst", 4),
            num_best=cfg.get("num_best", 2),
            num_random=cfg.get("num_random", 2),
            exclude_classes=list(cfg.get("exclude_classes", []) or []),
            seed=seed,
        )

    @torch.no_grad()
    def update(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        pred = to_class_map(predictions)
        target = masks.squeeze(1).long() if masks.ndim == 4 else masks.long()

        scores = per_image_dice(pred, target, self.num_classes, self.exclude_classes).cpu()
        self._update_confusion(pred, target)

        cache: dict[int, _Candidate] = {}

        def candidate(i: int) -> _Candidate:
            if i not in cache:
                cache[i] = _Candidate(
                    score=float(scores[i]),
                    index=self._seen + i,
                    image=images[i].detach().cpu(),
                    mask=masks[i].detach().cpu(),
                    prediction=pred[i].detach().cpu(),
                )
            return cache[i]

        for i in range(pred.shape[0]):
            score = float(scores[i])
            has_score = not math.isnan(score)

            if has_score and self.num_worst and (
                len(self._worst) < self.num_worst or score < self._worst[-1].score
            ):
                self._worst.append(candidate(i))
                self._worst.sort(key=lambda c: c.score)
                del self._worst[self.num_worst:]

            if has_score and self.num_best and (
                len(self._best) < self.num_best or score > self._best[-1].score
            ):
                self._best.append(candidate(i))
                self._best.sort(key=lambda c: -c.score)
                del self._best[self.num_best:]

            if self.num_random:
                position = self._seen + i
                if len(self._random) < self.num_random:
                    self._random.append(candidate(i))
                else:
                    j = self._rng.randint(0, position)
                    if j < self.num_random:
                        self._random[j] = candidate(i)

        self._seen += pred.shape[0]

    def _update_confusion(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        c = self.num_classes
        valid = (target >= 0) & (target < c) & (pred >= 0) & (pred < c)
        flat = (target[valid] * c + pred[valid]).flatten()
        counts = torch.bincount(flat, minlength=c * c).reshape(c, c)
        self._confusion += counts.cpu()

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._confusion.numpy()

    def result(self) -> dict[str, Any] | None:
        """Return the selected samples (deduplicated: worst > best > random)."""
        taken: set[int] = set()
        selected: list[tuple[str, _Candidate]] = []

        for group, candidates in (
            ("worst", self._worst),
            ("best", self._best),
            ("random", sorted(self._random, key=lambda c: c.index)),
        ):
            for cand in candidates:
                if cand.index in taken:
                    continue
                taken.add(cand.index)
                selected.append((group, cand))

        if not selected:
            return None

        return {
            "image": torch.stack([c.image for _, c in selected]),
            "mask": torch.stack([c.mask for _, c in selected]),
            "prediction": torch.stack([c.prediction for _, c in selected]),
            "group": [g for g, _ in selected],
            "score": [None if math.isnan(c.score) else c.score for _, c in selected],
            "index": [c.index for _, c in selected],
            "confusion_matrix": self.confusion_matrix.copy(),
        }
