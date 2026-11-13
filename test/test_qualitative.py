"""Tests for per-image scoring and worst/best/random sampling."""
from __future__ import annotations

import math

import numpy as np
import torch

from src.utils.qualitative import QualitativeSampler, per_image_dice, to_class_map


def _batch(num_images: int, num_classes: int = 3, size: int = 8, wrong_from: int = 0):
    """Images i >= wrong_from get a progressively more wrong prediction."""
    masks = torch.randint(0, num_classes, (num_images, 1, size, size))
    preds = masks.clone().squeeze(1)
    for i in range(wrong_from, num_images):
        flip = (i - wrong_from + 1) * size * size // (num_images - wrong_from + 1)
        flat = preds[i].flatten()
        flat[:flip] = (flat[:flip] + 1) % num_classes
        preds[i] = flat.view(size, size)
    images = torch.rand(num_images, 3, size, size)
    return images, masks, preds


def test_per_image_dice_perfect_and_disjoint():
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    pred = target.clone()
    pred[1] = 1
    score = per_image_dice(pred, target, num_classes=2)
    assert score[0].item() == 1.0
    assert score[1].item() == 0.0


def test_per_image_dice_nan_when_only_excluded_classes():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    score = per_image_dice(target.clone(), target, num_classes=2, exclude_classes=[0])
    assert math.isnan(score[0].item())


def test_to_class_map_handles_logits_and_single_channel():
    logits = torch.zeros(2, 3, 4, 4)
    logits[:, 2] = 1.0
    assert to_class_map(logits).unique().tolist() == [2]
    assert to_class_map(torch.ones(2, 1, 4, 4)).shape == (2, 4, 4)


def test_sampler_selects_worst_best_random_across_batches():
    torch.manual_seed(0)
    images, masks, preds = _batch(12, wrong_from=2)
    logits = torch.nn.functional.one_hot(preds, 3).permute(0, 3, 1, 2).float()

    sampler = QualitativeSampler(num_classes=3, num_worst=2, num_best=2, num_random=2, seed=1)
    for start in range(0, 12, 4):  # three batches of 4
        sampler.update(images[start:start + 4], masks[start:start + 4], logits[start:start + 4])

    result = sampler.result()
    groups = result["group"]
    scores = result["score"]
    indices = result["index"]

    assert groups.count("worst") == 2 and groups.count("best") == 2
    assert len(set(indices)) == len(indices)  # deduplicated

    all_scores = per_image_dice(preds, masks.squeeze(1), 3)
    ordered = sorted(all_scores.tolist())
    worst_scores = [sc for g, sc in zip(groups, scores) if g == "worst"]
    best_scores = [sc for g, sc in zip(groups, scores) if g == "best"]
    # compare scores (indices are ambiguous when several images tie)
    assert np.allclose(sorted(worst_scores), ordered[:2], atol=1e-6)
    assert np.allclose(sorted(best_scores), ordered[-2:], atol=1e-6)
    for s, i in zip(scores, indices):
        assert s == np.float32(all_scores[i]).item() or math.isclose(s, all_scores[i].item(), rel_tol=1e-5)

    # images/masks/predictions correspond to the reported index
    assert torch.equal(result["image"][0], images[indices[0]])
    assert torch.equal(result["prediction"][0], preds[indices[0]])
    assert result["image"].shape[0] == len(indices)


def test_confusion_matrix_counts_all_pixels():
    images, masks, preds = _batch(4, wrong_from=1)
    logits = torch.nn.functional.one_hot(preds, 3).permute(0, 3, 1, 2).float()
    sampler = QualitativeSampler(num_classes=3)
    sampler.update(images, masks, logits)
    cm = sampler.confusion_matrix
    assert cm.sum() == 4 * 8 * 8
    expected = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(masks.flatten().tolist(), preds.flatten().tolist()):
        expected[t, p] += 1
    assert (cm == expected).all()


def test_small_dataset_has_no_duplicate_samples():
    images, masks, preds = _batch(3)
    logits = torch.nn.functional.one_hot(preds, 3).permute(0, 3, 1, 2).float()
    sampler = QualitativeSampler(num_classes=3, num_worst=4, num_best=4, num_random=4)
    sampler.update(images, masks, logits)
    result = sampler.result()
    assert sorted(result["index"]) == [0, 1, 2]


def test_empty_sampler_returns_none():
    assert QualitativeSampler(num_classes=3).result() is None
