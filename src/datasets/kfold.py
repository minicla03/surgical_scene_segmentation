"""Group-level (video) K-fold splitting with class-coverage guarantees.

Rules implemented here:

* The split unit is the group (video / surgery), never the frame.
* Classes that live in a single group can never be both trained and validated:
  the groups containing them are *pinned* to train and those classes are not
  evaluated (they are reported as n/a).
* A class present in >= 2 groups is spread so that, whenever possible, a fold
  has it in validation AND in train. The class is therefore evaluable in
  ``min(K, groups_with_class)`` folds at best.
* Optionally a test set is carved out *before* the K-fold, with priority
  train > validation > test: it never removes the groups a class needs to be
  trained and validated.
* Fold assignment starts from multilabel stratification
  (``iterative-stratification``) and is then improved with a greedy swap
  search with two criteria, in this order: (1) maximise the number of folds in which
  each class is evaluable; (2) keep the class distribution of every validation fold close
  to the distribution of the whole dataset (pixel share of the populous classes and fold size).
  Several deterministic restarts (seed, seed+1, ...) are tried and the best one is kept.

The core (:func:`plan_kfold`) works on a group x class presence matrix, so it
does not depend on any dataset. :func:`resolve_kfold_splits` adapts it to the
dataset classes used in this project.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.datasets.validation import (
    _config_get,
    _group_indices,
    _resolve_group_key,
    compute_sample_counts,
    get_class_names,
    get_samples,
)


# ============================================================
# PLAN
# ============================================================

@dataclass
class KFoldPlan:
    """Result of the group assignment.

    All group references are positions in ``group_ids``.
    ``val_folds[k]`` are the validation groups of fold ``k``; the train groups
    of fold ``k`` are every other non-test group (pinned groups included).
    """

    n_folds: int
    seed: int
    test_ratio: float
    group_ids: list[str]
    class_names: list[str]
    presence: np.ndarray            # (G, C) bool
    group_sizes: np.ndarray         # (G,) frames per group
    pinned: list[int]
    pinned_reasons: dict[int, list[int]]  # group -> single-group class ids it holds
    test: list[int]
    val_folds: list[list[int]]
    targets: np.ndarray             # (C,) max achievable number of evaluable folds
    group_pixels: np.ndarray | None = None   # (G, C) class pixels per group (if known)
    balance_cost: float | None = None        # distribution-balance cost of the chosen folds

    # ---------------- derived views ----------------

    def _non_test(self) -> np.ndarray:
        mask = np.ones(len(self.group_ids), dtype=bool)
        mask[self.test] = False
        return mask

    def evaluable_matrix(self) -> np.ndarray:
        """(K, C) bool: class c has pixels in the validation AND train of fold k."""
        return _evaluable_matrix(self.presence, self._non_test(), self.val_folds)

    def evaluable_classes(self, fold_index: int) -> np.ndarray:
        """(C,) bool mask of the classes that can be evaluated in ``fold_index``.

        The other classes must be reported as NaN for that fold.
        """
        self._check_fold(fold_index)
        return self.evaluable_matrix()[fold_index]

    def groups_for(self, fold_index: int) -> dict[str, list[str]]:
        """Group ids of train / validation / test for one fold."""
        self._check_fold(fold_index)
        val = set(self.val_folds[fold_index])
        test = set(self.test)
        ids = self.group_ids
        return {
            "train": [ids[g] for g in range(len(ids)) if g not in val and g not in test],
            "validation": [ids[g] for g in sorted(val)],
            "test": [ids[g] for g in sorted(test)],
        }

    def _check_fold(self, fold_index: int) -> None:
        if not 0 <= fold_index < self.n_folds:
            raise ValueError(f"fold_index must be in [0, {self.n_folds - 1}], got {fold_index}.")

    # ---------------- reports ----------------

    def coverage_table(self) -> pd.DataFrame:
        """One row per class: videos, target/achieved evaluable folds, per-fold status."""
        evaluable = self.evaluable_matrix()
        test_presence = self.presence[self.test].any(0) if self.test else np.zeros(
            self.presence.shape[1], dtype=bool
        )
        rows = []
        for c, name in enumerate(self.class_names):
            row: dict[str, Any] = {
                "class_id": c,
                "class": name,
                "videos": int(self.presence[:, c].sum()),
                "single_video": bool(self.presence[:, c].sum() == 1),
                "target_folds": int(self.targets[c]),
                "valid_folds": int(evaluable[:, c].sum()),
                "valid_fold_ids": [int(k) for k in np.flatnonzero(evaluable[:, c])],
                "in_test": bool(test_presence[c]),
            }
            for k, val in enumerate(self.val_folds):
                in_val = bool(self.presence[val, c].any()) if val else False
                row[f"fold{k}"] = "ok" if evaluable[k, c] else (
                    "n/a (not in val)" if not in_val else "n/a (not in train)"
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def fold_table(self) -> pd.DataFrame:
        """One row per fold: number of videos and frames in each split."""
        rows = []
        for k in range(self.n_folds):
            groups = self.groups_for(k)
            index = {g: i for i, g in enumerate(self.group_ids)}
            row: dict[str, Any] = {"fold": k}
            for split, ids in groups.items():
                row[f"{split}_videos"] = len(ids)
                row[f"{split}_frames"] = int(sum(self.group_sizes[index[g]] for g in ids))
            row["validation_ids"] = groups["validation"]
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "test_ratio": self.test_ratio,
            "balance_cost": self.balance_cost,
            "class_names": self.class_names,
            "pinned_train": [self.group_ids[g] for g in self.pinned],
            "test": [self.group_ids[g] for g in self.test],
            "folds": [
                {"fold": k, **self.groups_for(k),
                 "evaluable_classes": [
                     self.class_names[c] for c in np.flatnonzero(self.evaluable_classes(k))
                 ]}
                for k in range(self.n_folds)
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ============================================================
# CORE HELPERS
# ============================================================

def _evaluable_matrix(
    presence: np.ndarray, non_test: np.ndarray, val_folds: Sequence[Sequence[int]]
) -> np.ndarray:
    total = presence[non_test].sum(0)
    out = np.zeros((len(val_folds), presence.shape[1]), dtype=bool)
    for k, val in enumerate(val_folds):
        val_count = presence[list(val)].sum(0) if len(val) else np.zeros_like(total)
        out[k] = (val_count > 0) & ((total - val_count) > 0)
    return out


def _coverage_targets(
    presence: np.ndarray, free: np.ndarray, pinned: np.ndarray, n_folds: int
) -> np.ndarray:
    """Max number of folds in which each class can be evaluable."""
    free_count = presence[free].sum(0)
    pinned_count = presence[pinned].sum(0) if len(pinned) else np.zeros_like(free_count)
    reachable = (free_count > 0) & ((pinned_count > 0) | (free_count >= 2))
    return np.where(reachable, np.minimum(free_count, n_folds), 0)


def _select_pinned(presence: np.ndarray) -> tuple[list[int], dict[int, list[int]]]:
    """Groups holding a class that exists in exactly one group."""
    single = np.flatnonzero(presence.sum(0) == 1)
    reasons: dict[int, list[int]] = {}
    for c in single:
        g = int(np.flatnonzero(presence[:, c])[0])
        reasons.setdefault(g, []).append(int(c))
    return sorted(reasons), reasons


def _select_test(
    presence: np.ndarray, candidates: list[int], n_test: int, seed: int
) -> list[int]:
    """Greedy test selection with priority train > validation > test.

    A group may enter the test only if every class present in >= 2 groups keeps
    at least two non-test groups, one of them free (not pinned), so that it can
    still be trained and validated. Among safe groups, the one covering the most
    not-yet-covered classes (rare classes weighted more) is taken.
    """
    if n_test <= 0:
        return []

    counts = presence.sum(0)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(candidates))
    free_total = presence[candidates].sum(0)
    removed = np.zeros_like(counts)
    covered = np.zeros(presence.shape[1], dtype=bool)
    weight = 1.0 / np.maximum(counts, 1)
    chosen: list[int] = []

    def safe(g: int) -> bool:
        after = removed + presence[g]
        multi = counts >= 2
        return bool(
            np.all((counts - after)[multi] >= 2) and np.all((free_total - after)[multi] >= 1)
        )

    while len(chosen) < n_test:
        best, best_score = None, -1.0
        for g in order:
            if g in chosen or not safe(g):
                continue
            score = float((weight * (presence[g] & ~covered)).sum())
            if score > best_score:
                best, best_score = g, score
        if best is None:
            warnings.warn(
                f"Only {len(chosen)}/{n_test} test groups could be selected without "
                "removing groups needed to train/validate rare classes.",
                stacklevel=3,
            )
            break
        chosen.append(int(best))
        removed += presence[best]
        covered |= presence[best]

    return sorted(chosen)


def _initial_folds(presence_free: np.ndarray, n_folds: int, seed: int) -> list[list[int]]:
    """Multilabel-stratified K-fold over the free groups (positions in the free array)."""
    positions = np.arange(len(presence_free))
    keep = presence_free.sum(0) > 0
    if keep.any():
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        except ImportError as exc:
            raise ImportError(
                "Install iterative-stratification with:\npip install iterative-stratification"
            ) from exc
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        parts = splitter.split(positions, presence_free[:, keep].astype(int))
    else:
        from sklearn.model_selection import KFold

        parts = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(positions)
    return [sorted(int(i) for i in val) for _, val in parts]


def _balance_cost_fn(
    pixels: np.ndarray,
    sizes: np.ndarray,
    free: list[int],
    targets: np.ndarray,
    n_folds: int,
    min_pixel_pct: float,
    size_weight: float,
):
    """Cost of a fold assignment: how far every validation fold is from the dataset distribution.

    * classes: those that can be validated (``targets > 0``) with at least ``min_pixel_pct`` % of
      the pixels of the groups available for validation; for each, the relative deviation of its
      pixel share in the validation fold from the share over all those groups, squared and
      weighted by the class share (populous classes count more);
    * size: relative deviation of the number of frames of the fold from ``total / K``.
    """
    pixels = pixels.astype(np.float64)
    pool = pixels[free].sum(axis=0)
    pool_share = pool / pool.sum() if pool.sum() > 0 else pool
    chosen = (targets > 0) & (pool_share * 100.0 >= min_pixel_pct)
    weights = pool_share[chosen] / pool_share[chosen].sum() if chosen.any() else None
    frames = sizes.astype(np.float64)

    def cost(folds: list[list[int]]) -> float:
        total = 0.0
        if weights is not None:
            val = np.stack([pixels[f].sum(axis=0) for f in folds])
            fold_total = val.sum(axis=1, keepdims=True)
            share = np.divide(val, fold_total, out=np.zeros_like(val), where=fold_total > 0)
            rel = (share[:, chosen] - pool_share[chosen]) / pool_share[chosen]
            total += float((weights * rel**2).sum(axis=1).mean())
        per_fold = np.array([frames[f].sum() for f in folds])
        if per_fold.sum() > 0:
            total += size_weight * float((((per_fold * len(folds) / per_fold.sum()) - 1) ** 2).mean())
        return total

    return cost


def _greedy_repair(
    presence: np.ndarray,
    non_test: np.ndarray,
    folds: list[list[int]],
    targets: np.ndarray,
    cost_fn=None,
    max_iterations: int = 500,
) -> list[list[int]]:
    """Swap groups between folds, improving (coverage deficit, balance cost) lexicographically.

    The first criterion is the class-coverage deficit (see ``_coverage_targets``); the second,
    when ``cost_fn`` is given, is the distribution-balance cost. A swap keeps the number of
    groups per fold unchanged and is accepted only if it strictly improves the pair, so the
    search always terminates.
    """
    pres = presence.astype(np.int64)
    total = pres[non_test].sum(0)
    n_folds = len(folds)
    val_count = np.stack([pres[f].sum(0) for f in folds])

    def evaluable_row(count: np.ndarray) -> np.ndarray:
        return (count > 0) & ((total - count) > 0)

    evaluable = np.stack([evaluable_row(c) for c in val_count])
    coverage = evaluable.sum(0)

    def deficit(cov: np.ndarray) -> int:
        return int(np.clip(targets - cov, 0, None).sum())

    tolerance = 1e-12
    current = deficit(coverage)
    current_cost = cost_fn(folds) if cost_fn is not None else 0.0

    for _ in range(max_iterations):
        if current == 0 and cost_fn is None:
            break
        best: tuple[int, int, int, int] | None = None
        best_key = (current, current_cost - tolerance)
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                for i, v in enumerate(folds[a]):
                    for j, w in enumerate(folds[b]):
                        delta = pres[w] - pres[v]
                        new_a = evaluable_row(val_count[a] + delta)
                        new_b = evaluable_row(val_count[b] - delta)
                        cov = coverage - evaluable[a] - evaluable[b] + new_a + new_b
                        d = deficit(cov)
                        if d > best_key[0]:
                            continue
                        c = 0.0
                        if cost_fn is not None:
                            swapped = [list(f) for f in folds]
                            swapped[a][i], swapped[b][j] = w, v
                            c = cost_fn(swapped)
                        if (d, c) < best_key:
                            best, best_key = (a, b, i, j), (d, c)
        if best is None:
            break
        a, b, i, j = best
        v, w = folds[a][i], folds[b][j]
        folds[a][i], folds[b][j] = w, v
        delta = pres[w] - pres[v]
        val_count[a] += delta
        val_count[b] -= delta
        evaluable[a] = evaluable_row(val_count[a])
        evaluable[b] = evaluable_row(val_count[b])
        coverage = evaluable.sum(0)
        current, current_cost = best_key

    return [sorted(f) for f in folds]


# ============================================================
# PUBLIC API
# ============================================================

def plan_kfold(
    presence: np.ndarray,
    group_ids: Sequence[Any],
    class_names: Sequence[str],
    n_folds: int = 4,
    test_ratio: float = 0.0,
    seed: int = 42,
    group_sizes: Sequence[int] | None = None,
    group_pixels: np.ndarray | None = None,
    n_restarts: int = 30,
    balance_min_pixel_pct: float = 1.0,
    size_weight: float = 0.5,
) -> KFoldPlan:
    """Assign groups to pinned-train / test / K validation folds.

    Args:
        presence: (G, C) matrix, presence[g, c] truthy if class c appears in group g.
        group_ids: identifier of each group, same order as ``presence`` rows.
        class_names: name of each class, same order as ``presence`` columns.
        n_folds: number of folds K (>= 2).
        test_ratio: fraction of the groups held out as test before the K-fold.
        seed: seed for stratification and test selection.
        group_sizes: frames per group (fold-size balance and reports).
        group_pixels: optional (G, C) class pixels per group. When given, after the class
            coverage the folds are balanced so that the pixel share of the populous classes
            in each validation fold stays close to the dataset's (see ``_balance_cost_fn``).
        n_restarts: number of deterministic restarts (seed, seed+1, ...) when balancing.
        balance_min_pixel_pct: classes below this % of the pixels do not enter the balance.
        size_weight: weight of the fold-size term in the balance cost.
    """
    presence = np.asarray(presence).astype(bool)
    group_ids = [str(g) for g in group_ids]
    class_names = [str(c) for c in class_names]

    if presence.ndim != 2 or presence.shape != (len(group_ids), len(class_names)):
        raise ValueError(
            f"presence must have shape (groups={len(group_ids)}, classes={len(class_names)}), "
            f"got {presence.shape}."
        )
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group_ids must be unique.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}.")

    sizes = (
        np.ones(len(group_ids), dtype=np.int64)
        if group_sizes is None
        else np.asarray(group_sizes, dtype=np.int64)
    )

    # Canonical order: the result must depend only on the group ids and the seed, never on the
    # order in which the dataset happens to list its groups.
    order = sorted(range(len(group_ids)), key=lambda i: group_ids[i])
    group_ids = [group_ids[i] for i in order]
    presence = presence[order]
    sizes = sizes[order]
    pixels = None
    if group_pixels is not None:
        pixels = np.asarray(group_pixels)
        if pixels.shape != presence.shape:
            raise ValueError(f"group_pixels must have shape {presence.shape}, got {pixels.shape}.")
        pixels = pixels[order]

    pinned, pinned_reasons = _select_pinned(presence)
    pinned_set = set(pinned)
    candidates = [g for g in range(len(group_ids)) if g not in pinned_set]

    n_test = int(round(test_ratio * len(group_ids)))
    test = _select_test(presence, candidates, n_test, seed)
    test_set = set(test)

    free = [g for g in candidates if g not in test_set]
    if len(free) < n_folds:
        raise ValueError(
            f"Only {len(free)} groups are free for validation (after pinning "
            f"{len(pinned)} and holding out {len(test)} for test), "
            f"fewer than n_folds={n_folds}."
        )

    non_test = np.ones(len(group_ids), dtype=bool)
    non_test[test] = False
    targets = _coverage_targets(presence, free, pinned, n_folds)

    cost_fn = (
        _balance_cost_fn(pixels, sizes, free, targets, n_folds, balance_min_pixel_pct, size_weight)
        if pixels is not None
        else None
    )

    best_folds, best_key = None, None
    for restart in range(max(1, n_restarts) if cost_fn is not None else 1):
        initial = _initial_folds(presence[free], n_folds, seed + restart)
        folds = [[free[p] for p in part] for part in initial]
        folds = _greedy_repair(presence, non_test, folds, targets, cost_fn=cost_fn)
        achieved_r = _evaluable_matrix(presence, non_test, folds).sum(0)
        key = (
            int(np.clip(targets - achieved_r, 0, None).sum()),
            cost_fn(folds) if cost_fn is not None else 0.0,
        )
        if best_key is None or key < (best_key[0], best_key[1] - 1e-12):
            best_folds, best_key = folds, key
    folds = best_folds

    plan = KFoldPlan(
        n_folds=n_folds,
        seed=seed,
        test_ratio=test_ratio,
        group_ids=group_ids,
        class_names=class_names,
        presence=presence,
        group_sizes=sizes,
        pinned=pinned,
        pinned_reasons=pinned_reasons,
        test=test,
        val_folds=folds,
        targets=targets,
        group_pixels=pixels,
        balance_cost=best_key[1] if cost_fn is not None else None,
    )

    achieved = plan.evaluable_matrix().sum(0)
    short = np.flatnonzero(achieved < targets)
    if len(short):
        warnings.warn(
            "Coverage target not reached for: "
            + ", ".join(f"{class_names[c]} ({achieved[c]}/{targets[c]} folds)" for c in short),
            stacklevel=2,
        )
    return plan


def kfold_indices(
    plan: KFoldPlan, groups: dict[str, list[int]], fold_index: int
) -> dict[str, list[int]]:
    """Expand the group assignment of one fold into sample indices."""
    selected = plan.groups_for(fold_index)
    return {
        split: [index for group in ids for index in groups[group]]
        for split, ids in selected.items()
    }


def pixels_from_counts(
    groups: dict[str, list[int]],
    cache: dict[int, tuple[np.ndarray, int]],
    class_count: int,
) -> np.ndarray:
    """(G, C) class pixels per group from per-sample pixel counts (``compute_sample_counts``)."""
    pixels = np.zeros((len(groups), class_count), dtype=np.int64)
    for row, indices in enumerate(groups.values()):
        pixels[row] = np.sum([cache[i][0] for i in indices], axis=0)
    return pixels


def presence_from_counts(
    groups: dict[str, list[int]],
    cache: dict[int, tuple[np.ndarray, int]],
    class_count: int,
) -> np.ndarray:
    """(G, C) presence matrix (at least one pixel) from per-sample pixel counts."""
    return pixels_from_counts(groups, cache, class_count) > 0


def build_kfold_plan(
    dataset: Any,
    split_config: Any,
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
) -> tuple[KFoldPlan, dict[str, list[int]]]:
    """Build the plan for a dataset exposing ``samples`` and ``class_names``.

    Config keys read from ``split_config``: ``group_by``, ``n_folds``, ``test_ratio``,
    ``seed``, and for the distribution balance ``balance`` (default True), ``n_restarts``
    (30) and ``balance_min_pixel_pct`` (1.0). If the per-sample pixel counts (``cache``) are
    already available, the masks are not read again.
    """
    samples = get_samples(dataset)
    group_by = _resolve_group_key(samples, str(_config_get(split_config, "group_by", "video")))
    groups = _group_indices(samples, group_by)
    class_names = get_class_names(dataset)
    group_ids = list(groups)

    if cache is None:
        cache = compute_sample_counts(dataset, [i for idx in groups.values() for i in idx])
    pixels = pixels_from_counts(groups, cache, len(class_names))

    plan = plan_kfold(
        presence=pixels > 0,
        group_ids=group_ids,
        class_names=class_names,
        n_folds=int(_config_get(split_config, "n_folds", 4)),
        test_ratio=float(_config_get(split_config, "test_ratio", 0.0)),
        seed=int(_config_get(split_config, "seed", 42)),
        group_sizes=[len(groups[g]) for g in group_ids],
        group_pixels=pixels if bool(_config_get(split_config, "balance", True)) else None,
        n_restarts=int(_config_get(split_config, "n_restarts", 30)),
        balance_min_pixel_pct=float(_config_get(split_config, "balance_min_pixel_pct", 1.0)),
    )
    return plan, groups


def resolve_kfold_splits(
    dataset: Any, split_config: Any
) -> tuple[dict[str, list[int]], KFoldPlan]:
    """Sample indices of ``split_config.fold_index`` plus the full plan."""
    plan, groups = build_kfold_plan(dataset, split_config)
    fold_index = int(_config_get(split_config, "fold_index", 0))
    return kfold_indices(plan, groups, fold_index), plan
