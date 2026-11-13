"""Tests for the video x class (frames + pixels) validation table."""
from __future__ import annotations

import numpy as np

from src.datasets.validation import (
    class_video_matrix,
    class_video_presence,
    class_video_table,
    flag_classes,
    validate_dataset,
)


class FakeDataset:
    class_names = ["bg", "liver", "tool"]

    def __init__(self):
        m = np.zeros((2, 2), dtype=np.int64)
        liver = m.copy(); liver[0, 0] = 1
        both = m.copy(); both[0, 0] = 1; both[1, 1] = 2
        self.masks = [m, liver, both, m]
        self.samples = [
            {"video": "v1", "split": "train"},
            {"video": "v1", "split": "train"},
            {"video": "v2", "split": "test"},
            {"video": "v2", "split": "test"},
        ]

    def __getitem__(self, i):
        return {"mask": self.masks[i]}


def test_table_shape_and_values():
    t = class_video_table(FakeDataset())
    assert list(t.index) == ["v1", "v2"]
    assert list(t.columns) == [(c, m) for c in FakeDataset.class_names
                               for m in ("frames", "pixels")]
    assert t.loc["v1", ("liver", "frames")] == 1
    assert t.loc["v1", ("liver", "pixels")] == 1
    assert t.loc["v2", ("tool", "frames")] == 1
    assert t.loc["v2", ("tool", "pixels")] == 1
    assert t.loc["v1", ("tool", "frames")] == 0
    assert t.loc["v1", ("bg", "pixels")] == 7
    # all pixels accounted for
    assert t.xs("pixels", axis=1, level="metric").to_numpy().sum() == 16


def test_matrix_matches_table_frames():
    ds = FakeDataset()
    m = class_video_matrix(ds)
    assert m.loc["v2", "bg"] == 2 and m.loc["v1", "liver"] == 1


def test_validate_dataset_has_split_level_and_flags():
    ds = FakeDataset()
    out = validate_dataset(ds)
    t = out["video_class_table"]
    assert t.index.names == ["split", "video"]
    assert ("train", "v1") in t.index
    flagged = flag_classes(out["global_video_presence"], 1, 101.0)
    assert flagged == ["tool"]
