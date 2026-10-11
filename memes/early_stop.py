"""Early stopping on a higher-is-better validation metric (here: validation AUROC)."""
from __future__ import annotations


class EarlyStopper:
    """update(value) -> (improved, stop). `improved` means a new best by more than min_delta; `stop` means `patience`
    consecutive epochs passed without an improvement. patience <= 0 never stops (best epoch is still tracked)."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best: float | None = None
        self.best_epoch = -1
        self.bad = 0
        self.epoch = -1

    def update(self, value: float) -> tuple[bool, bool]:
        self.epoch += 1
        if value != value:                                   # NaN never counts as an improvement
            improved = False
        else:
            improved = self.best is None or value > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad = value, self.epoch, 0
        else:
            self.bad += 1
        return improved, bool(self.patience > 0 and self.bad >= self.patience)
