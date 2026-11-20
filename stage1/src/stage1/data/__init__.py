"""Stage 1 datasets: Speed Dating in its OpenML form and the original Columbia CSV."""

from stage1.data.speed_dating import DyadicData, load

DATASETS = ("columbia", "openml")

__all__ = ["DATASETS", "DyadicData", "load"]
