"""Rebuilt color iEEG preprocessing pipeline.

The pipeline is intentionally split into two gates:

1. audit and manual review of channel quality;
2. preprocessing and HDF5 export using only reviewed exclusions.
"""

from .condition_registry import condition_for_trigger, conditions_for_group
from .epoch_plots import plot_epoch_mean_shading, plot_hdf5_conditions

__all__ = [
    "condition_for_trigger",
    "conditions_for_group",
    "plot_epoch_mean_shading",
    "plot_hdf5_conditions",
]
