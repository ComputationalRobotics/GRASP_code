"""Synchronous, non-pickle diagnostic records; never consumes optimizer RNG."""
from pathlib import Path
import numpy as np


class OptimizationTrace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = 0

    def __call__(self, event, **values):
        arrays = {"event": np.asarray(event)}

        def flatten(key, value):
            if value is None:
                arrays[key + "/is_none"] = np.asarray(True)
            elif isinstance(value, dict):
                for name, item in value.items():
                    flatten(key + "/" + str(name), item)
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    flatten(key + "/" + str(i), item)
            else:
                array = np.asarray(value)
                if array.dtype.hasobject:
                    # PCG64 state contains Python integers wider than uint64.
                    if isinstance(value, int):
                        array = np.asarray(str(value))
                    else:
                        raise TypeError(f"Unsupported trace value: {key}")
                arrays[key] = array

        for key, value in values.items():
            flatten(key, value)
        path = self.directory / f"{self.index:05d}_{event}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        self.index += 1
