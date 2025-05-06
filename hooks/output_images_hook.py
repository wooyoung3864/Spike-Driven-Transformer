# hooks/output_images_hook.py
# 05/02 update (wyjung): modularized image saving hook.
import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

__all__ = ["OutputImageSaver"]


class OutputImageSaver:
    """
    Collects layer outputs during `model(...)` and saves .npy + preview images.

    Usage
    -----
    saver = OutputImageSaver(base_dir, sample_batches=[0, 100, 156])
    outputs = model(x, hook={})          # <- pass empty dict
    saver.dump(batch_idx, hook)

    Call `saver.remove()` once you no longer need the hooks.
    """

    def __init__(self, base_dir: str, sample_batches=(0,), cmap="viridis"):
        self.base_dir       = base_dir                  # <-- remember it
        self.sample_batches = set(sample_batches)
        self.cmap = cmap
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def dump(self, batch_idx: int, hook_dict: dict, raw_input=None):
        """Save outputs if batch_idx is in sample_batches."""
        if batch_idx not in self.sample_batches:
            return

        if raw_input is not None:
            self._save_input(batch_idx, raw_input)

        for name, tensor in hook_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            self._save_tensor(batch_idx, name, tensor.cpu())

    # ------------------------------------------------------------------ #
    def _save_input(self, idx, img):
        img_dir = os.path.join(self.base_dir, "input_images")
        os.makedirs(img_dir, exist_ok=True)
        vutils.save_image(img[0], os.path.join(img_dir, f"b{idx}_input.png"))

    def _save_tensor(self, idx: int, name: str, tensor: torch.Tensor):
        # ------------------------------------------------------------------
        # 0) make /<base_dir>/<layer-name>/ if it doesn't exist ------------
        lay_dir = os.path.join(self.base_dir, name)
        os.makedirs(lay_dir, exist_ok=True)            # <-   define & create
    
        # 1. flatten time dimension etc. -------------------------------
        if tensor.dim() == 5:         # (T, C, H, W, ?)
            tensor = tensor[0]        # keep first time-slice
        if tensor.dim() == 4:         # (N, C, H, W)
            # take at most the first 3 channels to satisfy PIL
            if tensor.size(1) > 3:
                tensor = tensor[:, :3]

            # replicate 1-channel tensors to fake RGB so we can still
            # visualise them with colour maps later
            if tensor.size(1) == 1:
                tensor = tensor.expand(-1, 3, -1, -1)

            grid = vutils.make_grid(
                tensor.float(), nrow=min(tensor.size(0), 8),
                normalize=True, scale_each=True
            )

            # ---> NEW: guarantee C == 3
            if grid.size(0) == 1:
                grid = grid.expand(3, -1, -1)        # 1->3
            elif grid.size(0) > 3:
                grid = grid[:3]                      # >3 → first 3

            img_path = os.path.join(lay_dir, f"b{idx}.png")
            vutils.save_image(grid, img_path)
