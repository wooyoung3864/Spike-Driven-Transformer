# hooks/syops_hook.py
import torch
import numpy as np          
from syops import get_model_complexity_info

def _to_num(x):
    """make sure we end up with a Python `float` / `int`"""
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return float(x.item())                   # scalar ndarray / tensor
    return float(x)

def estimate_ops(
    model: torch.nn.Module,
    input_size: tuple,                 # (C, H, W)
    dataloader=None,                   # spike-rate aware if you pass one
    spike_ac_energy_pj: float = 0.9,   # energy for one AC @45 nm
    spike_mac_energy_pj: float = 4.6,  # energy for one MAC @45 nm
    input_constructor=None,
    **kwargs,
):
    """
    Run SyOps counter and return a dict with:
        * acs_G, macs_G       – accumulated ops, multiply-acc ops   (billions)
        * params_M    – parameters        (millions)
        * energy_mJ   – AC+MAC energy in millijoules
    """
    result = get_model_complexity_info(
        model,
        input_res=input_size,
        dataloader=dataloader,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        **kwargs,
    )

    # ------------------------------------------------------------------ #
    # 1. unpack the various formats syops might return
    # ------------------------------------------------------------------ #
    if isinstance(result, dict):                       # future-proof
        macs = _to_num(result.get("macs", 0))
        acs  = _to_num(result.get("acs", 0))
        params = _to_num(result.get("params", 0))

    else:                                              # tuple / list / ndarray
        if isinstance(result, (tuple, list)):
            items = list(result)
        else:  # ndarray / tensor
            items = list(result.tolist())

        if len(items) == 3:
            macs, params, acs = map(_to_num, items)
        elif len(items) == 2:                          # (ops, params)
            ops_arr, params = items
            ops_arr = np.asarray(ops_arr).astype(float)

            if ops_arr.size == 2:                      # [acs, macs] OR [macs, acs]
                acs, macs = sorted(ops_arr)            # smaller is ACs (≈ spike count)
            else:
                # assume first column ACs, second MACs
                acs, macs = ops_arr[0], ops_arr[1]
            acs, macs = map(float, (acs, macs))
        else:
            raise RuntimeError(
                f"Cannot parse `ops` returned by syops: type={type(result)}"
            )

    # ------------------------------------------------------------------ #
    # 2. energy estimate
    # ------------------------------------------------------------------ #
    energy_j = (
        acs  * spike_ac_energy_pj  * 1e-12 +
        macs * spike_mac_energy_pj * 1e-12
    )

    return dict(
        acs_G     = acs   / 1e9,
        macs_G    = macs  / 1e9,
        energy_mJ = energy_j * 1e3,
        params_M  = params / 1e6,
    )
    