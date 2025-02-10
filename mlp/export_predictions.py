"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path
import h5py
import numpy as np
import torch
from .tensor import batch_to_device
from tqdm import tqdm

@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)

        pred = {**pred, **data}

        # get sparse depth key point
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        
        # get key:value pair for pred
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {key: value for key, value in pred.items() if key in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        # for key in pred.keys():
        #     if key.startswith("keypoints"):
        #         idx = key.replace("keypoints", "")
        #         scales = 1.0 / (data["scales"] if len(idx) == 0 
        #                                         else data[f"view{idx}"]["scales"])
        #         pred[key] = pred[key] * scales[None]
        # breakpoint()
        pred = {key: value[0].cpu().numpy() for key, value in pred.items()}

        if as_half:
            for key in pred:
                dt = pred[key].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[key] = pred[key].astype(np.float16)

        try:
            name = data["name"][0]
            file_group = hfile.create_group(name)
            for key, value in pred.items():
                file_group.create_dataset(key, data=value)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file




