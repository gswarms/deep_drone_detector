""" convert checkpoint to .pth
    Used for loading initial parameters for nanodet_detector (and maybe others)
"""
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.serialization import safe_globals


ckpt_input_file = '/home/roee/Projects/nanodet/models/pretrained/nanodet-plus-m_320_checkpoint.ckpt'
pth_output_file = '/home/roee/Projects/nanodet/models/pretrained/nanodet_detector-plus-m_320_checkpoint_converted.pth'


# Allow loading of Lightning's checkpoint class
with safe_globals([ModelCheckpoint]):
    # ckpt = torch.load(ckpt_input_file, weights_only=False)
    ckpt = torch.load(ckpt_input_file, map_location=torch.device('cpu'), weights_only=False)

# Extract weights
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Wrap in NanoDet-compatible format
# converted = {"net": state_dict}
# Create a fake checkpoint with required keys
# Create a fake checkpoint with required keys
# === Build NanoDet-compatible checkpoint ===
converted = {
    "state_dict": state_dict,
    "epoch": 0,
    "iter": 0,
    "optimizer": None,
    "meta": {
        "architecture": "NanoDet-Plus",
        "version": "0.5.0"
    }
}

# Save new .pth file
torch.save(converted, pth_output_file)
print(f"Saved NanoDet-compatible model to {pth_output_file}")
