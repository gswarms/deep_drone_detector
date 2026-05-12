import torch
from nanodet_detector.model.arch import build_model
from nanodet_detector.util import load_config, load_model_weight

# Load config
cfg = load_config("config/nanodet_detector-m-416.yml")  # Match the ckpt config
cfg.device = torch.device("cpu")  # or 'cuda'

# Build model
model = build_model(cfg.model)
load_model_weight(model, "nanodet_m.ckpt", cfg.device)

model.eval()