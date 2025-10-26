import os.path

from openvino.tools import mo
from pathlib import Path

onnx_model_path = Path("/home/roee/Projects/deep_drone_detector/runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256.onnx")
output_dir = Path("/home/roee/Projects/deep_drone_detector/runs/detect/drone_detector_yolov11n_320x240_20251021/weights")

# Convert ONNX → OpenVINO IR
ir_model = mo.convert_model(
    input_model=onnx_model_path,
    framework="onnx",
    compress_to_fp16=True,   # optional, creates FP16 model (faster, smaller)
)

# Save to disk
output_dir.mkdir(parents=True, exist_ok=True)
file_name = os.path.basename(onnx_model_path)
xml_file_name = file_name[:-5] + '_openvino.xml'
from openvino.runtime import serialize
serialize(ir_model, str(output_dir / xml_file_name))

print("OpenVINO IR saved to:", output_dir)