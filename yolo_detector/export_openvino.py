import os.path
import openvino as ov
# from openvino.tools import mo
from pathlib import Path

# onnx_model_path = Path("/home/roee/Projects/deep_drone_detector/runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256.onnx")
# output_dir = Path("/home/roee/Projects/deep_drone_detector/runs/detect/drone_detector_yolov11n_320x240_20251021/weights")

onnx_model_path = Path('/home/roee/Projects/deep_drone_detector/yolo_detector/runs/drone_detector_yolov26n_256x256_20260119/weights/best_256x256.onnx')
output_dir = onnx_model_path.parent


# Convert ONNX → OpenVINO IR
# ir_model = ov.convert_model(
#     input_model=onnx_model_path,
#     framework="onnx",
#     compress_to_fp16=True,   # optional, creates FP16 model (faster, smaller)
# )

# # Save to disk
# output_dir.mkdir(parents=True, exist_ok=True)
# file_name = os.path.basename(onnx_model_path)
# xml_file_name = file_name[:-5] + '_openvino.xml'
# from openvino.runtime import serialize
# serialize(ir_model, str(output_dir / xml_file_name))
# print("OpenVINO IR saved to:", output_dir)


# 1. Load the ONNX model into memory
ov_model = ov.convert_model(onnx_model_path)

# 2. Save with FP16 compression
# compress_to_fp16=True is the default, but setting it explicitly is safer
output_dir.mkdir(parents=True, exist_ok=True)
xml_file_name = output_dir / (onnx_model_path.stem + '_openvino.xml')
ov.save_model(ov_model, xml_file_name, compress_to_fp16=True)

print("FP16 OpenVINO model saved successfully to: {}".format(xml_file_name))

# # Load ONNX-openvino model
# from openvino.runtime import Core
# output_model_xml_file = os.path.join(output_dir, xml_file_name)
# output_model_bin_file = output_model_xml_file.replace('.xml','.bin')
#
# ie = Core()
# model = ie.read_model(model=output_model_xml_file, weights=output_model_bin_file)
#
# # Inspect ops / types on the uncompiled model
# # for op in model.get_ops():
# #     for i, out in enumerate(op.outputs):
# #         print(op.get_type_name(), op.get_friendly_name(), out.get_element_type())
#
# compiled_model = ie.compile_model(model=model, device_name="CPU")
# print(compiled_model.input(0))
# print(compiled_model.output(0))
#
#
core = ov.Core()
model = core.read_model(xml_file_name)

# Check the precision of the first parameter/constant
for node in model.get_ops():
    if node.get_type_name() == "Constant":
        print(f"Weights precision: {node.get_element_type()}")
        break