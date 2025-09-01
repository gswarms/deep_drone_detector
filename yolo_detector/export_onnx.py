import os
import shutil
from ultralytics import YOLO
import onnxruntime as ort

# Load your saved model
# pt_model_file_path = '../runs/detect/drone_detector_yolov8n/weights/best.pt'
pt_model_file_path = '/home/roee/Projects/deep_drone_detector/runs/detect/20250709_drone_detector_yolov8n3/weights/best.pt'


model = YOLO(pt_model_file_path)
model.export(format='onnx', simplify=True, device='cpu', imgsz=(256, 256), name='best')   # or 'tflite', 'coreml', etc.
res_file_name = '256'


filename, file_extension = os.path.splitext(pt_model_file_path)
onnx_model_file_path = filename + '.onnx'
onnx_model = ort.InferenceSession(onnx_model_file_path)
input_info = onnx_model.get_inputs()[0]
print('-----------------------------------------------')
print('onnx exported to: {}'.format(onnx_model_file_path))
print(input_info)

# quickly coy file before ultralytics export with (640X640) and overwrites it!
filename, file_extension = os.path.splitext(onnx_model_file_path)
onnx_model_file_path2 = filename + '_{}'.format(res_file_name) + file_extension
shutil.copyfile(onnx_model_file_path, onnx_model_file_path2)


# further documentatino of export functions: https://docs.ultralytics.com/modes/export/#arguments

# format - str. default='torchscript' Target format for the exported model, such as 'onnx', 'torchscript', 'engine' (TensorRT), or others.
#          Each format enables compatibility with different deployment environments.
# imgsz - int or tuple. default=640. Desired image size for the model input.
#         Can be an integer for square images (e.g., 640 for 640×640) or a tuple (height, width) for specific dimensions.
# keras - bool. default=False. Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.
# optimize - bool. default=False. Applies optimization for mobile devices when exporting to TorchScript,
#            potentially reducing model size and improving inference performance.
#            Not compatible with NCNN format or CUDA devices.
# half - bool. default=False. Enables FP16 (half-precision) quantization,
#        reducing model size and potentially speeding up inference on supported hardware.
#        Not compatible with INT8 quantization or CPU-only exports for ONNX.
# int8 - bool. default=False.
#        Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss,
#        primarily for edge devices. When used with TensorRT, performs post-training quantization (PTQ).
# dynamic -	bool. default=False. Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports,
#           enhancing flexibility in handling varying image dimensions. Automatically set to True when using TensorRT with INT8.
# simplify - bool. default=True. Simplifies the model graph for ONNX exports with onnxslim,
#            potentially improving performance and compatibility with inference engines.
# opset - int. default=None. Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes.
#         If not set, uses the latest supported version.
# workspace - float or None. default=None. Sets the maximum workspace size in GiB for TensorRT optimizations,
#             balancing memory usage and performance. Use None for auto-allocation by TensorRT up to device maximum.
# nms -	bool. default=False. Adds Non-Maximum Suppression (NMS) to the exported model when supported (see Export Formats),
#       improving yolo_detector post-processing efficiency. Not available for end2end models.
# batch - int. default=1. Specifies export model batch inference size or the maximum number of images
#         the exported model will process concurrently in predict mode. For Edge TPU exports, this is automatically set to 1.
# device - str. default=None. Specifies the device for exporting: GPU (device=0), CPU (device=cpu),
#          MPS for Apple silicon (device=mps) or DLA for NVIDIA Jetson (device=dla:0 or device=dla:1).
#          TensorRT exports automatically use GPU.
# data - str. default='coco8.yaml'.	Path to the dataset configuration file (default: coco8.yaml),
#        essential for INT8 quantization calibration. If not specified with INT8 enabled,
#        a default dataset will be assigned.
# fraction - float. default=1.0. Specifies the fraction of the dataset to use for INT8 quantization calibration.
#            Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited.
#            If not specified with INT8 enabled, the full dataset will be used.

model.export(format='onnx')   # or 'tflite', 'coreml', etc.
