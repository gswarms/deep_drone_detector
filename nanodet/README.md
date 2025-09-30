# Nanodet utils

This subfiolder helps use nanodet.


---

## 🛠️ Nanodet Installation

Create a Python environment:
        
    conda create -n nanodet python=3.8 -y
    conda activate nanodet

Install dependencies:

    # clone nanodet repo
    git clone https://github.com/RangiLyu/nanodet.git
    cd nanodet

    # Install requirements
    pip install -r requirements.txt

    # Install NanoDet as package
    python setup.py develop

✅ PyTorch ≥1.8.0 is required. You might want to use PyTorch with CPU if you're working on Raspberry Pi.

## 📁 Prepare Your Dataset
NanoDet supports COCO format and VOC format.

Convert to COCO format. If your dataset isn't in COCO format already, use tools like:
- [Roboflow](https://roboflow.com/)
- [Labelme2coco](https://github.com/fcakyon/labelme2coco)
- FiftyOne / CVAT / [COCO-annotator](https://github.com/jsbroks/coco-annotator)

Folder Structure (COCO-style):

    datasets/
    └── mydata/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── annotations/
        │   ├── instances_train.json
        │   └── instances_val.json

## 🛠️ Modify Config File


NanoDet uses YAML config files to define model and dataset.

### Use a base config:

Copy the nanodet .yml - we used nanodet-plus-m_320.yml:

    cp config/nanodet-plus-m_320.yml config/mydata_config.yml

The model_size: 1.5x parameter in the NanoDet config (specifically for ShuffleNetV2) refers to the width multiplier used to scale the number of channels (filters) in the network’s convolutional layers.\
ShuffleNetV2 supports different scaling factors: 0.5x, 1.0x, 1.5x, 2.0x\
The larger the number, the wider the network, meaning:
- More parameters
- More FLOPs
- Higher accuracy
- Slower inference and higher memory usage

We chose model_size: 1x


### Modify model section:

    model:
    arch:
        name: NanoDetPlus
        backbone:
        name: ShuffleNetV2
        model_size: 1.0x
        fpn:
          name: PAN
          in_channels: [116, 232, 464]
          out_channels: 96
        head:
          name: NanoDetPlusHead
          num_classes: 1  # Your number of classes (e.g., 1)
        ...
        ...
        ...
        # Auxiliary head, only use in training time.
        aux_head:
          name: SimpleConvHead
          num_classes: 1  # Your number of classes (e.g., 1)
        ...
        ...
        ...

### Modify class names:
    
    class_names: ['rc-plane']  # changed class name

### Modify dataset section:

    data:
      train:
        name: CocoDataset
        ann_file: datasets/mydata/annotations/instances_train.json
        img_path: datasets/mydata/images/train
      val:
        name: CocoDataset
        ann_file: datasets/mydata/annotations/instances_val.json
        img_path: datasets/mydata/images/val
    
      input_size: [320, 320]  # Match model input
      keep_ratio: False
      mean: [103.53, 116.28, 123.675]
      std: [57.375, 57.12, 58.395]


### set device load:

    device:
      gpu_ids: [0] # Set like [0, 1, 2, 3] if you have multi-GPUs
      workers_per_gpu: 10
      batchsize_per_gpu: 4  # Adjust this to fit your GPU (96->4)
      precision: 32 # set to 16 to use AMP training

### set transfer learning

    trainer:
      freeze_backbone: False  # Set to True if dataset is small

### Load pre-trained weights:
good for transfer training, but only if you keep the original input size.
There are two options:
- load_model:	Loads weights only (transfer learning)	✅ Fine-tuning on new dataset\

        load_model: checkpoints/nanodet-plus-m_320.pth  # download from NanoDet releases
        #  resume:  # This should stay as remark to use pretrained models, but don't resume checkpoint!

- resume:	Resumes full training (including optimizer state)	Resuming from interrupted training

        # load_model:
        resume: checkpoints/converted_model.pth

For transfer learning use `load model`

You can download the pretrained NanoDet-Plus-m weights from:\
https://github.com/RangiLyu/nanodet/releases

Nanodet uses .pth files for loading a model. If you have a .ckpt file, you need to convert it:\
Install PyTorch Lightning (to enable python to read .ckpt file) 

    pip install pytorch-lightning
Or if you're using conda:

    conda install -c conda-forge pytorch-lightning

Now convert .ckpt to .pth using python:
            
    import torch
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    from torch.serialization import safe_globals
    
    # Paths
    ckpt_input_file = "your_model.ckpt"
    output_file = "converted_model.pth"
    
    # Load Lightning checkpoint
    with safe_globals([ModelCheckpoint]):
        ckpt = torch.load(ckpt_input_file, map_location="cpu", weights_only=False)
    
    # Extract weights
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
    # Create fully NanoDet-compatible checkpoint
    converted = {
        "state_dict": state_dict,
        "epoch": 0,
        "iter": 0,                 # ✅ ADD THIS LINE
        "optimizer": None,
        "meta": {
            "architecture": "NanoDet-Plus",
            "version": "0.5.0"
        }
    }
    
    # Save it
    torch.save(converted, output_file)
    print(f"✅ Final checkpoint saved to {output_file}")

    

### freeze_backbone 
for transfer training.

    freeze_backbone: False  # Set to True if dataset is small


## 🚀 Train the Model

    python tools/train.py config/mydata_config.yml

By default, it saves models to workspace/.\
Checkpoint structure:

    workspace/
    └── mydata/
        ├── model_best.pth
        └── epoch_XX.pth
You can resume training or continue fine-tuning from any of these.


## ✅ Evaluate & Export
Evaluate model on validation set:

    python tools/eval.py config/mydata_config.yml --model_path workspace/mydata/model_best.pth

NanoDet supports ONNX and NCNN exports:

    python tools/export.py config/mydata_config.yml --model_path workspace/mydata/model_best.pth --backend onnx

or

    python tools/export_onnx.py \
        --cfg config/nanodet-plus-m_320_lulav_dit.yml \
        --model_path workspace/nanodet-plus-m_320/model_best/model_best.pth \
        --out_path workspace/nanodet-plus-m_320/nanodet.onnx \
        --input_shape  320,320
