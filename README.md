# InterFormer
This repo is the official implementation of "InterFormer: Real-time Interactive Image Segmentation"

# Introduction
InterFormer follows a new pipeline to address the issues of existing pipeline's low computational efficiency. InterFormer extracts and preprocesses the computationally time-consuming part i.e. image processing from the existing process. Specifically, InterFormer employs a large vision transformer (ViT) on high-performance devices to preprocess images in parallel, and then uses a lightweight module called interactive multi-head self attention (I-MSA) for interactive segmentation. InterFormer achieves real-time high-quality interactive segmentation on CPU-only devices.

# Usage
## Install

### Requirements

- Python 3.8+
- PyTorch 1.12.0
- mmcv-full 1.6.0
- mmsegmentation 0.26.0

### Install PyTorch

please refer to [INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)

### Install mmcv-full
```shell
pip install -U openmim
mim install mmcv-full==1.6.0
```

### Install mmsegmentation
```shell
cd mmsegmentation
pip install -e .
```

## Data preparation

### COCO Dataset
please refer to [cocodataset](https://cocodataset.org/#download) to download [2017 Train Images](http://images.cocodataset.org/zips/train2017.zip), [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) and [2017 Stuff Train/Val annotations](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip) into `data`

or 

use the following script
```shell
cd data/coco2017
bash coco2017.sh
```

the data is organized as follows
```
data/coco2017/
├── annotations
│   ├── panoptic_train2017 [118287 entries exceeds filelimit, not opening dir]
│   ├── panoptic_train2017.json
│   ├── panoptic_train2017.zip
│   ├── panoptic_val2017 [5000 entries exceeds filelimit, not opening dir]
│   ├── panoptic_val2017.json
│   └── panoptic_val2017.zip
├── annotations_trainval2017.zip
├── coco2017.sh
├── panoptic_annotations_trainval2017.zip
├── train2017 [118287 entries exceeds filelimit, not opening dir]
└── val2017 [5000 entries exceeds filelimit, not opening dir]
```

### LVIS Dataset

please refer to [lvisdataset](https://www.lvisdataset.org/dataset) to download the images and annotations

data is organized as follows
```
data/lvis/
├── lvis_v1_train.json
├── lvis_v1_train.json.zip
├── lvis_v1_val.json
├── lvis_v1_val.json.zip
├── train2017 [118287 entries exceeds filelimit, not opening dir]
├── train2017.zip
├── val2017 [5000 entries exceeds filelimit, not opening dir]
└── val2017.zip
```

### SBD Dataset
please refer to [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) to download SBD dataset

the data is organized as follows
```
data/sbd/
├── benchmark_RELEASE
│   ├── dataset
│   │   ├── cls [11355 entries exceeds filelimit, not opening dir]
│   │   ├── img [11355 entries exceeds filelimit, not opening dir]
│   │   ├── inst [11355 entries exceeds filelimit, not opening dir]
│   │   ├── train.txt
│   │   └── val.txt
└── benchmark.tgz
```


### DAVIS & GrabCut & Berkeley Datasets

please download [DAVIS](https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip)
[GrabCut](https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip)
[Berkeley](https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip) from [Reviving Iterative Training with Mask Guidance for Interactive Segmentation](https://github.com/SamsungLabs/ritm_interactive_segmentation#reviving-iterative-training-with-mask-guidance-for-interactive-segmentation)

the data is organized as follows
```
data/davis/
└── DAVIS
    ├── gt [345 entries exceeds filelimit, not opening dir]
    ├── img [345 entries exceeds filelimit, not opening dir]
    └── list
        ├── val_ctg.txt
        └── val.txt
```

```
data/grabcut/
└── GrabCut
    ├── gt [50 entries exceeds filelimit, not opening dir]
    ├── img [50 entries exceeds filelimit, not opening dir]
    └── list
        └── val.txt
```

```
data/berkeley/
└── Berkeley
    ├── gt [100 entries exceeds filelimit, not opening dir]
    ├── img [100 entries exceeds filelimit, not opening dir]
    └── list
        └── val.txt
```

## Training

## Evaluation

# Results
Include the results reported in the paper, along with any additional results obtained by the implementation.