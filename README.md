# Unsupervised Domain Adaptation for Object Detection

## Updates
- *04/2024*: Provide CycleGAN translated datasets.


## Installation
Our code is based on [Detectron latest(v0.6)](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), please install it before usage.

The following is an example based on PyTorch 1.9.0 with CUDA 11.1. For other versions, please refer to 
the official website of [PyTorch](https://pytorch.org/) and 
[Detectron](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
```shell
# create environment
conda create -n detection python=3.8.3
# activate environment
conda activate detection
# install pytorch 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# install detectron
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
# install other requirements
pip install -r requirements.txt
```

## Dataset

Following datasets can be downloaded automatically:
- [PASCAL_VOC 07+12](http://host.robots.ox.ac.uk/pascal/VOC/)
- Clipart
- WaterColor
- Comic

You need to prepare following datasets manually if you want to use them:

#### Cityscapes, Foggy Cityscapes
  - Download Cityscapes and Foggy Cityscapes dataset from the [link](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *leftImg8bit_trainvaltest.zip* for Cityscapes and *leftImg8bit_trainvaltest_foggy.zip* for Foggy Cityscapes.
  - Unzip them under the directory like

```
object_detction/datasets/cityscapes
├── gtFine
├── leftImg8bit
├── leftImg8bit_foggy
└── ...
```
Then run 
```
python prepare_cityscapes_to_voc.py 
```
This will automatically generate dataset in `VOC` format.
```
object_detction/datasets/cityscapes_in_voc
├── Annotations
├── ImageSets
└── JPEGImages
object_detction/datasets/foggy_cityscapes_in_voc
├── Annotations
├── ImageSets
└── JPEGImages
```

#### Sim10k
  - Download Sim10k dataset from the following links: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix). Particularly, we use *repro_10k_images.tgz* , *repro_image_sets.tgz* and *repro_10k_annotations.tgz* for Sim10k.
  - Extract the training set from *repro_10k_images.tgz*, *repro_image_sets.tgz* and *repro_10k_annotations.tgz*, then rename directory `VOC2012/` to `sim10k/`.
  
After preparation, there should exist following files:
```
object_detction/datasets/
├── VOC2007
│   ├── Annotations
│   ├──ImageSets
│   └──JPEGImages
├── VOC2012
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── clipart
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── watercolor
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── comic
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── cityscapes_in_voc
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── foggy_cityscapes_in_voc
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
└── sim10k
    ├── Annotations
    ├── ImageSets
    └── JPEGImages
```

**Note**: The above is a tutorial for using standard datasets. To use your own datasets, 
you need to convert them into corresponding format.

#### CycleGAN translated dataset

The following command use CycleGAN to translate VOC (with directory `datasets/VOC2007` and `datasets/VOC2012`) to Clipart (with directory `datasets/VOC2007_to_clipart` and `datasets/VOC2012_to_clipart`).
```
mkdir datasets/VOC2007_to_clipart
cp -r datasets/VOC2007/* datasets/VOC2007_to_clipart
mkdir datasets/VOC2012_to_clipart
cp -r datasets/VOC2012/* datasets/VOC2012_to_clipart

CUDA_VISIBLE_DEVICES=0 python cycle_gan.py \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --translated-source datasets/VOC2007_to_clipart datasets/VOC2012_to_clipart \
  --log logs/cyclegan_resnet9/translation/voc2clipart --netG resnet_9
```

You can also download and use datasets that are translated by us.

- PASCAL_VOC to Clipart [[07]](https://cloud.tsinghua.edu.cn/f/1b6b060d202145aea416/?dl=1)+[[12]](https://cloud.tsinghua.edu.cn/f/818dbd8e41a043fab7c3/?dl=1) (with directory `datasets/VOC2007_to_clipart` and `datasets/VOC2012_to_clipart`)
- PASCAL_VOC to Comic [[07]](https://cloud.tsinghua.edu.cn/f/89382bba64514210a9f8/?dl=1)+[[12]](https://cloud.tsinghua.edu.cn/f/f90289137fd5465f806d/?dl=1) (with directory `datasets/VOC2007_to_comic` and `datasets/VOC2012_to_comic`)
- PASCAL_VOC to WaterColor [[07]](https://cloud.tsinghua.edu.cn/f/8e982e9f21294b38be8a/?dl=1)+[[12]](https://cloud.tsinghua.edu.cn/f/b8235034cb4247ce809f/?dl=1) (with directory `datasets/VOC2007_to_watercolor` and `datasets/VOC2012_to_watercolor`)
- Cityscapes to Foggy Cityscapes [[Part1]](https://cloud.tsinghua.edu.cn/f/09ceeb25a476481bae29/?dl=1) [[Part2]](https://cloud.tsinghua.edu.cn/f/51fb05d3ee614e7d87a0/?dl=1) [[Part3]](https://cloud.tsinghua.edu.cn/f/646415daf6b344c3a9e3/?dl=1) [[Part4]](https://cloud.tsinghua.edu.cn/f/008d5d3c54344f83b101/?dl=1) (with directory `datasets/cityscapes_to_foggy_cityscapes`). Note that you need to use ``cat`` to merge the downloaded files.
- Sim10k to Cityscapes (Car) [[Download]](https://cloud.tsinghua.edu.cn/f/33ac656fcde34f758dcd/?dl=1) (with directory `datasets/sim10k2cityscapes_car`).


## Supported Methods

Supported methods include:

- [Cycle-Consistent Adversarial Networks (CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf)
- [Decoupled Adaptation for Cross-Domain Object Detection (D-adapt)](https://arxiv.org/abs/2110.02578)

## Experiment and Results

The shell files give the script to reproduce the [benchmarks](/docs/dalib/benchmarks/object_detection.rst) with specified hyper-parameters.
The basic training pipeline is as follows.

The following command trains a Faster-RCNN detector on task VOC->Clipart, with only source (VOC) data.
```
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/voc2clipart
```
Explanation of some arguments
- `--config-file`: path to config file that specifies training hyper-parameters.
- `-s`: a list that specifies source datasets, for each dataset you should pass in a `(name, path)` pair, in the
    above command, there are two source datasets **VOC2007** and **VOC2012**.
- `-t`: a list that specifies target datasets, same format as above.
- `--test`: a list that specifiers test datasets, same format as above.



### Visualization
We provide code for visualization in `visualize.py`. For example, suppose you have trained the source only model 
of task VOC->Clipart using provided scripts. The following code visualizes the prediction of the 
detector on Clipart.
```shell
CUDA_VISIBLE_DEVICES=0 python visualize.py --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  --test Clipart datasets/clipart --save-path visualizations/source_only/voc2clipart \
  MODEL.WEIGHTS logs/source_only/faster_rcnn_R_101_C4/voc2clipart/model_final.pth
```
Explanation of some arguments
- `--test`: a list that specifiers test datasets for visualization.
- `--save-path`: where to save visualization results.
- `MODEL.WEIGHTS`: path to the model.

## TODO
Support methods: SWDA, Global/Local Alignment

## Citation
If you use these methods in your research, please consider citing.

```
@inproceedings{jiang2021decoupled,
  title     = {Decoupled Adaptation for Cross-Domain Object Detection},
  author    = {Junguang Jiang and Baixu Chen and Jianmin Wang and Mingsheng Long},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{CycleGAN,
    title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
    author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
    booktitle={ICCV},
    year={2017}
}
```
