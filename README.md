# Unsupervised Cross-Domain Object Detection

## Updates
- *12/2024*: Unsupervised Cross-Domain Object Detection based on Dynamic Smooth Cross Entropy.



## Installation
Our code is based on Decoupled Adaptation, see https://github.com/thuml/Decoupled-Adaptation-for-Cross-Domain-Object-Detection for details.



## Revision
We primarily modified files such as category_adaptation.py and fast_rcnn.py (Transfer-Learning-Library\tllib\alignment\d_adapt\modeling\roi_heads
\fast_rcnn.py), with minimal changes to the overall framework. Our main goal was to improve the accuracy and utilization of pseudo-labels. To 
address this, we applied data augmentation with varying intensities before the category adapter processes the data. Additionally, during the
 pseudo-label self-training phase, we made adjustments to the loss function. For detailed information, please refer to the paper "Unsupervised
 Cross-Domain Object Detection based on Dynamic Smooth Cross Entropy."

## Experiment and Results

The following command trains a Faster-RCNN detector on task VOC->Clipart, with only source (VOC) data.
```
# Source_only Stage
CUDA_VISIBLE_DEVICES=0 python source_only.py \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 datasets/VOC2007 VOC2012 datasets/VOC2012 -t Clipart datasets/clipart \
  --test VOC2007Test datasets/VOC2007 Clipart datasets/clipart --finetune \
  OUTPUT_DIR logs/source_only/faster_rcnn_R_101_C4/voc2clipart

# ResNet101 Based Faster RCNN: Faster RCNN: VOC->Clipart
# 44.8 -> 47.6(47.1,47.5)
pretrained_models=../logs/source_only/faster_rcnn_R_101_C4/voc2clipart_trans/model_0017999.pth
CUDA_VISIBLE_DEVICES=1 python d_adapt.py  \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine  \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase1_swda_dsce \
  MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase1_swda_dsce/model_0003999.pth
 CUDA_VISIBLE_DEVICES=1 python d_adapt.py --confidence-ratio-c 0.1 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase2_swda_dsce MODEL.WEIGHTS ${pretrained_models} SEED 0

pretrained_models=logs/faster_rcnn_R_101_C4/voc2clipart/phase2_swda_dsce/model_0003999.pth
CUDA_VISIBLE_DEVICES=0 python d_adapt.py --confidence-ratio-c 0.2 \
  --config-file config/faster_rcnn_R_101_C4_voc.yaml \
  -s VOC2007 ../datasets/VOC2007 VOC2012 ../datasets/VOC2012  \
  -t Clipart ../datasets/clipart --test Clipart ../datasets/clipart \
  --finetune --bbox-refine \
  OUTPUT_DIR logs/faster_rcnn_R_101_C4/voc2clipart/phase3_swda_dsce MODEL.WEIGHTS ${pretrained_models} SEED 0
```
Explanation of some arguments
- `--config-file`: path to config file that specifies training hyper-parameters.
- `-s`: a list that specifies source datasets, for each dataset you should pass in a `(name, path)` pair, in the
    above command, there are two source datasets **VOC2007** and **VOC2012**.
- `-t`: a list that specifies target datasets, same format as above.
- `--test`: a list that specifiers test datasets, same format as above.

### VOC->Clipart

|                         |          | AP   | AP50 | AP75 | aeroplane | bicycle | bird | boat | bottle | bus  | car  | cat  | chair | cow  | diningtable | dog  | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor |
|-------------------------|----------|------|------|------|-----------|---------|------|------|--------|------|------|------|-------|------|-------------|------|-------|-----------|--------|-------------|-------|------|-------|-----------|
| Faster RCNN (ResNet101) | Source   | 14.9 | 29.3 | 12.6 | 29.6      | 38.0    | 24.7 | 21.7 | 31.9   | 48.0 | 30.8 | 15.9 | 32.0  | 19.2 | 18.2        | 12.1 | 28.2  | 48.8      | 38.3   | 34.6        | 3.8   | 22.5 | 43.7  | 44.0      |
|                         | D-adapt  | 24.8 | 49.0 | 21.5 | 56.4      | 63.2    | 42.3 | 40.9 | 45.3   | 77.0 | 48.7 | 25.4 | 44.3  | 58.4 | 31.4        | 24.5 | 47.1  | 75.3      | 69.3   | 43.5        | 27.9  | 34.1 | 60.7  | 64.0      |
|                         |  DSCE    | 27.4 | 51.6 | 25.1 | 56.7      | 71.1    | 40.7 | 31.2 | 46.5   | 80.0 | 57.4 | 40.0 | 38.6  | 60.3 | 24.5        | 40.8 | 38.9  | 90.9      | 66.7   | 46.1        | 20.6  | 34.1 | 71.3  | 74.8      |
|                         |          |      |      |      |           |         |      |      |        |      |      |      |       |      |             |      |       |           |        |             |       |      |       |           |


### VOC->Comic

|                         |  AP  | AP50 | AP75 | bicycle | bird |  car |  cat |  dog | person |
|:-----------------------:|:----:|:----:|:----:|:-------:|:----:|:----:|:----:|:----:|:------:|
| Faster RCNN (ResNet101) | 13.0 | 25.5 | 11.4 |   33.0  | 15.8 | 28.9 | 16.8 | 19.6 |  39.0  |
|         D-adapt         | 20.8 | 41.1 | 18.5 |   49.4  | 25.7 | 43.3 | 36.9 | 32.7 |  58.5  |
|          DSCE           | 21.5 | 43.1 | 17.9 |   55.3  | 25.7 | 46.3 | 36.9 | 36.0 |  58.5  |
