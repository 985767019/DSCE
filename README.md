# Unsupervised Cross-Domain Object Detection

## Updates
- *12/2024*: Unsupervised Cross-Domain Object Detection based on Dynamic Smooth Cross Entropy.


## Installation
Our code is based on Decoupled Adaptation, see https://github.com/thuml/Decoupled-Adaptation-for-Cross-Domain-Object-Detection for details.
```

## Revision
We primarily modified files such as category_adaptation.py and fast_rcnn.py, with minimal changes to the overall framework. 
Our main goal was to improve the accuracy and utilization of pseudo-labels. To address this, we applied data augmentation 
with varying intensities before the category adapter processes the data. Additionally, during the pseudo-label self-training
 phase, we made adjustments to the loss function. For detailed information, please refer to the paper "Unsupervised 
Cross-Domain Object Detection based on Dynamic Smooth Cross Entropy."



