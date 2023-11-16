# Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors

<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is Chao Lab's implementation of <br>
**Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors**<br>
[Yen-Cheng Liu](https://ycliu93.github.io/), [Chih-Yao Ma](https://chihyaoma.github.io/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/)<br>
The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2022 <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.pdf)] [[Project](https://ycliu93.github.io/projects/unbiasedteacher2.html)]

<p align="center">
<img src="teaser/teaser_utv2.png" width="85%">
</p>

# What have we changed?
- A hybrid implementation of features from unbiased-teacher and unbiased-teacher-v2
- Allows user input at run-time for labeled vs. unlabeled datasets
- Allows empty annos. in labeled datasets for control purposes
- More streamlined fully supervised training

# Work in progress
- Trying to make an effective inference pipeline
- Working on evaluation logic and model loading
- Moving away from hacking detectron2 to work with more image formats

# To Do
- Implement Mask-RCNN
- Support different dataset structures
- Remove temporary debugging hacks & hardcoding

# Feature wishlist
- Implement Mask-RCNN
- Transparency as to which data are being loaded at a given time
- Overfitting prevention
- Better visualizer during eval.
  


This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.
