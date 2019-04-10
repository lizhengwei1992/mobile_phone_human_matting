# mobile_phone_human_matting
This project provides training and testing codes to build real_time human mattig on mobile phone only based ***CPU***. Anather repo ([Fast_Portrait_Segmentation](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation)) show more details about whence the project.

# requirements
- python3.5 / 3.6
- pytorch 0.4/0.4.1
- opencv-python

# Usage 

## train

you need to prepare [dataset](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/tree/master/dataset) and run ```./train.sh```.

## test
use pre_trained model ```./pre_train/erd_seg_matting```.

test camera, you need a camera, run ```./camera.sh```.
test image, run ```./test.sh```.

# Speed Analysis

Platform    : [***ncnn***](https://github.com/Tencent/ncnn).

(use this [tools](https://github.com/starimeL/PytorchConverter) convert the pytorch model to ncnn.)

Mobile phone: Samsung Galaxy S8+(cpu).


|            | model size (M) | time(ms)      | 
| ---------- | :-----------:  | :-----------: |
| erd_seg_matting    |          3.4     |     ~40          |

Demo video on my iphone 6 ([baiduyun](https://pan.baidu.com/s/1nieS7dSMw6Kwzsa1dz4egA))


# References

## paper
- [1]  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [2] [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
- [3] [ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
- [4] [ShuffleSeg: Real-time Semantic Segmentation Network](https://arxiv.org/pdf/1803.03816.pdf)
## repos
- [pytorch-fast-matting-portrait](https://github.com/huochaitiantang/pytorch-fast-matting-portrait)
- [ESPNet](https://github.com/sacmehta/ESPNet)
- [Fast_Portrait_Segmentation](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation)
- [Semantic_Human_Matting](https://github.com/lizhengwei1992/Semantic_Human_Matting)





