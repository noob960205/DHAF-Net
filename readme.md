# DHAF-Net

Official PyTorch implementation of “DHAF-Net: Decoupled and Hierarchical Attention Fusion Network for RGB-Infrared Object Detection”

[Paper]()

🔒 The core implementation folder **`ultralytics/`** will be released after the paper is accepted.

## Abstract

融合可见光（RGB）与红外（IR）图像是实现全天候鲁棒目标检测的关键技术。然而，现有的多模态融合方法在特征交互过程中常常面临模态不平衡、信息冗余、模态干扰以及模态特有信息被抑制等问题。为应对这些挑战，本文提出了一种新颖的解耦与分层注意力融合网络（Decoupled and Hierarchical Attention Fusion Network, DHAF-Net），重新定义了多模态特征融合的范式。具体而言，DHAF-Net引入了一个特征解耦框架，将跨模态信息显式地分解为互补的“模态特性（Specificity）”与“模态共性（Commonality）”两个分支，从而最大化地保留并利用各模态信息。在此基础上，本文设计了解耦与分层注意力融合模块（DHAF module），在多尺度层面实现特征的精细化增强与融合。该模块通过自注意力机制捕获并强化特性流内部的上下文依赖关系，同时利用交叉注意力机制促进共性流之间的对称交互与语义对齐。最后，引入一个轻量级的门控加权机制，对多路增强后的特征流进行自适应加权，有效缓解模态不平衡问题。在LLVIP、M^3^FD等公开数据集上的大量实验结果表明，DHAF-Net在性能上显著优于现有的多模态融合方法，达到了当前最优（state-of-the-art）水平。这充分验证了所提出的解耦与分层融合策略的有效性，并为多模态目标检测建立了新的性能基准。

## Overview

![model architecture](./model_architecture.png)

## Environment Setup

**1. Create a virtual environment**

```bash
conda create -n dhaf python=3.8
conda activate dhaf
```

**2. Install PyTorch**

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**3. Install YOLOv8 dependencies**

Copy the ultralytics folder and the pyproject.toml file to your project root directory, then run:

```bash
pip install -e .
```

## Code Modifications

DHAF-Net is implemented based on the Ultralytics YOLOv8 framework with the following modifications:

- Data loading
	- ultralytics.data.base.BaseDataset.get_image_and_label
	- ultralytics.data.utils.img2label_paths
- Model initialization
	- ultralytics.nn.tasks.DetectionModel.\_\_init\_\_
- Custom modules
	- ultralytics/nn/modules/conv.py
	- ultralytics/nn/modules/block.py
- Others…


## Dataset & Weights

Please convert datasets to YOLO format and organize them as follows:

```
FLIR-aligned / LLVIP / M3FD / ...
    │
    ├─imageIR
    │    ├─test
    │    └─train
    ├─imageRGB
    │    ├─test
    │    └─train
    └─labels
         ├─test
         ├─train
         └─classes.txxt
```

Dataset & weights download link: 

- [Baidu drive](https://pan.baidu.com/s/1LgY7_Xs86yyOJX_olyyikg?pwd=dhaf)
- [Google drive]()

## Training

To start training:

```bash
python train.py
```

You can modify training configurations (dataset path, model architecture, hyperparameters) in the YAML or configuration files.

**Evaluation & Prediction**

```
python validate.py
python predict.py
```

## Results

<table border="1" cellpadding="5" cellspacing="0" style="text-align: center;">
    <thead>
        <tr>
            <th rowspan="2">Methods</th>
            <th rowspan="2">Pub.</th>
            <th rowspan="2">Modality</th>
            <th rowspan="2">Backbone</th>
            <th colspan="3">FLIR-Aligned</th>
            <th colspan="3">LLVIP</th>
        </tr>
        <tr>
            <th>AP50</th>
            <th>AP75</th>
            <th>mAP</th>
            <th>AP50</th>
            <th>AP75</th>
            <th>mAP</th>
        </tr>
    </thead>
    <tbody>
        <!-- 示例数据行 - 您可以根据需要添加更多行 -->
        <tr>
            <td>Faster R-CNN</td>
            <td>15</td>
            <td>IR</td>
            <td>ResNet50</td>
            <td>73.4</td>
            <td>34.2</td>
            <td>37.9</td>
            <td>92.6</td>
            <td>48.8</td>
            <td>50.7</td>
        </tr>
        <tr>
            <td>YOLOv5</td>
            <td>20</td>
            <td>IR</td>
            <td>CSPDarknet53v5</td>
            <td>73.9</td>
            <td>35.7</td>
            <td>39.5</td>
            <td>94.6</td>
            <td>72.2</td>
            <td>61.9</td>
        </tr>
        <tr>
            <td>YOLOv8</td>
            <td>23</td>
            <td>IR</td>
            <td>CSPDarknet53v8</td>
            <td>72.9</td>
            <td>34.8</td>
            <td>38.3</td>
            <td>95.2</td>
            <td>72.5</td>
            <td>62.1</td>
        </tr>
        <tr>
            <td>Faster R-CNN</td>
            <td>15</td>
            <td>RGB</td>
            <td>ResNet50</td>
            <td>65.0</td>
            <td>22.8</td>
            <td>30.2</td>
            <td>88.8</td>
            <td>45.7</td>
            <td>47.5</td>
        </tr>
        <tr>
            <td>YOLOv5</td>
            <td>20</td>
            <td>RGB</td>
            <td>CSPDarknet53v5</td>
            <td>67.8</td>
            <td>25.9</td>
            <td>31.8</td>
            <td>90.8</td>
            <td>51.9</td>
            <td>50.0</td>
        </tr>
        <tr>
            <td>YOLOv8</td>
            <td>23</td>
            <td>RGB</td>
            <td>CSPDarknet53v8</td>
            <td>66.3</td>
            <td>25.0</td>
            <td>28.2</td>
            <td>91.9</td>
            <td>53.0</td>
            <td>54.0</td>
        </tr>
        <tr>
            <td>GAFF <a href="https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf">[47]</a></td>
            <td>WWACV'21</td>
            <td>IR+RGB</td>
            <td>Resnet18</td>
            <td>72.9</td>
            <td>32.9</td>
            <td>37.5</td>
            <td>94.0</td>
            <td>60.2</td>
            <td>55.8</td>
        </tr>
        <tr>
            <td>ProbEn <a href="https://arxiv.org/pdf/2104.02904">[48]</a></td>
            <td>ECCV'22</td>
            <td>IR+RGB</td>
            <td>Resnet50</td>
            <td>75.5</td>
            <td>31.8</td>
            <td>37.9</td>
            <td>93.4</td>
            <td>50.2</td>
            <td>51.5</td>
        </tr>
        <tr>
            <td>CSAA <a href="https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf">[49]</a></td>
            <td>CVPR'23</td>
            <td>IR+RGB</td>
            <td>Resnet50</td>
            <td>79.2</td>
            <td>37.4</td>
            <td>41.3</td>
            <td>94.3</td>
            <td>66.6</td>
            <td>59.2</td>
        </tr>
        <tr>
            <td>CrossFormer <a href="https://www.sciencedirect.com/science/article/abs/pii/S016786552400045X">[50]</a></td>
            <td>PRL'24</td>
            <td>IR+RGB</td>
            <td>Resnet50</td>
            <td>79.3</td>
            <td>38.5</td>
            <td>42.1</td>
            <td>97.4</td>
            <td>75.4</td>
            <td>65.1</td>
        </tr>
        <tr>
            <td>RSDet <a href="https://arxiv.org/pdf/2401.10731">[51]</a></td>
            <td>24</td>
            <td>IR+RGB</td>
            <td>Resnet50</td>
            <td>83.9</td>
            <td>40.1</td>
            <td>43.8</td>
            <td>95.8</td>
            <td>70.4</td>
            <td>61.3</td>
        </tr>
        <tr>
            <td>Fusion-DETR <a href="https://ieeexplore.ieee.org/abstract/document/10929712/">[52]</a></td>
            <td>25</td>
            <td>IR+RGB</td>
            <td>Resnet101</td>
            <td>81.5</td>
            <td>-</td>
            <td>44.3</td>
            <td>96.4</td>
            <td>-</td>
            <td>64.6</td>
        </tr>
        <tr>
            <td>CFT <a href="https://arxiv.org/pdf/2111.00273">[53]</a></td>
            <td>21</td>
            <td>IR+RGB</td>
            <td>CSPDarknet53v5</td>
            <td>78.7</td>
            <td>35.5</td>
            <td>40.2</td>
            <td>97.5</td>
            <td>72.9</td>
            <td>63.6</td>
        </tr>
        <tr>
            <td>YOLO-MS <a href="https://ieeexplore.ieee.org/abstract/document/10021826">[54]</a></td>
            <td>TCDS'23</td>
            <td>IR+RGB</td>
            <td>CSPDarknet53v5</td>
            <td>75.2</td>
            <td>-</td>
            <td>38.3</td>
            <td>94.9</td>
            <td>-</td>
            <td>60.2</td>
        </tr>
        <tr>
            <td>ICAFusion <a href="https://www.sciencedirect.com/science/article/pii/S0031320323006118">[55]</a></td>
            <td>PR'24</td>
            <td>IR+RGB</td>
            <td>CSPDarknet53v5</td>
            <td>79.2</td>
            <td>36.9</td>
            <td>41.4</td>
            <td>95.2</td>
            <td>-</td>
            <td>60.1</td>
        </tr>
        <tr>
            <td>LRAF-Net <a href="https://ieeexplore.ieee.org/abstract/document/10144688">[56]</a></td>
            <td>TNNLS'24</td>
            <td>IR+RGB</td>
            <td>CSPDarknet53v5</td>
            <td>80.5</td>
            <td>-</td>
            <td>42.8</td>
            <td>97.9</td>
            <td>-</td>
            <td>66.3</td>
        </tr>
        <tr>
            <td rowspan="2">Fusion-Mamba <a href="https://ieeexplore.ieee.org/abstract/document/11124513">[57]</a></td>
            <td rowspan="2">TMM'25</td>
            <td rowspan="2">IR+RGB</td>
            <td>CSPDarknet53v5</td>
            <td>84.3</td>
            <td>-</td>
            <td>44.4</td>
            <td>96.8</td>
            <td>-</td>
            <td>62.8</td>
        </tr>
        <tr>
            <td>CSPDarknet53v8</td>
            <td>84.9</td>
            <td>45.9</td>
            <td>47.0</td>
            <td>97.0</td>
            <td>72.2</td>
            <td>64.3</td>
        </tr>
        <tr>
            <td>DHAF-Net (ours)</td>
            <td>-</td>
            <td>IR+RGB</td>
            <td>CSPDarknet53v8</td>
            <td>82.1</td>
            <td>48.5</td>
            <td>48.1</td>
            <td>97.7</td>
            <td>75.7</td>
            <td>67.4</td>
        </tr>
        <!-- 可以继续添加更多行 -->
    </tbody>
</table>
Please refer to the `./runs/detect` directory for training results.

## Citation

If you find this work useful in your research, please consider citing our paper:

```

```

## Acknowledgements

This project is built upon the excellent [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework. We gratefully acknowledge their open-source contributions.