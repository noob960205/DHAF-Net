# 环境安装

conda create -n v8 python=3.8
conda activate v8

**安装pytorch**

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda list

**安装YOLOv8所需依赖**

将官方YOLOv8的ultralytics文件夹与pyproject.toml文件拷贝到项目目录
pip install -e .

**源码修改**

 - 修改数据加载方法（ultralytics.data.base.BaseDataset.get_image_and_label）
 - 修改标签加载方法（ultralytics.data.utils.img2label_paths）
 - 修改检测模型初始化方法（ultralytics.nn.tasks.DetectionModel.__init__）
 - 添加自定义模块（ultralytics/nn/modules/conv.py、……）
 - 关闭预训练权重

pip install grad-cam==1.4.8 -i https://pypi.tuna.tsinghua.edu.cn/simple




## Dataset

将数据集格式转为YOLO格式

## LLVIP

### Directory Structure

LLVIP
  ├─Annotations
  ├─infrared
  │  ├─test
  │  └─train
  └─visible
        ├─test
        └─train

### Statistics

Our LLVIP dataset contains 30976 images (15488 pairs), 12025 pairs for train and 3463 for test.

The same pair of visible and infrared images share the same annotation, and they have the same name.

The labels are in VOC format.

### More

For more informations, please go to our [homepage](https://bupt-ai-cz.github.io/LLVIP/) and [github](https://github.com/bupt-ai-cz/LLVIP).

If you use this data for your research, please cite our [paper](https://arxiv.org/abs/2108.10831):

```
@misc{https://doi.org/10.48550/arxiv.2108.10831,
  doi = {10.48550/ARXIV.2108.10831}, 
  url = {https://arxiv.org/abs/2108.10831},
  author = {Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Liu, Shengjie and Zhou, Wenli}, 
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```





## Training

Separate the model and training  

1. Create an "empty" YOLO object. Assign it a standard model name, but we won’t actually use this model for training—it’s just a "launcher" to call the `.train()` method.  
2. Manually create an instance of the DetectionModel and explicitly pass `ch=6`.  
3. Call the `.train()` method and pass the constructed DetectionModel instance via the `model` parameter.

> This is because when using YOLOv8 for training, the two lines of code in `train.py`:  
>
> ```python
> model = YOLO()  
> results = model.train()  
> ```
>
> In the first line, `ultralytics.models.yolo.model.YOLO` inherits from `ultralytics.engine.model.Model`. When instantiating `model = YOLO()`, the parent class `Model`'s `__init__` initialization method is executed first. In this initialization method:  
>
> ```python
> if str(model).endswith((".yaml", ".yml")):  
>     self._new(model, task=task, verbose=verbose)  
> ```
>
> The code jumps to `ultralytics.engine.model.Model._new`, executing:  
>
> ```python
> self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  
> ```
>
> Eventually, it jumps to `ultralytics.nn.tasks.DetectionModel.__init__` for initialization. However, the `__init__` method of the `DetectionModel` class has a hardcoded default parameter `ch=3`, which conflicts with the number of channels we need for building a dual-stream/multi-modal model.









YOLOV8关闭预训练权值：https://zhuanlan.zhihu.com/p/661634636

YOLOV8模型训练、验证、预测与导出：https://zhuanlan.zhihu.com/p/717237053



https://github.com/CVandDetect/Cross-Modality-Detect