# MalFusionFSL
基于静态/动态分析特征融合的小样本恶意代码分类框架(few-shot malware classification based on fused analysis features)的研究代码。
包含所有数据集处理，元学习任务级封装，提出模型和基线模型的细节实现，训练/测试的运行代码等。本项目的特点包括：

- **高度耦合分层的模型实现**：模型可以通过继承包含基础功能的父类来获得一些模型无关的基础能力，例如序列和图像嵌入，任务参数的解析等，使得新模型可以快速地进行开发
- **高层封装的episode元学习任务采样**：将元学习任务采样封装到task实例中，方便自定义采样规则和利用接口自动采样任务batch
- **完全参数化的模型训练和模型测试**：完全使用易读的JSON配置文件来指定模型的内部参数，例如隐藏层维度，超参数的设定，训练周期等，确保同等配置下的实验效果的可复现性
- **模块化的模型架构解析**：使用高度封装的builder接口来获取模型，优化器，特征融合模块等架构部件，隔离了运行脚本对部件的改动的感知，使得新模型添加或者模型改动可以很容易快速完成
- **详细的运行时和测试时数据记录**：使用dataset+version的方式来管理训练和测试运行实例，使得每一次运行都有训练/测试记录保存，方便后续统计实验效果或者复盘
- **丰富的参数支持**：大量的运行时参数允许调整，例如Visdom可视化，控制台打印运行情况log，梯度裁剪和GPU指定等
- **自动化任务运行支持**：支持使用自动执行机来流水线施添加预定运行任务，使得模型训练测试可以自动无人值守时完成

本项目中用到的动静态数据的简介如下：
- 静态分析数据：恶意程序二进制转灰度图，见: <br/>_Tang, Z.; Wang, P.; Wang, J. ConvProtoNet: Deep Prototype Induction towards Better Class Representation for Few-Shot Malware Classification. Appl. Sci. 2020, 10, 2847. https://doi.org/10.3390/app10082847_
- 动态分析数据：Cuckoo沙箱运行恶意程序二进制文件获取的API调用序列，见：<br/> _Wang, Peng, Zhijie Tang, and Junfeng Wang. "A Novel Few-Shot Malware Classification Approach for Unknown Family Recognition with Multi-Prototype Modeling." Computers & Security (2021): 102273. https://doi.org/10.1016/j.cose.2021.102273_

## 运行环境
### 硬件环境
- CPU：Intel Core i7-9700F
- GPU：GTX 1660Ti 6GB
- 内存：16 GB

### 软件环境
- 操作系统：Majanro Linux KDE 20.2
- Python：3.8
- Cuda: 11.2
- Pip包（见requirement.txt）：
  - **PyTorch: 1.8.0 (cu111 ver. ,默认使用cuda运行)**
  - glove-python-binary: 0.2.0
  - numpy: 1.19.2
  - visdom: 0.1.8.9
  - matplotlib: 3.3.2
  - Pillow: 8.0.1
  - tqdm
  - gpustat

## 运行说明
本项目的正确运行依赖于若干前置数据处理和参数配置操作

### 数据集制作
训练/测试依赖多个数据文件，这些文件都按照以下结构存放在以“数据集名字”为名的文件夹中，一个数据集对应一个文件夹。数据集文件夹整体的结构如下：

```
(数据集名称)
    |-- all 
        |-- img
        |-- api
    |-- train
        |-- img 
        |-- api 
    |-- validate
        |-- img 
        |-- api 
    |-- test
        |-- img 
        |-- api 
    |-- doc    
    |-- models
    |-- data
        |-- train
            |-- api.npy
            |-- img.npy
            |-- idx_mapping.json
            |-- seq_length.json
        |-- validate
            |-- api.npy
            |-- img.npy
            |-- idx_mapping.json
            |-- seq_length.json
        |-- test
            |-- api.npy
            |-- img.npy
            |-- idx_mapping.json
            |-- seq_length.json
        |-- wordMap.json
        |-- matrix.npy
```
其中各个文件/文件夹的用途如下：
- all文件夹：包含了训练集，验证集和测试集中所有图像和api数据，便于进行数据集划分和做预训练
- train文件夹：训练集。其下的img对应的是恶意代码灰度图数据，包含若干个文件夹，其中每一个文件夹以恶意代码家族命名，文件夹内的样本同属于该家族，api文件夹内同理，包含的家族与img内一一对应
- validate文件夹：验证集，结构与训练集相似，但是其中包含的恶意代码家族与训练集不相同
- test文件夹：测试集。结构与训练集和验证集相似，但是其中包含的恶意代码家族与前两者都不一样
- doc：存放训练时记录数据，每一次运行对应一个版本（version），以该版本为名，doc中的一个version对应一个文件夹，文件夹内包含该次运行的所有数据
- models：存放训练之后保存的模型，所有版本的模型不以文件夹区分，以命名区分，训练运行时也会保存，测试时从这里读取
- data文件夹：存放预处理的数据
  - wordMap.json: 对应API序列在预训练之后，每一个token对应的映射整型值，便于从字典矩阵中取出对应嵌入向量
  - matrix.npy: 预训练得到的API序列词嵌入字典，2D：[词数量, 维度]
  - train文件夹：包含训练集打包好的数据
    - api.npy：打包好的2D的API序列的token矩阵，其中每一个API序列都已经对齐到一个指定长度上（超过截断，不足补0），每一个API序列实际上是一个被映射的int值（与词嵌入的映射对应）。一次性读入内存减少IO等待
    - img.npy：打包好的3D的恶意代码图像矩阵，其中每一个图像都是一个指定的长宽（例如256×256）。一次性读入内存减少IO等待
    - idx_mapping：包含数据集中的恶意代码家族index与其对应的真实名称的映射
    - seq_length：API数据截断前的真实长度，用于RNN的pack和unpack，或者注意力中的mask
  - validate文件夹：包含验证集打包好的数据，结构与训练集中同理
  - test文件夹：包含测试集打包好的数据，结构与训练集中同理

## 模型说明
模型整体结构的运行流程结构如下所示（运行一个episode）：
![](files/classification_workflow_cut.PNG)
