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

### 数据集结构
使用的数据集为VirusShare_00177，可以在[VirusShare网站](https://virusshare.com/)下载得到。训练/测试依赖多个数据文件，这些文件都按照以下结构存放在以“数据集名字”为名的文件夹中，一个数据集对应一个文件夹。数据集文件夹整体的结构如下：

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

### 标注恶意代码家族
由于下载的恶意代码源数据中没有家族标签，为了完成分类任务的监督学习需要对所有恶意代码样本进行标注，采用的标注方法是上传至[VirusTotal网站](https://support.virustotal.com/)进行扫描。扫描结果将会以JSON文件返回，内部包含了许多家公司的反恶意扫描引擎的扫描结果。由于各个公司的引擎之间的家族命名缺少一个统一的标准，因此需要使用[AVClass工具](https://github.com/malicialab/avclass)来从扫描结果中抽取一个唯一的家族名称作为恶意代码的ground-truth家族标签。最后统计家族数量的规模，只保留包含指定数目（实验中为20）以上个样本的家族，并将属于同一家族的样本收集到同一个文件夹中

### 序列数据处理
具体方法可以见最上方的基于API序列的参考文献。所有序列数据由[Cuckoo沙箱](https://cuckoosandbox.org/)运行恶意代码二进制文件后得到。Cuckoo沙箱在运行二进制文件后，会自动生成一个运行报告，其中只留下API调用的名称组装成一个序列，丢弃其他所有特征，存储为一个JSON文件。注意，太短的运行序列（例如长度小于10）的样本需要被弃用，因为太短的序列没有包含太多有意义的信息，通常意味着运行失败。处理流程大致如下：

1. 移除API序列中重复调用超过2次造成的冗余子序列，只保留最多2次调用，可参考代码中 *preprocessing/normalization.py/removeAPIRedundancy*的实现。该操作会覆盖掉原序列文件中的数据，注意备份
2. 提取序列的N-Gram（实验中N=3）序列并统计各个N-Gram子序列的频率，只保留top-l（实验中l=5000）的N-Gram子序列。同时为了防止因为生成的N-Gram的名称过长，将每一个保留的N-Gram项映射为一个int值，最后利用top-l的N-Gram序列int映射值替换原API序列。可参考代码中 *preprocessing/ngram.py/statNGramFrequency,mapAndExtractTopKNgram*的实现，前者主要是统计N-Gram频率并生成频率排序字典，后者完成实际的映射工作，本操作也是一个in-place操作，会覆盖源数据
3. 对于每一个标注的恶意代码家族，从每一个家族中采样一个固定数量的样本（实验中是20）来组成数据集，放置到上一节中介绍的数据集结构中的*all/api/*中，文件夹名称就为家族名称。可参见代码中 *preprocessing/dataset/sample.py/sampleClassWiseData*的实现
4. 对所有提取的序列数据运行GloVe算法来生成序列元素的词嵌入矩阵，可参见代码中 *utils/GloVe.py*中的实现，主要使用了glove-python库。生成的词嵌入矩阵向量存储为NumPy的NdArray类型，放置在 *data/matrix.npy*;”序列元素->矩阵下标值映射“存储为JSON类型，放置在*data/wordMap.json*下

### 图像数据处理
具体方法可以参考最上方的基于恶意代码灰度图的参考文献。主要是利用16进制读取二进制文件后，按顺序将每一个字节值（2个十六进制数）转换为一个0-255的灰度值，整个字节序列就转换为灰度值序列。然后将灰度值序列按照最大的平方值进行截断，转换为一个2D的正方形图像，最后上采样/下采样至一个指定大小的图像（实验中是256×256）。处理好的图像数据保存在上一节中介绍的 *all/img/*中。可参考代码中 *preprocessing/image.py*的实现。

![](files/malware_image_conversion.jpg)

### 数据集分割
由于元学习的特殊性，数据集分割需要将恶意代码家族（类）分割，而不是类中的样本。分割前所有恶意代码家族都位于 *all*文件夹中，分割时需要把整个家族的数据移动到train，validate或者test文件夹中，而且需要同时将API序列数据和恶意代码灰度图数据对应分割移动。具体的代码实现可以参考 *preprocessing/dataset/split.py/splitDataset*。分割后，训练集的所有数据位于 *train*文件夹中，验证集和测试集分别位于 *validate*和*test*文件夹中

### 数据打包
为了减少因为文件IO造成的处理延时增加，在训练/测试之前都先将所需数据一次性读入内存中，因此需要将数据打包以方便读取。打包时按照数据子集为单位（train，validate，test）为单位，每一个子集采用相同的打包方式，数据分别存储在数据集中 *data*目录下不同子集的文件夹中。

对于序列数据，打包过程主要包括：
- 对齐序列长度：指定一个序列最大长度t，对于长度超过*t*的序列做截断操作，对长度小于*t*的序列使用0填充到指定长度
- 统计序列长度：统计每一个序列的实际长度，以便于LSTM模型进行pack和unpack操作。长度文件存储为 *seqLength.json*
- 映射序列元素到词嵌入下标：使用词嵌入时生成的wordMap.json，将序列数据映射为词嵌入的下标数据，以便于词嵌入读取
  
序列数据打包好以后命名为 *api.npy*，放置在对应的子集文件夹下

对于图像数据，打包过程主要包括：
- 向量化图像：读取图像文件并将其转换为一个PyTorch的Tensor类型
- 标准化图像：按像素点统计所有图像数据的灰度均值和方差，利用均值和方差标准化图像
  
图像数据处理好以后命名为 *img.npy*，放置在对应的子集文件夹下

为了保证恶意代码家族的具体的可追溯性，在每一个子集打包时，每一个恶意代码家族在打包矩阵中的下标存储为 *idx_mapping.json*，放置在对应的子集文件夹下


## 模型说明
模型整体包括API序列嵌入结构，图像嵌入结构，特征融合模块和多原型生成算法，其中模型整体结构的运行流程结构如下所示（运行一个episode）：

![](files/classification_workflow_cut.PNG)

### 基础模型
基础模型包含所有模型都需要的一些基础能力，还声明了一些所有模型都需要的数据，例如损失函数，数据源，特征融合输出维度等。基础模型的代码实现参见 *model/common/base_model.py/BaseModel*

基础模型包含的功能包括：
- 解析数据中的Episode任务参数（K，C，N等），
- 调用嵌入结构嵌入序列和图像嵌入结构来嵌入输入数据，主要是按次序调用SeqEmbedPipeline和ImgEmbedPipeline，因此后续模型嵌入具体实现需要将组建放入这两个列表中

### 嵌入模型
嵌入模型继承了基础模型BaseModel，主要增加了模型需要的数据嵌入功能，能够根据配置参数来调整模型的实际使用子结构和超参数。嵌入模型的代码实现见 *model/common/base_embed_model.py/BaseEmbedModel*

嵌入模型的包含的功能包括：
- 根据数据源配置嵌入结构：如果数据源中不包括序列数据，则在初始化嵌入结构时不会初始化序列数据的嵌入结构，以减少显存开销。数据源的配置可以在训练参数配置文件 *config/train.json*中的“training | data_source"中设置
- 设置词嵌入初始化的使用：可以选择使用或者不使用词嵌入，在“model | embedding"中设置
- 设置序列嵌入结构和其超参数：在"model | sequence_backbone"中调整序列嵌入的参数，默认使用**LSTM+时序卷积最大池化**作为序列嵌入结构，可以通过该参数下的"type"子参数来调整其他结构（需要在BaseEmbedModel中实现）。序列长度通过参数下的"max_seq_len"来调整。其他参数，例如LSTM类型的参数，包括在了该参数下的"LSTM“子项，包括双向LSTM，隐藏层维度，LSTM层数等。如果需要实现新的模型，则参数可以自定义调整，只需要在BaseEmbedModel中自行判断读取即可
- 设置图像嵌入结构和其超参数：在“model | conv_backbone"中调整图像嵌入的参数，默认使用**Conv-4**结构作为图像嵌入结构，可以通过该参数下的“type"子参数来调整为其他结构，目前已实现的还有resnet18和resnet34，其他结构可以自行在BaseEmbedModel中判断并添加即可。Conv-4内的参数，包括是否使用global_pooling来将 *batch,channel,dimension* 形式的图像特征约减为一个特征向量，每一个卷积层的通道数量，卷积步幅stride，填充padding，是否使用非线性激活等，都包含在子参数“params | conv-n"中，如果需要自行实现新的模型和其参数，则在"params"下新加一个子参数项，然后在BaseEmbedModel初始化时自行读取即可
- 将序列嵌入和图像嵌入流程放入嵌入管道：按顺序将序列数据的嵌入子结构放入SeqEmbedPipeline中，将图像数据的嵌入子结构放入ImgEmbedPipeline中。例如，典型的一个序列嵌入管道为: self.Embedding -> self.EmbedDrop -> self.SeqEncoder -> self.SeqTrans
- 设置重投影模块：在序列和图像数据嵌入后分别使用两个线性层，重投影两部分特征到一个共同的维度上，可以用于统一两个特征的维度，也可以用于两个特征空间的重组
- 初始化特征融合模块：根据参数中的"model | fusion | type"初始化对应的特征融合模块。主要调用了buildFusion接口，见 *builder/fusion_builder.py/buildFusion*

