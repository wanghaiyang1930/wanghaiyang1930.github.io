# 传统图像分割方案总结

## 概述

图像分割是计算机视觉中的基础任务，目标是将图像划分为具有语义意义的区域。传统图像分割方法经历了从经典计算机视觉算法到深度学习方法的演进。本文总结传统分割方案的输入、输出、监督方式以及常用网络架构。

## 分割任务类型

**语义分割(Semantic Segmentation)**：为每个像素分配类别标签，同类物体不区分实例。

**实例分割(Instance Segmentation)**：区分同类别的不同实例对象。

**全景分割(Panoptic Segmentation)**：结合语义分割和实例分割，统一处理"stuff"类(天空、道路)和"thing"类(车辆、行人)。

## 经典计算机视觉方法

在深度学习之前，传统方法主要基于像素特征和区域性质：

**基于阈值的方法**：
- 全局阈值、自适应阈值、Otsu算法
- 适用于背景简单、对比度高的场景

**基于区域的方法**：
- 区域生长(Region Growing)：从种子点开始，根据相似性准则合并相邻像素
- 分水岭算法(Watershed)：将图像视为地形表面，基于梯度进行区域划分
- 均值漂移(Mean Shift)：在特征空间中寻找密度峰值

**基于边缘的方法**：
- Canny边缘检测、Sobel算子等
- 需要后处理将边缘连接成封闭区域

**基于图论的方法**：
- 图割(Graph Cut)、归一化图割(Normalized Cut)
- 将分割问题建模为图的最小割问题

**基于聚类的方法**：
- K-means聚类：在颜色或特征空间中对像素进行聚类
- 超像素(Superpixel)：SLIC、Felzenszwalb等

**概率图模型**：
- 马尔可夫随机场(Markov Random Field, MRF)
- 条件随机场(Conditional Random Field, CRF)
- 建模像素之间的空间依赖关系

这些方法的局限性：依赖手工设计的特征，对光照变化、遮挡、复杂场景的鲁棒性较差。

## 深度学习语义分割方案

### 输入

**图像数据**：
- RGB三通道图像，通常需要归一化处理
- 输入尺寸：固定尺寸(如512×512、1024×2048)或任意尺寸
- 预处理：均值减法、标准化、数据增强(翻转、旋转、缩放、裁剪、色彩变换)

**多模态输入(部分方法)**：
- RGB-D：RGB图像+深度图
- RGB-T：RGB图像+热成像

### 输出

**像素级预测**：
- 每个像素的类别概率分布或类别标签
- 输出张量形状：[H, W, C]，其中C为类别数
- 通常输出logits或softmax概率图

**输出分辨率**：
- 与输入图像相同分辨率的密集预测
- 通过上采样(Upsampling)或反卷积(Transposed Convolution)恢复分辨率

### 监督方法

**标注数据**：
- 像素级密集标注：每个像素都有对应的类别标签
- 标注格式：通常为与输入图像同尺寸的标签图(Label Map)
- 标注成本：非常高，需要精细勾画每个对象的边界

**损失函数**：

**交叉熵损失(Cross-Entropy Loss)**：
- 最常用的损失函数
- 逐像素计算预测分布与真实标签的交叉熵
- 加权交叉熵：针对类别不平衡问题，对不同类别赋予不同权重

**Dice Loss**：
- 基于Dice系数(F1 score的变体)
- 直接优化分割质量指标
- 对小目标和类别不平衡更鲁棒
- 公式：Dice = 2|X∩Y| / (|X|+|Y|)

**Focal Loss**：
- 解决前景-背景严重不平衡问题
- 降低易分类样本的权重，关注难分类样本

**组合损失**：
- 常见组合：Cross-Entropy + Dice Loss
- Boundary Loss：额外关注边界区域的准确性

**弱监督/半监督方法**：
- 边界框标注(Bounding Box)
- 涂鸦标注(Scribble)
- 图像级标签(Image-level Labels)
- 这些方法降低标注成本，但性能通常低于全监督方法

### 网络架构

#### 1. 全卷积网络 (FCN, Fully Convolutional Networks)

**开创性工作**：
- 首次将CNN用于语义分割的端到端训练
- 用1×1卷积替换全连接层，实现任意尺寸输入
- 使用反卷积进行上采样恢复分辨率

**跳跃连接(Skip Connections)**：
- FCN-8s、FCN-16s、FCN-32s
- 融合不同层级的特征，恢复空间细节
- 深层特征提供语义信息，浅层特征提供空间细节

#### 2. U-Net

**编码器-解码器结构**：
- 对称的U型架构
- 编码器(Encoder)：逐步下采样提取特征
- 解码器(Decoder)：逐步上采样恢复分辨率

**跳跃连接**：
- 将编码器每层特征直接拼接(Concatenate)到解码器对应层
- 帮助恢复空间信息，减少上采样过程中的信息损失

**应用领域**：
- 最初为医学图像分割设计
- 在小样本数据集上表现优异
- 广泛应用于各类分割任务

#### 3. SegNet

**架构特点**：
- 使用编码器-解码器结构
- 编码器：VGG16的前13层卷积层
- 解码器：对应的上采样层

**池化索引(Pooling Indices)**：
- 在上采样时复用编码器的最大池化索引
- 减少参数量，保留空间信息
- 比学习反卷积参数更高效

#### 4. DeepLab系列 (v1/v2/v3/v3+)

**空洞卷积(Atrous/Dilated Convolution)**：
- 在不增加参数的情况下扩大感受野
- 避免池化导致的分辨率损失
- 保持特征图尺寸的同时捕获多尺度上下文

**空洞空间金字塔池化(ATROUS Spatial Pyramid Pooling, ASPP)**：
- 使用多个不同扩张率的空洞卷积并行处理
- 捕获多尺度上下文信息
- 在DeepLab v2中引入

**全连接CRF后处理**：
- DeepLab v1/v2使用
- 细化分割边界，增强空间一致性

**改进的编码器-解码器结构(DeepLab v3+)**：
- 结合ASPP和编码器-解码器架构
- 使用深度可分离卷积减少计算量
- 在边界处理上表现更好

#### 5. PSPNet (Pyramid Scene Parsing Network)

**金字塔池化模块(Pyramid Pooling Module)**：
- 使用不同尺度的平均池化(1×1, 2×2, 3×3, 6×6)
- 捕获全局上下文信息
- 聚合多尺度特征

**全局上下文信息**：
- 解决场景理解中的上下文关系问题
- 对复杂场景分割效果显著

#### 6. 其他重要架构

**ENet(Efficient Neural Network)**：
- 轻量级实时分割网络
- 早期下采样，使用瓶颈模块减少计算

**ICNet(Image Cascade Network)**：
- 多分辨率级联网络
- 平衡速度和精度

**RefineNet**：
- 多路径细化网络
- 显式利用所有可用信息，精细融合多尺度特征

**DeconvNet**：
- 深度反卷积网络
- 使用反卷积和unpooling进行上采样

## 编码器骨干网络

大多数分割网络使用在ImageNet上预训练的分类网络作为编码器：

**VGG系列**：VGG-16、VGG-19

**ResNet系列**：ResNet-50、ResNet-101、ResNet-152
- 残差连接解决深层网络退化问题
- 广泛用于分割任务

**MobileNet系列**：MobileNet v1/v2/v3
- 深度可分离卷积，适合移动端部署

**EfficientNet系列**：
- 通过神经架构搜索优化网络结构

**其他骨干网络**：Xception、DenseNet、HRNet等

## 训练策略

**迁移学习**：
- 使用ImageNet预训练权重初始化编码器
- 加速收敛，提高小数据集性能

**多尺度训练**：
- 输入不同尺寸的图像训练
- 增强模型对尺度变化的鲁棒性

**学习率调度**：
- Poly学习率衰减：常用于分割任务
- Step衰减、余弦退火等

**批归一化(Batch Normalization)**：
- 加速训练，提高稳定性
- 分割任务中通常使用较小batch size，可能导致BN不稳定
- 替代方案：Group Normalization、Layer Normalization

**深度监督(Deep Supervision)**：
- 在网络中间层添加辅助损失
- 帮助梯度传播，加速收敛

## 评估指标

**像素准确率(Pixel Accuracy)**：
- 正确分类的像素占比

**平均像素准确率(Mean Pixel Accuracy)**：
- 每类像素准确率的平均

**交并比(IoU, Intersection over Union)**：
- IoU = |A∩B| / |A∪B|
- 最常用的评估指标

**平均交并比(mIoU, Mean IoU)**：
- 所有类别IoU的平均值
- 语义分割的主要评估指标

**Dice系数/F1 Score**：
- 医学图像分割中常用

## 常用数据集

**PASCAL VOC 2012**：
- 20个前景类+1个背景类
- 经典分割基准数据集

**Cityscapes**：
- 城市街景语义分割
- 高分辨率图像(1024×2048)
- 30个类别，精细标注

**ADE20K**：
- 场景解析数据集
- 150个类别，涵盖多种场景

**COCO-Stuff**：
- 在COCO数据集基础上添加stuff类标注

**Mapillary Vistas**：
- 街景数据集，类别丰富

**CamVid**：
- 道路场景分割，常用于实时分割算法评估

## 传统方案的局限性

**计算效率**：
- 深层网络计算量大，难以实时部署
- 需要GPU加速

**小目标分割**：
- 下采样导致小目标信息丢失
- 边界模糊问题

**类别不平衡**：
- 背景类像素远多于前景类
- 需要特殊的损失函数和采样策略

**标注成本**：
- 像素级密集标注非常昂贵
- 限制了数据集规模

**泛化能力**：
- 领域迁移性能下降
- 需要目标域数据微调

## 向现代方案的演进

传统分割方案为现代Transformer-based和Foundation Model方法(如Segment Anything Model, SAM)奠定了基础。现代方案主要改进方向：

- **Transformer架构**：SegFormer、Segmenter、Mask2Former等
- **少样本/零样本分割**：基于CLIP等视觉-语言预训练模型
- **通用分割模型**：SAM等可处理任意对象的提示式分割
- **高效架构**：MobileNet-based、搜索架构等实时分割方案
- **弱监督/自监督学习**：降低标注成本

## 参考资料

- Long et al., "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- Badrinarayanan et al., "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation", TPAMI 2017
- Chen et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs", TPAMI 2018
- Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017
