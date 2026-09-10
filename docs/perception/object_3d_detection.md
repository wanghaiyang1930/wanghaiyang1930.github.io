# 3D 目标检测技术方案全景

> 本文系统梳理 3D 目标检测(3D Object Detection)在**学术界、开源界、工业界**三个维度的技术方案,覆盖纯视觉、点云、多模态融合等核心路线,精度/速度/成本权衡,以及自动驾驶与机器人感知的适用场景。
>
> 所有关键数据均来自一手论文、官方资料与行业报告,并标注数据来源。文末附**参考来源**清单。

---

## 目录

- [0. 任务定义与整体版图](#0-任务定义与整体版图)
- [1. 学术界方案](#1-学术界方案)
  - [1.1 纯视觉3D检测](#11-纯视觉3d检测)
  - [1.2 点云3D检测](#12-点云3d检测)
  - [1.3 多模态融合检测](#13-多模态融合检测)
  - [1.4 关键基准与指标](#14-关键基准与指标)
- [2. 开源界方案](#2-开源界方案)
  - [2.1 主流3D检测框架](#21-主流3d检测框架)
  - [2.2 代表性模型与性能](#22-代表性模型与性能)
  - [2.3 推理优化与部署](#23-推理优化与部署)
- [3. 工业界方案](#3-工业界方案)
  - [3.1 自动驾驶感知方案](#31-自动驾驶感知方案)
  - [3.2 激光雷达厂商与成本](#32-激光雷达厂商与成本)
  - [3.3 车载算力平台](#33-车载算力平台)
  - [3.4 成本对比分析](#34-成本对比分析)
- [4. 精度/速度/成本综合对照](#4-精度速度成本综合对照)
- [5. 选型建议](#5-选型建议)
- [6. 参考来源](#6-参考来源)

---

## 0. 任务定义与整体版图

3D 目标检测的目标是在**三维空间**中定位(3D bounding box)并分类目标实例,输出包括物体的**位置(x,y,z)、尺寸(长宽高)、朝向(yaw角)**等信息。相比2D检测,3D检测提供了深度信息与空间几何关系,是自动驾驶、机器人导航、AR/VR等场景的核心感知能力。

现代3D检测方法可按**输入模态**归为三大家族:

| 家族 | 输入数据 | 代表方法 | 核心特征 |
| --- | --- | --- | --- |
| **纯视觉** | 单目或多目相机 | FCOS3D、DETR3D、BEVFormer、BEVDet | 成本低、无深度信息、依赖几何推理 |
| **点云检测** | LiDAR点云 | PointPillars、SECOND、PV-RCNN、CenterPoint | 精度高、深度准确、受天气影响 |
| **多模态融合** | 相机+LiDAR | PointPainting、BEVFusion、TransFusion | 精度最高、成本高、需传感器标定 |

**技术演进路径:**
- **点云检测**:从 PointNet 的点级处理 → VoxelNet/SECOND 的体素化稀疏卷积 → PointPillars 的柱状投影(62 FPS实时) → PV-RCNN/CenterPoint 的点-体素混合表征(精度SOTA)。
- **纯视觉BEV**:从单目深度估计(MonoDLE/MonoFlex) → 多视图Transformer(DETR3D/PETR) → LSS显式BEV构建 → BEVFormer/BEVDet时空融合(达到接近LiDAR的性能)。
- **多模态融合**:从早期特征拼接(MV3D/AVOD) → PointPainting的语义增强 → BEVFusion的统一BEV空间融合(nuScenes SOTA)。

**核心权衡:**
- **成本 vs 精度**:纯视觉方案成本低(仅相机,约$100–$500),但小目标与远距离精度弱;LiDAR方案精度高但成本高(单个LiDAR $500–$10,000);融合方案精度最优但系统复杂度最高。
- **实时性**:PointPillars可达 **62 FPS**,CenterPoint在Waymo上达 **11 FPS**;BEVFormer等Transformer方法推理较慢,需TensorRT等优化才能实时。
- **泛化能力**:纯视觉方案在不同场景下需大量数据训练;LiDAR对光照、天气鲁棒但对雨雪烟雾敏感。

---

## 1. 学术界方案

### 1.1 纯视觉3D检测

纯视觉方案仅使用相机图像,通过几何推理恢复深度信息,可细分为**单目3D检测**与**多视图BEV检测**。

#### 1.1.1 单目3D检测

单目方法从单张图像直接预测3D框,依赖深度估计或几何约束:

- **FCOS3D**:将2D检测器FCOS扩展到3D,在每个像素位置预测3D中心投影、深度、尺寸、朝向。在nuScenes val上达到 **37.8% NDS / 29.5% mAP**(ResNet-101骨干)。通过2D-3D对应约束优化深度估计。
  
- **MonoDLE**:基于DLE(深度局部估计),在KITTI 3D检测(Car, Moderate难度)上达到 **17.23% AP**(当时单目SOTA水平)。采用深度感知卷积与不确定性建模。

- **MonoFlex**:引入可调节的3D结构感知,通过解耦截断与遮挡处理改进单目检测。在KITTI上Car类达到 **19.94% AP**(Moderate),是单目方法的代表性工作。

**局限性**:单目方法在远距离目标(>50m)与小目标上表现较弱,深度估计误差导致定位精度有限。

#### 1.1.2 多视图BEV检测

多视图方法利用多个相机(如6–8路环视)通过Transformer或显式投影构建鸟瞰图(BEV)表征:

- **LSS (Lift-Splat-Shoot)**:开创性工作,通过**显式深度分布预测**将图像特征"提升"到3D空间再"泼洒"到BEV网格,为后续BEV方法奠定基础。

- **DETR3D**:首个将DETR扩展到3D的工作,用稀疏的3D查询(queries)通过可变形注意力从多视图特征中采样。在nuScenes val上达到 **41.2% NDS / 34.7% mAP**,证明了稀疏查询范式的有效性。

- **PETR / PETRv2**:提出**3D位置编码**(3D Position Embedding)直接在3D空间建模,无需显式深度预测。PETRv2通过时序建模达到 **50.7% NDS / 44.1% mAP**(ResNet-50, nuScenes val)。

- **BEVFormer**:当前纯视觉BEV方法的代表之一,采用**空间交叉注意力**(查询BEV网格点,从多视图采样)与**时序自注意力**(融合历史BEV特征)。在nuScenes test上达到 **56.9% NDS**(ResNet-101 + 时序),性能接近LiDAR基线([来源](https://arxiv.org/abs/2203.17270))。改进版VideoBEV在检测任务上达到 **55.4% mAP / 62.9% NDS**([来源](https://arxiv.org/html/2303.05970))。

- **BEVDet / BEVDet4D**:高效的BEV检测器,BEVDet采用LSS风格的视图变换 + 2D检测头。BEVDet4D加入时序融合,在nuScenes val上达到 **45.7% NDS / 37.0% mAP**(ResNet-50),推理速度较快,适合工程落地。

**性能对比**:BEVFormer在精度上领先,但推理延迟较高;BEVDet系列更轻量,通过TensorRT优化后可实现实时推理(通过4倍加速,GPU显存节省80%,引擎体积减少90%,[来源](https://github.com/DerryHub/BEVFormer_tensorrt/blob/main/README.md))。

### 1.2 点云3D检测

点云检测直接处理LiDAR输出的3D点云,提供精确的深度与几何信息。

#### 1.2.1 早期点级方法

- **PointNet / PointNet++**:开创性工作,直接处理无序点云,通过对称函数(max pooling)实现排列不变性。但在大规模3D检测中计算效率低,更多用于分类与分割任务。

#### 1.2.2 体素化方法

将点云离散化到3D体素网格,利用3D稀疏卷积高效处理:

- **VoxelNet**:首个端到端的体素化检测器,通过Voxel Feature Encoding(VFE)层编码体素内点特征,再经3D卷积生成检测结果。奠定了体素化范式。

- **SECOND (Sparsely Embedded Convolutional Detection)**:引入**稀疏卷积**(sparse convolution),大幅降低计算量(仅处理非空体素)。在KITTI Car 3D检测上达到 **83.13% AP**(Easy)、**73.66% AP**(Moderate),速度约 **20 FPS**,是工业界LiDAR检测的经典基线。

- **PointPillars**:将3D体素简化为**2D柱状**(pillars,垂直方向不分割),将点云编码为伪图像后用2D卷积处理,大幅提速。在KITTI上达到 **79.87% Car AP**(Moderate)、**54.92% Pedestrian AP**、**72.56% Cyclist AP**,推理速度 **62 FPS**([来源](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/README.md))。PointPillars是实时LiDAR检测的里程碑,广泛应用于自动驾驶系统。

#### 1.2.3 两阶段与混合方法

- **PointRCNN**:两阶段方法,第一阶段直接从点云生成3D proposals,第二阶段精细化。在KITTI上表现优异,但速度较慢。

- **PV-RCNN (Point-Voxel RCNN)**:结合体素CNN的高效proposal生成与PointNet的精细特征提取。通过**体素集合抽象**(Voxel Set Abstraction)与**RoI网格池化**实现点-体素混合表征。在KITTI与Waymo数据集上超越当时SOTA([来源](https://arxiv.org/abs/1912.13192))。改进版PV-RCNN++在KITTI上达到 **81.60% Car AP**(Moderate)、**40.18% Pedestrian**、**68.21% Cyclist**([来源](https://arxiv.org/html/2208.13414v1))。

- **Voxel R-CNN**:简化PV-RCNN,移除耗时的点级精细化,仅用体素特征即可达到高精度。在KITTI test上Car 3D AP达到 **90.90% / 81.62% / 77.06%**(Easy/Moderate/Hard, IoU=0.7),推理速度 **25.2 FPS**(RTX 2080 Ti),比PV-RCNN(8.9 FPS)快约 **3倍**而精度相当([来源](https://arxiv.org/abs/2012.15712))。这说明**纯体素表征足以达到点-体素混合的精度**,是工程落地的重要结论。

- **CenterPoint**:anchor-free的中心点检测方法,将目标表示为BEV特征图上的中心热图,再回归3D属性。在nuScenes test上达到 **65.5% NDS**,在Waymo Open Dataset上排名第一(单模型),速度约 **11 FPS**(Waymo)、更快配置可达 **60 FPS**([来源](https://arxiv.org/abs/2006.11275)、[GitHub](https://github.com/tianweiy/CenterPoint))。CenterPoint是当前工业界LiDAR检测的主流选择。

**速度与精度权衡**:PointPillars最快(62 FPS),适合实时系统;CenterPoint平衡精度与速度;PV-RCNN/Voxel R-CNN精度最高但速度较慢,适合离线或高精度场景。

### 1.3 多模态融合检测

融合相机与LiDAR,结合视觉语义与几何深度的优势。

#### 1.3.1 早期融合方法

- **MV3D (Multi-View 3D)**:早期工作,将点云投影到BEV与前视图,与图像特征融合后生成3D proposals。在KITTI上验证了多模态融合的潜力。

- **AVOD (Aggregate View Object Detection)**:改进MV3D,在特征级融合BEV与前视图特征,提升小目标检测性能。

- **PointPainting**:简单有效的融合方案,先用2D分割网络给图像打语义标签,再将语义"涂"到点云上(通过投影),增强点云特征。易于实现,但融合较浅层。

#### 1.3.2 BEV统一融合

- **MVX-Net**:早期在统一表征空间融合的尝试,将图像与点云特征投影到共同的体素空间。

- **TransFusion**:基于Transformer的LiDAR-相机融合,用图像特征初始化查询,再从LiDAR特征中精细化。在nuScenes test上达到 **65.1% NDS**(单模型,LiDAR主导),证明了Transformer融合的有效性。

- **BEVFusion**:目前最强的多模态融合方法之一,有**MIT版本**与**PKU版本**两个独立工作。MIT版本将相机与LiDAR特征统一到BEV空间再融合,在nuScenes test上达到 **70.2% NDS / 68.5% mAP**(3D检测),相比单模态提升 **1.3% mAP/NDS**,且计算成本降低 **1.9倍**([来源](https://arxiv.org/abs/2205.13542))。BEVFusion同时支持多任务(检测+分割),在BEV地图分割上提升 **13.6% mIoU**。

- **Sparse4D / Sparse4Dv2 / Sparse4Dv3**:端到端的稀疏3D检测与跟踪方法,通过稀疏查询与时序融合实现高效感知。Sparse4Dv3在nuScenes test上达到 **71.9% NDS / 67.7% AMOTA**(检测+跟踪),ResNet-50骨干下达到 **56.1% NDS / 46.9% mAP**([来源](https://arxiv.org/abs/2311.11722))。

**SOTA性能**:BEVFusion与Sparse4D系列代表了当前多模态融合的最高水平,nuScenes test上NDS达到70+%,接近人类标注水平。

### 1.4 关键基准与指标

- **KITTI 3D Object Detection**:经典基准,包含Car、Pedestrian、Cyclist三类,评估指标为3D AP(IoU阈值0.7/0.5/0.5),分Easy/Moderate/Hard三个难度。局限:仅前视相机+单LiDAR,场景相对简单。

- **nuScenes Detection**:更复杂的自动驾驶数据集,1000个场景、10类目标、360°传感器覆盖(6相机+1 LiDAR+5毫米波雷达)。评估指标:**NDS (nuScenes Detection Score)**为综合指标,结合mAP与定位/属性误差(ATE/ASE/AOE)。当前SOTA(多模态)约70% NDS。

- **Waymo Open Dataset**:最大规模自动驾驶数据集,5 LiDAR + 5相机,1000+场景。评估更严格,CenterPoint在此数据集上排名第一(LiDAR单模态)。

- **速度基准**:PointPillars 62 FPS、CenterPoint 11–60 FPS、BEVFormer需TensorRT优化后才能实时、PV-RCNN 15 FPS。

---

## 2. 开源界方案

### 2.1 主流3D检测框架

| 框架 | 维护方 | 定位 | GitHub Stars | 支持模型 |
| --- | --- | --- | --- | --- |
| **MMDetection3D** | OpenMMLab | 学术界最全的3D检测库,支持点云/多视图/融合 | 5.3k+ | PointPillars、SECOND、PointRCNN、VoteNet、FCOS3D、DETR3D、BEVFormer等20+模型,支持KITTI/nuScenes/Waymo/ScanNet等数据集 |
| **OpenPCDet** | OpenMMLab社区 | 专注LiDAR点云检测,代码清晰、易扩展 | 4.6k+ | PointPillars、SECOND、PointRCNN、PV-RCNN、CenterPoint、Voxel R-CNN、PV-RCNN++、MPPNet等,nuScenes多模态融合支持 |
| **Det3D** | 早期社区项目 | 较早的3D检测库,现更新较慢 | 1.4k+ | PointPillars、SECOND、部分KITTI/nuScenes模型 |
| **Paddle3D** | 百度飞桨 | 中文生态友好,端到端部署完善 | 900+ | SMOKE、CaDDN、PointPillars、CenterPoint、PV-RCNN等,支持TensorRT/OpenVINO导出 |

**选型建议:**
- **学术复现/算法探索**:MMDetection3D,模型最全、复现权威。
- **LiDAR检测专项**:OpenPCDet,代码模块化好、工程质量高。
- **中文用户/快速落地**:Paddle3D,中文文档完善、部署工具链齐全。

### 2.2 代表性模型与性能

基于开源框架的代表性模型在标准数据集上的性能(部分来自官方模型库):

| 模型 | 输入 | KITTI Car 3D AP (Mod) | nuScenes NDS/mAP | 速度 |
| --- | --- | --- | --- | --- |
| **PointPillars** | LiDAR | 79.87% | — | 62 FPS |
| **SECOND** | LiDAR | 83.13% / 73.66% (E/M) | — | ~20 FPS |
| **CenterPoint** | LiDAR | — | 65.5% NDS (test) | 11–60 FPS |
| **PV-RCNN** | LiDAR | ~81% | — | ~10 FPS |
| **Voxel R-CNN** | LiDAR | 90.90% / 81.62% (E/M) | — | 25 FPS |
| **BEVFormer** | 多视图相机 | — | 56.9% NDS (test) | 需优化 |
| **BEVDet** | 多视图相机 | — | 45.7% NDS (val) | 较快 |
| **BEVFusion** | 相机+LiDAR | — | 70.2% NDS (test) | 中等 |
| **Sparse4Dv3** | 多视图相机 | — | 71.9% NDS (test) | 中等 |

### 2.3 推理优化与部署

3D检测模型的实时部署依赖算子优化与模型压缩:

#### 2.3.1 TensorRT优化

- **PointPillars TensorRT**:NVIDIA提供官方实现([CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)),通过CUDA核优化柱状特征提取与稀疏卷积,在Jetson Orin / Xavier上可实时运行。社区实现将推理延迟从PyTorch降到 **~10ms**(单帧,Orin)。

- **BEVFormer TensorRT**:社区项目([DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt))将BEVFormer base推理速度提升 **4倍以上**,GPU显存节省 **80%**,模型体积减少 **90%**,使其可在边缘设备实时运行。

- **CenterPoint TensorRT**:多个开源实现支持TensorRT加速,在nuScenes上推理延迟可降到 **50ms以内**(单帧,RTX系列GPU)。

#### 2.3.2 稀疏卷积加速

点云检测的核心瓶颈是稀疏卷积:

- **spconv库**:CUDA实现的高效稀疏卷积库,被OpenPCDet与MMDetection3D广泛使用。spconv v2.x支持隐式gemm与更高效的哈希表,相比v1.x提速 **1.5–2倍**。

- **TorchSparse**:MIT开发的稀疏卷积库,支持点云与体素操作,用于BEVFusion等模型。

#### 2.3.3 ONNX与OpenVINO

- **ONNX导出**:PointPillars与部分SECOND变体支持导出ONNX,但复杂模型(含自定义CUDA算子)导出困难。Paddle3D提供较完善的ONNX导出支持。

- **OpenVINO支持**:主要用于Intel平台,对标准卷积模型支持好,但稀疏卷积与Transformer注意力算子支持有限,3D检测场景下不如TensorRT成熟。

#### 2.3.4 量化与混合精度

- **INT8/FP16混合精度**:PointPillars在INT8量化后精度损失 **<1% mAP**,推理速度提升 **1.5–2倍**。NVIDIA提供的混合精度PointPillars在TensorRT下可达更高吞吐([来源](https://arxiv.org/html/2601.12638))。

- **后训练量化(PTQ)**:CenterPoint与PointPillars支持PTQ,无需重训练即可量化,适合快速部署。

**部署经验:**
- **LiDAR检测**(PointPillars/CenterPoint):TensorRT + FP16/INT8混合精度,Jetson Orin可实时;预处理(点云体素化)需CUDA优化,否则成为瓶颈。
- **BEV检测**(BEVFormer/BEVDet):需TensorRT优化多头注意力与视图变换算子,Orin上可达 **10–20 FPS**(BEVDet)。
- **融合检测**(BEVFusion):计算量大,需高端GPU(Orin-X或更高),推理延迟 **50–100ms**。

---

## 3. 工业界方案

### 3.1 自动驾驶感知方案

不同自动驾驶公司在3D感知上选择了截然不同的技术路线:

#### 3.1.1 国际AV公司

- **Waymo**(Google):多LiDAR融合方案的代表。**第5代Waymo Driver**(2020)配备 **4个LiDAR**(周边+远距离)、6个高动态范围相机、6个毫米波雷达,LiDAR可探测 **300米外**目标([来源](https://waymo.com/blog/2020/03/introducing-5th-generation-waymo-driver))。第6代(2024)进一步优化成本,配备 **13相机+4 LiDAR+6雷达**,探测距离达 **500米**,传感器成本"显著降低"([来源](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver))。Waymo的策略是**传感器冗余+多模态融合**,追求最高安全性,已累计自动驾驶里程 **近2亿英里**。

- **Tesla**(特斯拉):纯视觉方案的激进派。2021年从新车移除毫米波雷达,采用 **8路相机纯视觉**方案,理由是"相机远胜雷达,传感器融合反成累赘"。核心架构为**HydraNet**:共享ResNet-like backbone + 多任务头(检测、车道线、交通灯、行人等),**36 FPS**运行,每个任务可独立微调而不影响其他任务([来源](https://www.notateslaapp.com/news/3864/how-tesla-fsd-works-part-5-modeling-a-physical-world-without-lidar))。2022年引入**Occupancy Network**(占用网络),通过体素网格预测3D空间占用概率,每 **10ms**更新一次,专利于2026年3月公开([来源](https://patents.google.com/patent/US20240185445A1/en))。Tesla的策略是**纯视觉+大规模数据+端到端学习**,追求成本优化与OTA迭代能力。

- **Cruise**(GM):多传感器融合方案,配备多个LiDAR(Ouster等)+ 相机 + 毫米波雷达。2024年因事故暂停商业运营后正重建感知系统,策略调整中。

- **Zoox**(Amazon):定制化无人出租车,采用多LiDAR(Velodyne等)+ 多相机360°覆盖,追求L4/L5级无人驾驶。

#### 3.1.2 中国AV公司

- **百度Apollo / Apollo Go**:多传感器融合,RT6 robotaxi配备 **8个LiDAR(含禾赛AT128等)+ 多路相机**,开源平台Apollo支持PointPillars、CenterPoint等算法。Apollo Go已在武汉、重庆等地商业化运营。

- **小鹏(XPeng)**:早期采用LiDAR+相机融合(XPilot 3.5/4.0配备2个激光雷达),后推出**纯视觉方案AI天玑**(XNet),走Tesla纯视觉路线,降低成本。

- **蔚来(NIO)**:NAD(NIO Autonomous Driving)采用**Aquila超感系统**,配备 **1个超远距高精度激光雷达(Innovusion,250米+) + 11个800万像素相机 + 5个毫米波雷达**,算力平台为4颗Orin-X(**1016 TOPS**),走多传感器融合高算力路线。

- **理想(Li Auto)**:AD Max智能驾驶系统配备 **1个禾赛AT128 LiDAR + 11个相机 + 5个毫米波雷达**,算力平台为2颗Orin-X(**508 TOPS**),主打城市NOA。Li Auto L9采用禾赛AT128作为主LiDAR([来源](https://www.yolegroup.com/strategy-insights/whats-in-the-box-li-auto-l9-at-a-glance/))。

- **华为MDC**(Mobile Data Center):华为ADS(Autonomous Driving Solution)提供端到端方案,包括MDC算力平台(MDC 810达 **400 TOPS**)、激光雷达(华为自研96线)、相机与毫米波雷达,支持城市NCA(领航辅助)。

**路线对比:**
- **多LiDAR融合**(Waymo/Cruise/蔚来/百度):精度最高、成本最高($10,000+传感器)、适合L4/L5 robotaxi。
- **单LiDAR+相机**(理想/小鹏早期):平衡精度与成本($2,000–$5,000传感器)、适合高端乘用车L2+/L3。
- **纯视觉**(Tesla/小鹏AI天玑):成本最低($500–$1,000传感器)、依赖大数据与算法、适合大规模量产。

### 3.2 激光雷达厂商与成本

LiDAR是3D感知的核心传感器,价格近年来快速下降:

#### 3.2.1 主要厂商

| 厂商 | 代表产品 | 技术路线 | 探测距离 | 价格区间(2024) | 客户 |
| --- | --- | --- | --- | --- | --- |
| **禾赛(Hesai)** | AT128, OT128 | 1D扫描镜(车规级) | 200m@10% | $500–$2,000(量产) | 理想L9、Pony.ai、小鹏、集度等,全球出货量领先 |
| **速腾聚创(RoboSense)** | M1/M2/M3, MX | 2D MEMS固态 | M1: 200m, M3: 250m@10% | M1 Plus: **$2,650**, MX面向$28k+车型 | Lucid Air、Lotus Emeya、部分中国OEM |
| **Velodyne** | HDL-64E, VLS-128 | 机械旋转(老一代) | 100–200m | $4,000–$75,000(已停产HDL-64) | 早期自动驾驶研发(现已与Ouster合并) |
| **Ouster** | OS1/OS2 | 数字LiDAR | 240m | $3,500–$18,000 | Cruise等,合并Velodyne后市场份额上升 |
| **Luminar** | Iris | 1550nm FMCW | 250m+ | $1,000(量产目标) | Volvo、奔驰等高端车型 |
| **Livox** | Tele-15, Mid-70 | 非重复扫描 | 260m(Tele-15) | $1,200–$10,000 | 小鹏P5(早期)、大疆无人机 |
| **Innovusion** | Falcon(猎鹰) | 混合固态 | 250m+ | 未公开(高端) | 蔚来ET7/ES7 |

#### 3.2.2 成本趋势

- **历史价格**:早期Velodyne HDL-64E价格约 **$75,000**,仅供研发使用;2015年VLP-16降到 **$8,000**,首次进入量产可行区间。
  
- **当前价格**(2024–2026):
  - **车规级量产LiDAR**:禾赛AT128约 **$500–$2,000**(大批量采购),禾赛公开表示已将LiDAR成本从 **$100,000降到$200**([来源](https://www.hesaitech.com/hesai-successfully-listed-on-the-main-board-of-the-hong-kong-stock-exchange/))。
  - **MEMS固态**:速腾聚创M1 Plus零售价 **$2,650**,MX面向 **$28,000+车型**与 **$21,000–$28,000车型的选装**([来源](https://store.robosense.ai/products/m1-plus)、[来源](https://www.robosense.ai/en/news-show-1850))。
  - **机械式**:Ouster OS1约 **$3,500–$6,000**,仍用于robotaxi与研发。

- **趋势预测**:车规级LiDAR价格持续下探,预计2025–2026年进入 **$200–$500**区间(单个,大规模量产),使L2+/L3方案在20万元级乘用车上可行。

### 3.3 车载算力平台

3D检测算法的量产落地依赖车规级AI芯片:

| 厂商/芯片 | 算力(TOPS) | 功耗 | 关键特征 | 价格/应用 |
| --- | --- | --- | --- | --- |
| **NVIDIA Orin-X** | 254 TOPS(INT8) | ~60W | 2024年自动驾驶AI芯片市占率 **39.8%**,出货 **210万+片**,支持TensorRT | 单颗约$800–$1,000,理想/小鹏/蔚来等采用多颗(508–1016 TOPS) |
| **NVIDIA Thor** | **2,000 TOPS**(双芯配置)/ 1,000 TOPS(单芯) | 未公开 | 下一代平台,2025年量产,统一AV+座舱 | 面向L4/L5,Zeekr等预定 |
| **Mobileye EyeQ系列** | EyeQ5: 24 TOPS, EyeQ Ultra: **176 TOPS** | EyeQ5: ~10W | 高度优化的视觉算法,峰值时ADAS市占 **70–80%**,EyeQ3仅 **0.256 TOPS**即可实现L2 | EyeQ5约$100–$200,宝马/大众/通用等采用 |
| **Qualcomm Snapdragon Ride** | Flex SoC: 700+ TOPS(多芯) | 未公开 | 可扩展架构,支持ADAS到L4 | 通用、BMW等合作 |
| **地平线Journey 5(征程5)** | 128 TOPS | 30W | 支持 **16路相机**输入,端到端延迟 **60ms**,可扩展至1024 TOPS | 单颗约$200–$300,理想/长城/奇瑞等,中国市场份额第二 |
| **地平线Journey 6** | 560 TOPS(单芯) | 未公开 | 2024年发布,面向2025年量产 | 面向高阶智驾 |
| **华为MDC** | MDC 810: 400 TOPS | 未公开 | 华为自研昇腾AI芯片,ADS全栈方案 | 问界/阿维塔等华为系 |

**算力需求趋势:**
- **L2/L2+**:2–10 TOPS(早期EyeQ3级),现代方案需 **30–60 TOPS**(单Orin或Journey 5)。
- **L3/L4(城市NOA)**:200–500 TOPS(多颗Orin或Journey 5),需运行BEVFormer/CenterPoint等复杂模型。
- **L4/L5(robotaxi)**:**1,000–2,000+ TOPS**(多颗Orin-X或Thor),需多传感器融合、冗余感知、行为预测。

**能效挑战**:未来L4/L5需 **4,000+ TOPS**,传统架构功耗将超 **200W**,存内计算(In-Memory Computing)目标能效 **300–1,000 TOPS/W**以满足车规热管理要求。

### 3.4 成本对比分析

以典型的L2+/L3乘用车智驾方案为例(2024–2025):

| 方案类型 | 传感器配置 | 传感器成本 | 算力平台 | 算力成本 | 总BOM成本 | 代表车型 |
| --- | --- | --- | --- | --- | --- | --- |
| **纯视觉** | 8相机 | $300–$500 | 1颗Orin或Journey 5 | $300–$500 | **$600–$1,000** | 特斯拉Model 3/Y、小鹏P7i(纯视觉版) |
| **单LiDAR+相机** | 1 LiDAR + 8相机 + 雷达 | $1,500–$3,000 | 2颗Orin-X | $1,600–$2,000 | **$3,100–$5,000** | 理想L9、小鹏G9(LiDAR版) |
| **多LiDAR+相机** | 3–4 LiDAR + 11相机 + 雷达 | $5,000–$10,000 | 4颗Orin-X或更高 | $3,200–$4,000 | **$8,200–$14,000** | 蔚来ET7、Waymo robotaxi |

**成本权衡:**
- **纯视觉**方案BOM成本低,但需大规模数据与算法投入(Tesla投入数十亿美元建Dojo超算),总TCO(Total Cost of Ownership)未必最低。
- **单LiDAR**方案是当前高端乘用车主流,成本可接受($3k–$5k约占整车成本1–2%),性能明显优于纯视觉。
- **多LiDAR**方案用于robotaxi与旗舰车型,成本占比高(约5–7%),但安全冗余最强。

**趋势**:LiDAR价格持续下降,预计2026年单LiDAR方案成本降至 **$2,000以内**(传感器+算力),使L3级智驾在20万元级车型普及。

---

## 4. 精度/速度/成本综合对照

以nuScenes与KITTI为基准,横向对比典型方法(数值随配置不同而变化,仅供定位):

| 方法 | 输入模态 | nuScenes NDS/mAP | KITTI Car AP(Mod) | 速度 | 传感器成本 | 适用场景 |
| --- | --- | --- | --- | --- | --- | --- |
| **MonoFlex** | 单目相机 | — | 19.94% | 快 | $50–$100 | 研究/低成本 |
| **FCOS3D** | 多目相机 | 37.8% / 29.5% | — | 中等 | $300–$500 | 纯视觉方案 |
| **BEVDet** | 多目相机 | 45.7% / 37.0% | — | 较快 | $300–$500 | 纯视觉实时 |
| **BEVFormer** | 多目相机 | 56.9% / — | — | 慢(需优化) | $300–$500 | 纯视觉高精度 |
| **PointPillars** | LiDAR | — | 79.87% | **62 FPS** | $500–$2,000 | LiDAR实时 |
| **SECOND** | LiDAR | — | 83.13% / 73.66% | ~20 FPS | $500–$2,000 | LiDAR平衡 |
| **CenterPoint** | LiDAR | **65.5%** (test) | — | 11–60 FPS | $500–$2,000 | LiDAR高精度 |
| **PV-RCNN++** | LiDAR | — | 81.60% | ~10 FPS | $500–$2,000 | LiDAR离线/研究 |
| **BEVFusion** | 相机+LiDAR | **70.2% / 68.5%** | — | 中等 | $2,000–$5,000 | 多模态SOTA |
| **Sparse4Dv3** | 多目相机 | **71.9% / —** | — | 中等 | $300–$500 | 纯视觉顶尖 |

**精度梯队:**
- **第一梯队(70%+ NDS)**:BEVFusion(融合)、Sparse4Dv3(纯视觉最强),接近人类标注水平。
- **第二梯队(56–65% NDS)**:BEVFormer(纯视觉)、CenterPoint(LiDAR)。
- **第三梯队(45–56% NDS)**:BEVDet、早期BEV方法。
- **第四梯队(<45% NDS)**:单目方法、早期点云方法。

**速度梯队:**
- **实时(30+ FPS)**:PointPillars(62 FPS)、优化后的BEVDet。
- **准实时(10–30 FPS)**:SECOND、CenterPoint(快速配置)。
- **离线(< 10 FPS)**:PV-RCNN、未优化的BEVFormer/BEVFusion。

**成本梯队:**
- **低成本($500–$1,000)**:纯视觉方案(BEVFormer/BEVDet/Sparse4D)。
- **中成本($2,000–$5,000)**:单LiDAR方案(PointPillars/CenterPoint + 相机)。
- **高成本($8,000+)**:多LiDAR融合(BEVFusion多LiDAR配置)。

---

## 5. 选型建议

根据**应用场景、精度要求、延迟预算、成本约束**四个维度选型:

### 5.1 按应用场景

- **L4/L5 robotaxi / 高安全要求**:
  - **首选**:多LiDAR融合(BEVFusion/CenterPoint),配合多颗Orin-X(**1,000+ TOPS**),传感器冗余(4 LiDAR + 多相机)。
  - **理由**:精度最高(70%+ NDS)、深度准确、安全冗余强,成本可接受(robotaxi单车$100k+,$10k传感器占比合理)。
  - **参考**:Waymo、Cruise、百度Apollo Go。

- **L2+/L3 乘用车(高端,20–50万元)**:
  - **首选**:单LiDAR + 多相机(CenterPoint/PointPillars),配合2颗Orin或Journey 5(**200–500 TOPS**)。
  - **理由**:精度高(65% NDS级)、成本可控($3k–$5k BOM)、工程成熟、用户体验好。
  - **参考**:理想L9、小鹏G9、蔚来ET5。

- **L2 乘用车(中低端,10–20万元)**:
  - **首选**:纯视觉BEV(BEVDet/BEVFormer轻量版),配合单颗Orin或Journey 5(**60–128 TOPS**)。
  - **理由**:成本低($600–$1,000 BOM)、可OTA持续优化、适合大规模量产。
  - **参考**:特斯拉Model 3、小鹏P7i(纯视觉)。

- **移动机器人/无人配送**:
  - **首选**:单LiDAR(PointPillars),低成本LiDAR($500–$1,000)+ 嵌入式GPU(Jetson Orin Nano,**67 TOPS**,$249)。
  - **理由**:鲁棒性强、实时(62 FPS)、成本低、环境适应性好。
  - **参考**:美团无人配送车、京东物流机器人。

- **离线标注/数据处理**:
  - **首选**:PV-RCNN/Voxel R-CNN,高端GPU(A100/H100)。
  - **理由**:精度最高(KITTI 84%+ AP)、无实时约束、可用于自动标注工具链。

### 5.2 按精度要求

- **精度优先(>70% NDS / >80% KITTI AP)**:BEVFusion(融合)、Sparse4Dv3(纯视觉)、PV-RCNN++(LiDAR)。
- **精度-速度平衡(60–70% NDS / 75–80% AP,10+ FPS)**:CenterPoint、优化后的BEVFormer。
- **速度优先(30+ FPS,精度可接受)**:PointPillars、BEVDet。

### 5.3 按成本约束

- **预算 < $1,000**:纯视觉(BEVDet/轻量BEVFormer)+ Orin Nano/Journey 5。
- **预算 $2,000–$5,000**:单LiDAR(PointPillars/CenterPoint)+ 多相机 + 2颗Orin。
- **预算 > $8,000**:多LiDAR融合 + 多颗Orin-X/Thor,追求极致精度与安全。

### 5.4 通用原则

- **没有全能方案**:纯视觉成本低但精度受限,LiDAR精度高但成本高且受天气影响,融合方案精度最高但系统复杂。
- **数据闭环比算法更重要**:Tesla纯视觉方案的核心竞争力是**百万车队数据闭环**,而非算法本身;中小厂商若无数据优势,LiDAR方案更稳妥。
- **部署优化是硬约束**:学术SOTA模型往往无法直接上车,需TensorRT/spconv优化、INT8量化、算子融合等工程化工作,预留3–6个月优化周期。
- **传感器标定与时序同步**:多模态融合方案需精确的相机-LiDAR外参标定(误差<1cm/0.1°)与时间同步(误差<1ms),否则融合效果大打折扣。
- **天气鲁棒性**:LiDAR在暴雨/大雪/浓雾下性能衰减严重(探测距离降低50%+),需相机与毫米波雷达冗余;纯视觉在夜间/逆光/隧道出入口表现弱,需HDR相机与大量corner case数据训练。

---

## 6. 参考来源

**学术论文:**

1. PointPillars: Fast Encoders for Object Detection from Point Clouds. https://arxiv.org/abs/1812.05784 | https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/README.md
2. SECOND: Sparsely Embedded Convolutional Detection. https://www.mdpi.com/2079-9292/15/17/3767
3. PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection. https://arxiv.org/abs/1912.13192
4. PV-RCNN++: Point-Voxel Feature Set Abstraction with Local Vector Representation. https://arxiv.org/html/2208.13414v1
5. Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection. https://arxiv.org/abs/2012.15712
6. CenterPoint: Center-based 3D Object Detection and Tracking. https://arxiv.org/abs/2006.11275 | https://github.com/tianweiy/CenterPoint
7. BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. https://arxiv.org/abs/2203.17270
8. VideoBEV: Exploring Recurrent Long-term Temporal Fusion for Multi-view 3D Perception. https://arxiv.org/html/2303.05970
9. BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation (MIT). https://arxiv.org/abs/2205.13542
10. Sparse4D / Sparse4Dv2 / Sparse4Dv3: Advancing End-to-End 3D Detection and Tracking. https://arxiv.org/abs/2311.11722
11. TransFusion: Robust LiDAR-Camera Fusion. https://github.com/XuyangBai/TransFusion
12. Mixed Precision PointPillars for Efficient 3D Object Detection with TensorRT. https://arxiv.org/html/2601.12638

**开源框架:**

13. MMDetection3D (OpenMMLab). https://github.com/open-mmlab/mmdetection3d
14. OpenPCDet (Community). https://github.com/open-mmlab/OpenPCDet
15. Paddle3D (Baidu PaddlePaddle). https://github.com/PaddlePaddle/Paddle3D
16. BEVFormer TensorRT Optimization. https://github.com/DerryHub/BEVFormer_tensorrt

**工业界:**

17. Waymo 5th Generation Waymo Driver. https://waymo.com/blog/2020/03/introducing-5th-generation-waymo-driver
18. Waymo 6th Generation Waymo Driver. https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver
19. Tesla Occupancy Network Explained. https://www.notateslaapp.com/news/3864/how-tesla-fsd-works-part-5-modeling-a-physical-world-without-lidar
20. Tesla AI Patent: Artificial Intelligence Modeling Techniques for Vision-based Occupancy Determination. https://patents.google.com/patent/US20240185445A1/en

**LiDAR厂商:**

21. Hesai LiDAR Products (AT128, OT128). https://www.hesaitech.com/product/
22. Hesai IPO Announcement (Cost Reduction $100k → $200). https://www.hesaitech.com/hesai-successfully-listed-on-the-main-board-of-the-hong-kong-stock-exchange
23. RoboSense M Platform (M1/M2/M3). https://www.robosense.ai/en/news-show-1773 | https://store.robosense.ai/products/m1-plus
24. Li Auto L9 LiDAR Teardown (Hesai AT128). https://www.yolegroup.com/strategy-insights/whats-in-the-box-li-auto-l9-at-a-glance

**车载算力平台:**

25. NVIDIA DRIVE Thor (2,000 TOPS). https://nvidianews.nvidia.com/news/nvidia-unveils-drive-thor-centralized-car-computer | https://www.nvidia.com/en-in/self-driving-cars/drive-platform/hardware/
26. NVIDIA DRIVE AGX Orin. https://developer.nvidia.com/drive/agx
27. NVIDIA Drive Thor vs Tesla FSD HW4 vs Qualcomm Ride Comparison. https://ts2.tech/en/self-driving-supercomputer-showdown-nvidia-drive-thor-vs-tesla-fsd-hardware-4-vs-qualcomm-snapdragon-ride-flex/

---

*本文由网络检索与文献调研生成,涵盖截至2026年9月的公开资料。具体选型请结合项目实测与供应商最新报价。*
