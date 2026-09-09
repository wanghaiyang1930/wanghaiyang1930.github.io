# UVTR 视图变换（深度引导的体素化）

## 概述

UVTR（Unified Voxel Transformer，统一体素变换）是一种将**多视角 2D 图像特征提升到 3D 体素空间**的视图变换方法。它属于 LSS（Lift-Splat-Shoot）系的"深度引导"路线：为图像上的每个像素预测一个沿相机射线方向的**深度分布**，再按这个分布把图像特征"散射"到三维体素网格中，从而得到一个显式的、带高度信息的 3D 特征体。

在 Qwen-Drive-1.0 的感知头中，UVTR 由 `Uni3DVoxelPoolDepth`（`src/qwen_drive_perception/view_transform.py`）实现，与 BEVFormer 风格的 BEV Transformer 并行工作，主要为**占用预测（occupancy）**提供保留高度维的 3D 几何先验，同时把压扁后的 BEV 版本贡献给三任务共享的 BEV 表示。

> 说明：本文基于当前仓库的**推理代码**撰写。仓库不含训练脚本，故本文只描述前向结构与张量流转，不涉及训练损失（占用的损失见占用相关文档）。

## 为什么需要视图变换

自动驾驶感知需要在**统一的 3D / BEV 空间**做检测、占用、地图等任务，而相机给出的是 2D 透视图像。从 2D 到 3D 的关键难点是**深度歧义**：一个像素对应相机射线上的一条线，不知道物体在射线上的哪个位置。

解决这一问题主要有两条路线：

- **隐式采样（BEVFormer 类）**：先在 BEV / 3D 空间放置查询，把查询点投影回图像做可变形注意力采样。深度是被注意力"隐式"学出来的。
- **显式深度提升（LSS / UVTR 类）**：直接为每个像素预测深度分布，沿射线把特征散射到体素。深度是被"显式"建模的，得到的 3D 体保留了高度结构，天然适合占用这类需要垂直分辨率的任务。

Qwen-Drive 同时用了这两条路线：BEVFormer 负责共享 BEV 表示，UVTR 负责显式的 3D 体素几何，两者互补。

## 核心原理

UVTR 的前向可以拆成四步：**深度预测 → 视锥反投影 → 深度感知体素池化 → 3D 卷积精化**。

### 1. 深度分布预测（DepthNet）

对每一路相机的图像特征，`DepthNet` 预测每个特征格子沿射线的**离散深度分布**（softmax 概率），而不是单一深度值。用分布而非单值，能表达"这个像素大概率在 12m，也有一定概率在 15m"的不确定性，并让特征沿射线软性铺开。

`DepthNet` 结构（`view_transform.py`）：

```
reduce_conv (Conv3x3 + GroupNorm + ReLU)   # 降维到 mid_channels
  ↓
3 × BasicBlock (残差块, GroupNorm)
  ↓
ASPP (空洞空间金字塔池化, 空洞率 1/6/12/18 + 全局池化)   # 多尺度上下文
  ↓
Conv1x1 → depth_channels                    # 输出每格的深度 logits
```

其中 **ASPP**（空洞空间金字塔池化）用多个不同空洞率的卷积并行捕获多尺度上下文，帮助网络在不同距离上都能给出合理的深度分布。输出经过 `softmax(dim=1)` 得到深度概率。

深度 bin 的数量由视锥配置决定：深度范围 1.0m–60.0m、步长 0.5m，因此 **118 个深度 bin**（`depth_dim = (60.0 - 1.0) / 0.5`）。

### 2. 视锥网格与反投影（coord_preparing）

UVTR 先构造一个覆盖图像平面的**视锥网格（frustum）**：在 896×512 的图像平面上以 16×16 像素为一格，配合 118 个深度步，形成一个 `(W, H, D, 3)` 的网格，每个元素是 `(u, v, d)`——图像坐标加上假设深度。

关键实现细节（`view_transform.py` 的 `frustum` 属性）：视锥网格**刻意不注册为 buffer**，而是以普通属性存放并固定 fp32。这样 `model.to(bfloat16)` 不会把它转成 bf16，保证反投影的矩阵求逆在 fp32 下进行，避免数值精度问题。

反投影流程：

```
(u, v, d) 视锥点
  ↓  齐次化, 前两维乘以深度: (u·d, v·d, d, 1)
  ↓  inv(lidar2img): 图像 → lidar 坐标
  ↓  lidar2ego: lidar → ego 坐标
ego 坐标下的 3D 点
  ↓  (point - pc_range_min) / voxel_size
体素索引 (x, y, z)
```

最后用 `mask` 标记哪些体素索引落在有效网格范围内（`voxel_shape = [200, 200, 16]`），越界的丢弃。参考点最终落在 **ego 坐标系**下，与 BEV 分支保持一致。

### 3. 深度感知的体素池化（feat_sampling）

这一步把图像特征按深度概率散射进体素，是 UVTR 的核心。它由自定义 CUDA 核 `voxel_pool_depth` 完成（`src/qwen_drive_perception/ops/`）：

- 输入：图像特征 `img_feats`、深度概率 `img_depth`、体素坐标 `voxel_coords`、有效掩码 `mask`。
- 对每个"像素 × 深度 bin"，把 `特征 × 该深度的概率` 累加到它反投影命中的体素里。
- 核把**深度加权**和**特征散射**融合在一次遍历中完成（fuse），避免生成庞大的中间张量。

论文对该核的描述：*"depth-aware voxel pooling of the view transform, fusing the per-pixel depth distribution and the feature scatter into a single pass."*

输出体素体形状为 `[B, C, Z, Y, X]`（经 permute 后），其中 `C = embed_dim = 256`，`Z=16, Y=200, X=200`。

### 4. 3D 卷积精化（feat_encoding）

多相机散射到同一体素网格后，先对（单帧场景下只有一个的）sweep 维求和，再过 **3 层 3D 卷积**（Conv3d + BatchNorm3d + ReLU）做局部精化，输出最终的体素特征体。

> 单帧推理下 sweep 维退化为 1，求和只是保持接口一致。

## 关键配置参数

以下常量来自 `src/qwen_drive_perception/configuration_perception.py`：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `FRUSTUM_RANGE` | `(0, 0, 1.0, 896, 512, 60.0)` | 视锥范围：图像 896×512，深度 1.0–60.0m |
| `FRUSTUM_SIZE` | `(16.0, 16.0, 0.5)` | 视锥步长：16×16 像素格，0.5m 深度步 |
| 深度 bin 数 | 118 | `(60.0 - 1.0) / 0.5` |
| `voxel_shape` | `[200, 200, 16]` | 体素网格 (X, Y, Z) |
| `EMBED_DIM` | 256 | 体素特征通道数 |
| `VIT_DIM` | 1024 | ViT patch 特征维度（输入） |
| `IMAGE_SIZE` | `(896, 512)` | 训练时固定的输入分辨率 |

`DepthNet` 的 `aspp_mid_channels=96`（见 `modeling_perception.py`）。

## 在本项目中的使用方式

### 数据流位置

UVTR 位于 `BEVFormerModelV2`（`modeling_perception.py`）中，输入是 VLM **视觉编码器（ViT）的 patch 特征**，而不是 LLM 的 image token：

```
多视角图像
  ↓ Qwen3.5 ViT
img_vit_feats [N_cam, H, W, 1024]
  ↓ vit_neck (SimpleFPN, scale=1.0)
vit_mlvl_feats [N_cam, 256, H, W]
  ↓ DepthNet → softmax
vit_depths [N_cam, 118, H, W]
  ↓ Uni3DVoxelPoolDepth.forward(vit_mlvl_feats, vit_depths, img_metas)
vit_voxel_space [B, 256, 16, 200, 200]
```

即 ViT 特征经 `SimpleFPN` 适配、`DepthNet` 出深度分布后，一起送进 `view_trans`（`Uni3DVoxelPoolDepth`）得到 3D 体素体。

### 一体两用：occ 专属 + BEV 共享

UVTR 产出的体素被用了两次（`modeling_perception.py`）：

```python
vit_voxel_space = self.view_trans(...)          # [B, 256, 16, 200, 200]
uvtr_occ_space = vit_voxel_space                # 路径A: 保留 Z 的完整 3D 体素
uvtr_bev_space = self._uvtr_voxel_to_bev_tokens(vit_voxel_space)  # 路径B: 压扁成 BEV
```

- **路径 A（occ 专属）**：完整 3D 体素 `uvtr_occ_feat` 只传给占用分支，在 `PerceptionTransformer` 里与裁剪后的 BEV 体素融合（`_fuse_uvtr_occ_feat`），再进 3D U-Net 解码。检测和地图分支不使用它。
- **路径 B（三任务共享）**：`_uvtr_voxel_to_bev_tokens` 用一个 `1×1` 卷积（`uvtr_query_proj`）把 `channels × depth` 压成 BEV 通道，得到 `uvtr_bev_feat`；它在 `BEVFormerHead` 里被**加到 BEV 查询上**：

```python
bev_queries = self.bev_embedding.weight
if uvtr_bev_feat is not None:
    bev_queries = bev_queries + uvtr_bev_feat   # 增强共享 BEV 表示
```

由于 `bev_queries` 经 BEVFormer 编码后产出的 `bev_embed` 被检测、占用、地图三任务共享，所以 UVTR 的 BEV 分量间接惠及三个任务。

一句话概括分工：**UVTR 主要为 occupancy 提供保留高度维的显式 3D 几何，同时把压扁的 BEV 版本贡献给共享表示。**

### 与 BEVFormer 分支的关系

| 分支 | 输入特征 | 深度处理 | 输出 | 服务对象 |
| --- | --- | --- | --- | --- |
| BEVFormer Encoder | LLM image token 特征 | 隐式（可变形注意力） | `bev_embed` 2D BEV | 三任务共享 |
| UVTR | ViT patch 特征 | 显式（深度分布） | 3D 体素体 | occ 专属 + BEV 共享分量 |

两者从**不同的 VLM 特征分支**取特征（LLM vs ViT），用**不同的深度策略**（隐式 vs 显式），最后在 BEV / 体素层面融合。

## 精度与工程细节

- **fp32 视锥**：视锥网格固定 fp32、不随模型转 bf16，保证 `inv(lidar2img)` 的矩阵求逆稳定。
- **融合 CUDA 核**：`voxel_pool_depth` 把深度加权与特征散射合并为单次遍历，JIT 编译缓存于 `~/.cache/torch_extensions`，首次编译需要与 torch 匹配的 `nvcc`。
- **单层特征约束**：`feat_sampling` 断言只支持单尺度特征（`vit_neck` 的 `scale_factors=(1.0,)`），与训练配置一致。
- **固定相机配置**：模型在固定的 896×512 分辨率与固定相机布局下训练，更换分辨率或相机数会偏离释放权重的适用范围。

## 小结

UVTR 是 Qwen-Drive-1.0 感知头里"显式深度提升"的视图变换模块：

1. `DepthNet`（含 ASPP）为每个图像特征格预测 118 个深度 bin 的分布；
2. 构造 896×512 图像平面的视锥网格，反投影到 ego 坐标系得到体素索引；
3. 融合 CUDA 核按深度概率把 ViT 特征散射进 `200×200×16` 体素，再过 3 层 3D 卷积精化；
4. 输出的 3D 体素一路作为 occupancy 的几何先验，一路压扁成 BEV 增强三任务共享表示。

它与 BEVFormer 的隐式采样互补，共同支撑检测、占用、地图三个感知任务。

## 参考

- 代码：`src/qwen_drive_perception/view_transform.py`、`modeling_perception.py`、`perception_transformer.py`、`ops/`
- 相关方法：LSS（Lift-Splat-Shoot）、BEVDepth、UVTR、BEVFormer
- 项目文档：`docs/perception.md`、技术报告 arXiv:2609.00111 第 4–5 页（BEV Perception Head）
