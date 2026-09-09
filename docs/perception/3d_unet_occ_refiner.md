# 3D U-Net 占用精化器（OccVoxelUNetRefiner）

## 概述

`OccVoxelUNetRefiner` 是 Qwen-Drive-1.0 感知头中**专门为占用预测（occupancy）设计的 3D 解码器**，采用经典的 **U-Net 编码器-解码器架构**，在三维体素空间对粗糙的占用特征做精化。它是占用任务的**独占模块**——检测任务用 DETR 风格的 Transformer 解码器，地图任务用 2D 的 ResNet18 式解码器（`MapSegEncode`），而占用任务因为需要处理带高度维的 `200×200×16` 体素，因此使用了 3D 卷积构建的 U-Net。

模块位于 `src/qwen_drive_perception/occ_refiner.py`，输入是融合了 BEV 和 UVTR 信息的粗体素特征 `[B, 16, 16, 200, 200]`（通道维被拆到 Z 维），输出精化后的体素特征 `[B, 32, 16, 200, 200]`，再经过一个简单的 MLP 分类头得到 10 类占用 logits。

> 说明：本文基于当前仓库的**推理代码**撰写。仓库不含训练脚本，故本文只描述前向结构与张量流转，不涉及训练损失（占用的损失见占用相关文档）。

## 为什么占用任务需要 3D U-Net

占用预测的目标是为每个 `0.4m × 0.4m × 0.4m` 的体素分配一个语义类别（vehicle / pedestrian / driveable / empty 等），本质是**稠密的 3D 体素分割**任务。与 2D 图像分割类似，U-Net 的编码器-解码器 + 跳连接架构非常适合：

- **编码器下采样**：提取多尺度抽象特征，扩大感受野
- **解码器上采样**：逐步恢复空间分辨率
- **跳连接**：把编码器的细节特征传给解码器，保留边界、小物体等高频信息

相比直接用全卷积网络，U-Net 能在保持计算效率的同时，兼顾全局语义（编码器）和局部细节（跳连接）。

### 为什么是 3D 卷积而非 2D

占用网格是 `[X=200, Y=200, Z=16]` 的三维立方体，Z 维（高度）带有关键的垂直结构信息：

- 地面（Z=0–1 层）是 driveable
- 车辆（Z=2–5 层）是 vehicle
- 高层（Z>10）大多是 empty 或 background

如果用 2D 卷积在 XY 平面逐层独立处理，会丢失跨高度的上下文（比如"上面是空、下面是车"这种垂直相关性）。**3D 卷积**的核在 `(X, Y, Z)` 三个维度都做滑窗，能同时捕获水平和垂直方向的特征关联，更适合 3D 占用任务。

## 核心架构

`OccVoxelUNetRefiner` 是标准的 3D U-Net：

```
输入 [B, inC, Z, Y, X]
  ↓  input_proj (Conv3d 3×3×3 + BN + ReLU)
skip0 [B, c0, 16, 200, 200]
  ↓  enc1 (ResidualStage3D, stride=(1,2,2))  ← 只在 XY 下采样，Z 保持
skip1 [B, c1, 16, 100, 100]
  ↓  enc2 (stride=(1,2,2))
skip2 [B, c2, 16, 50, 50]
  ↓  enc3 (stride=(1,2,2))
skip3 [B, c3, 16, 25, 25]
  ↓  bottleneck (ResidualStage3D)
[B, c3, 16, 25, 25]
  ↓  upsample (trilinear) + concat(skip3)
  ↓  dec2 (ResidualStage3D)
[B, c2, 16, 50, 50]
  ↓  upsample + concat(skip2)
  ↓  dec1
[B, c1, 16, 100, 100]
  ↓  upsample + concat(skip1)
  ↓  dec0
[B, c0, 16, 200, 200]
  ↓  out_block (ResidualStage3D, 2 blocks)
  ↓  out_proj (Conv3d + BN + ReLU)
输出 [B, outC, 16, 200, 200]
```

### 关键设计特点

1. **下采样只在 XY 平面**  
   所有编码器层的 stride 都是 `(1, 2, 2)`，即 Z 方向 stride=1（不下采样），只在 X、Y 方向减半分辨率。这样保留了 16 层的垂直分辨率，避免丢失高度信息。如果 Z 也下采样到 2–4 层，就无法区分"地面、车身、车顶"这类垂直结构。

2. **三线性插值上采样**  
   解码器用 `F.interpolate(..., mode='trilinear')` 把特征图从 `(16, 25, 25)` 上采样回 `(16, 50, 50)`，并用 `align_corners=False` 避免边界对齐歧义。

3. **跳连接的 concat 策略**  
   解码器每层先上采样，再与对应编码器层的输出 concat（通道维拼接），送入残差块融合。这把编码器的高分辨率细节带回解码器。

4. **ResidualStage3D 作为基础块**  
   每个 stage 由多个 `BasicBlock3D`（3D 版残差块，类似 ResNet 的 BasicBlock）堆叠。`BasicBlock3D` 结构：
   ```
   x → Conv3d(3×3×3) → BN → ReLU → Conv3d(3×3×3) → BN
        ↓ (residual)
   x ────────────────────────────────────→ + → ReLU → output
   ```
   残差连接让深层网络更容易训练，避免梯度消失。

## 模块接口

### 初始化参数

```python
OccVoxelUNetRefiner(
    in_channels: int,   # 输入通道数（项目中是 middle_dims = 256 // 16 = 16）
    out_channels: int,  # 输出通道数（项目中是 occ_dim = 32）
    base_channels: int = 64,  # U-Net 的基础通道数（c0）
    num_blocks: tuple = (2, 2, 2, 2),  # 各 stage 的残差块数量
)
```

### 前向传播

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: [B, in_channels, Z, Y, X]，占用体素特征
           项目中是 [B, 16, 16, 200, 200]
    
    Returns:
        [B, out_channels, Z, Y, X]，精化后的体素特征
        项目中是 [B, 32, 16, 200, 200]
    """
```

## 在本项目中的使用

### 数据流位置

3D U-Net 位于占用预测的末端，接收融合了多种信息的粗体素特征：

```
BEV Transformer 输出 bev_embed [B, 200, 200, 256]
  ↓  permute + reshape: 把 256 通道拆成 16 层 × 16 通道
bev_3d [B, 16, 16, 200, 200]
  ↓  (可选) 与 UVTR 的 3D 体素融合
  ↓  _adapt_bev_for_occ: 裁剪到目标占用范围
fused [B, 16, 16, 200, 200]
  ↓  OccVoxelUNetRefiner (3D U-Net)
occ_feat [B, 32, 16, 200, 200]
  ↓  permute: [B, 32, 16, 200, 200] → [B, 200, 200, 16, 32]
  ↓  occ_pred_head (MLP: Linear(32→64)→Softplus→Linear(64→10))
occ_pred [B, 200, 200, 16, 10]  # 每个体素的 10 类 logits
```

关键步骤解释：

1. **通道到深度的转换**  
   `perception_transformer.py` 中，`bev_embed` 的 256 通道被 reshape 成 `(16, 16, 200, 200)`——相当于把通道维拆成 16 层，每层 16 通道。这个 16 对应 `occ_pillar_h`，即占用网格的高度层数。计算公式：
   ```python
   middle_dims = embed_dims // occ_pillar_h  # 256 // 16 = 16
   ```

2. **UVTR 融合（可选）**  
   如果启用了 UVTR，它的 3D 体素特征（保留了深度估计信息）会先经过 `uvtr_occ_proj` 投影到 `middle_dims=16` 通道，再与 BEV 来源的体素 concat 后通过 `uvtr_occ_fuse` 融合：
   ```python
   fused = uvtr_occ_fuse(torch.cat([bev_3d, uvtr_proj], dim=1))  # [B,32,16,200,200]→[B,16,16,200,200]
   ```

3. **坐标裁剪与对齐**  
   不同数据集的占用范围不同（nuScenes ±40m、nuPlan ±50m），`_adapt_bev_for_occ` 用可微的三线性采样把 BEV 特征裁剪/对齐到目标范围的体素网格上，再送进 U-Net。

### 配置参数

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `embed_dims` | 256 | BEV Transformer 输出通道数 |
| `occ_pillar_h` | 16 | 占用网格的高度层数（Z 维） |
| `middle_dims` | 16 | `embed_dims // occ_pillar_h`，U-Net 输入通道 |
| `occ_dim` | 32 | U-Net 输出通道数 |
| `base_channels` | 64 | U-Net 内部的基础通道数（c0） |
| `num_blocks` | `(2, 2, 2, 2)` | 编码器各 stage 的残差块数 |

通道数演化：`16 (input) → 64 → 128 → 256 → 384 (bottleneck) → ... → 64 → 32 (output)`

### 与其他任务解码器的对比

| 任务 | 解码器 | 维度 | 输入来源 | 特点 |
| --- | --- | --- | --- | --- |
| **检测** | DetectionTransformerDecoder | 1D query | bev_embed | DETR 式 query-to-BEV 交叉注意力 |
| **地图** | MapSegEncode | 2D | bev_embed 裁剪后的 BEV | ResNet18 式 2D U-Net（只在 XY） |
| **占用** | OccVoxelUNetRefiner | 3D | bev_embed reshape 成 3D + UVTR | 3D U-Net（XY 下采样、Z 保持） |

可以看到，三个任务的解码器各有专门设计：检测用稀疏 query、地图用 2D 卷积、占用用 3D 卷积，充分利用各自任务的结构特性。

## 实现细节

### BasicBlock3D（残差块）

```python
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1,1)):
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果尺寸或通道变化，需要下采样快捷连接
        if stride != (1,1,1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差连接
        return self.relu(out)
```

### ResidualStage3D（多个残差块堆叠）

```python
class ResidualStage3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, stride=(1,1,1)):
        blocks = [BasicBlock3D(in_channels, out_channels, stride)]  # 第一个块可能有下采样
        blocks += [BasicBlock3D(out_channels, out_channels) for _ in range(num_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)
```

### 上采样实现

```python
def _upsample_to(self, x, target):
    """三线性插值上采样，匹配 target 的空间尺寸"""
    return F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)

# 解码器中的使用
x = self._upsample_to(x, skip2)  # 从 (16,25,25) → (16,50,50)
x = self.dec2(torch.cat([x, skip2], dim=1))  # concat 跳连接后过残差块
```

## 工程与性能考虑

- **内存占用**：3D 卷积比 2D 卷积内存消耗大（多一个维度），但由于 Z 维只有 16 层且不下采样，实际占用可控。瓶颈层的 `(16, 25, 25)` 远小于 `(16, 200, 200)`。
- **计算量**：编码器的下采样策略（XY 减半、Z 保持）在保留垂直分辨率的同时大幅降低了 XY 平面的计算量（200×200 → 100×100 → 50×50 → 25×25）。
- **BN vs GN**：本模块用的是 `BatchNorm3d`，而地图解码器 `MapSegEncode` 用的是 `GroupNorm`（代码注释说 BN 在训练时被原地替换成了 GN）。占用解码器保持 BN 可能是因为 3D 占用的 batch 内统计更稳定。
- **单帧推理**：项目只做单帧推理，没有时序聚合。如果要做多帧占用，可以在 U-Net 输入前沿时间维聚合多帧体素（类似 UVTR 的 sweep 维求和）。

## 与 2D U-Net（地图）的对比

项目里同时用了 3D U-Net（占用）和 2D U-Net 式的地图解码器，值得对比：

| 维度 | 占用（OccVoxelUNetRefiner） | 地图（MapSegEncode） |
| --- | --- | --- |
| **输入** | `[B, 16, 16, 200, 200]` 3D 体素 | `[B, 256, 200, 400]` 2D BEV |
| **卷积** | Conv3d（xyz 三维核） | Conv2d（xy 二维核） |
| **下采样** | stride=(1,2,2)，Z 不动 | stride=(2,2)，xy 都减半 |
| **归一化** | BatchNorm3d | GroupNorm（32 组） |
| **输出** | `[B, 32, 16, 200, 200]` | `[B, 6, 200, 400]` |
| **类别数** | 10 类（vehicle/pedestrian/driveable/empty...） | 6 类（road_line/crosswalk/driveable...） |

两者的 U-Net 思想相同（编码器-解码器-跳连接），但一个在 3D 空间、一个在 2D 平面，分别适配占用和地图任务的几何特性。

## 小结

`OccVoxelUNetRefiner` 是 Qwen-Drive-1.0 占用预测的核心解码器：

1. 采用经典 3D U-Net 架构，编码器在 XY 平面下采样（扩大感受野），Z 方向保持 16 层（保留垂直结构）；
2. 解码器通过三线性上采样 + 跳连接恢复空间分辨率并融合多尺度特征；
3. 输入是融合了 BEV Transformer 和 UVTR 的粗体素（16 通道），输出精化后的体素特征（32 通道），再经 MLP 得到 10 类占用 logits；
4. 是占用任务的独占模块，与检测的 DETR 解码器、地图的 2D U-Net 并列，共同构成三任务感知头。

它的设计体现了"任务适配解码器"的思想：3D 卷积天然适合处理带高度信息的体素分割，而 U-Net 的跳连接保留了边界与小物体细节，两者结合让占用预测兼顾全局语义和局部精度。

## 参考

- 代码：`src/qwen_drive_perception/occ_refiner.py`、`perception_transformer.py`
- 相关方法：U-Net (Ronneberger et al., 2015)、3D U-Net (Çiçek et al., 2016)、ResNet BasicBlock
- 项目文档：`docs/perception.md`、UVTR 文档、技术报告 arXiv:2609.00111 第 4–5 页
