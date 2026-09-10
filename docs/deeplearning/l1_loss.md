# L1Loss 原理与应用

## 1. 什么是 L1 Loss

### 1.1 定义

**L1 Loss（也称 Mean Absolute Error, MAE）** 是回归任务中最基础的损失函数之一，衡量预测值与真实值之间的**绝对差**。

**数学定义**：

```
L1(y, ŷ) = |y - ŷ|
```

对于一个批次（N 个样本）：

```
MAE = (1/N) ∑ᵢ |yᵢ - ŷᵢ|
```

- `y`：真实值（ground truth）
- `ŷ`：预测值（prediction）
- `|·|`：绝对值

### 1.2 直观理解

L1 Loss 测量的是预测值偏离真实值的**平均绝对距离**，单位与原始数据一致。

```python
# 示例
真实值 y   = [3.0, 5.0, 2.0]
预测值 ŷ   = [2.5, 6.0, 2.0]
绝对误差   = [0.5, 1.0, 0.0]
L1 Loss    = (0.5 + 1.0 + 0.0) / 3 = 0.5
```

---

## 2. L1 vs L2 Loss

### 2.1 数学对比

| 特性 | L1 Loss (MAE) | L2 Loss (MSE) |
|------|---------------|---------------|
| **公式** | `\|y - ŷ\|` | `(y - ŷ)²` |
| **梯度** | `±1`（常数） | `2(ŷ - y)`（线性） |
| **对异常值** | 鲁棒 | 敏感 |
| **可导性** | 0 点不可导 | 处处可导 |
| **收敛速度** | 稳定但慢 | 快但可能震荡 |
| **解的性质** | 中位数 | 均值 |

### 2.2 曲线对比

```
Loss
 │
 │     L2 (MSE)        L1 (MAE)
 │      ╲    ╱          ╲    ╱
 │       ╲  ╱            ╲  ╱
 │        ╲╱              ╲╱
 │     (抛物线)         (V 字形)
 └──────────────────────────── error (y - ŷ)
        0                  0
```

- **L2**：抛物线，误差大时惩罚急剧增大（平方）
- **L1**：V 字形，惩罚线性增长

### 2.3 对异常值的鲁棒性

**关键区别**：L1 对异常值（outliers）更鲁棒。

```python
# 假设有一个异常值
真实值 y = [1.0, 2.0, 3.0, 100.0]  # 100 是异常值
预测值 ŷ = [1.0, 2.0, 3.0, 4.0]

# L1 Loss
L1 = (0 + 0 + 0 + 96) / 4 = 24.0

# L2 Loss
L2 = (0 + 0 + 0 + 96²) / 4 = 2304.0

# L2 被异常值主导（96² = 9216），模型会过度拟合异常值
# L1 相对温和，不会被单个异常值严重影响
```

**结论**：
- **数据有噪声/异常值** → 用 L1（鲁棒）
- **数据干净，需要精确拟合** → 用 L2（收敛快）

### 2.4 梯度行为对比

```python
# L1 梯度：符号函数，恒为 ±1
∂L1/∂ŷ = sign(ŷ - y) = { +1  if ŷ > y
                        { -1  if ŷ < y
                        { 0   if ŷ = y (理论上未定义)

# L2 梯度：与误差成正比
∂L2/∂ŷ = 2(ŷ - y)
```

**影响**：
- **L1**：无论误差大小，梯度恒定，接近最优时可能来回震荡
- **L2**：误差大时梯度大（快速下降），误差小时梯度小（精细调整）

---

## 3. L1 Loss 的问题与改进

### 3.1 零点不可导问题

**问题**：L1 在 `y = ŷ` 处不可导（V 字形尖点）。

```
∂|x|/∂x = { +1   x > 0
          { -1   x < 0
          { 未定义  x = 0
```

**实际处理**：
- PyTorch 等框架在 0 点将梯度设为 0（次梯度）
- 通常不影响训练，因为精确等于 0 的概率极低

### 3.2 Smooth L1 Loss（Huber Loss 变体）

**动机**：结合 L1 的鲁棒性和 L2 的平滑性。

**定义**：

```
SmoothL1(x) = { 0.5 x²/β        if |x| < β
              { |x| - 0.5β      otherwise
```

其中 `x = y - ŷ`，`β` 是阈值（默认 1.0）。

**分段解释**：
- **小误差（|x| < β）**：使用 L2（平滑，梯度随误差减小）
- **大误差（|x| ≥ β）**：使用 L1（鲁棒，梯度恒定）

**曲线**：

```
Loss
 │
 │  L1 部分         L1 部分
 │    ╲              ╱
 │     ╲            ╱
 │      ╲___    ___╱
 │          ╲__╱      ← L2 部分（平滑过渡）
 └──────────────────── error
       -β    0    β
```

**优势**：
- ✅ 0 点附近平滑可导（来自 L2）
- ✅ 对异常值鲁棒（来自 L1）
- ✅ 目标检测框回归的标准选择

### 3.3 三者对比

| 损失 | 小误差行为 | 大误差行为 | 0 点可导 | 异常值鲁棒 |
|------|-----------|-----------|---------|-----------|
| **L1** | 线性 | 线性 | ❌ | ✅ |
| **L2** | 二次 | 二次 | ✅ | ❌ |
| **Smooth L1** | 二次（L2） | 线性（L1） | ✅ | ✅ |

---

## 4. PyTorch 实现

### 4.1 `nn.L1Loss`

```python
import torch
import torch.nn as nn

# 类形式
criterion = nn.L1Loss(reduction='mean')  # 'none' | 'mean' | 'sum'
loss = criterion(predictions, targets)

# 函数形式
import torch.nn.functional as F
loss = F.l1_loss(predictions, targets, reduction='mean')
```

**参数说明**：
- `reduction`：
  - `'none'`：返回每个元素的损失 `(N, *)`
  - `'mean'`：返回所有元素的平均（默认）
  - `'sum'`：返回所有元素之和

**示例**：

```python
predictions = torch.tensor([2.5, 6.0, 2.0])
targets = torch.tensor([3.0, 5.0, 2.0])

# 平均
loss_mean = F.l1_loss(predictions, targets)  # 0.5

# 逐元素
loss_none = F.l1_loss(predictions, targets, reduction='none')  # [0.5, 1.0, 0.0]

# 求和
loss_sum = F.l1_loss(predictions, targets, reduction='sum')  # 1.5
```

### 4.2 `nn.SmoothL1Loss`

```python
# Smooth L1（beta 默认 1.0）
criterion = nn.SmoothL1Loss(beta=1.0, reduction='mean')
loss = criterion(predictions, targets)

# 函数形式
loss = F.smooth_l1_loss(predictions, targets, beta=1.0)
```

**beta 参数**：
- 控制 L1 和 L2 的切换阈值
- `beta` 越小，越接近 L1
- `beta` 越大，越接近 L2

### 4.3 `nn.HuberLoss`

```python
# Huber Loss（PyTorch 1.9+）
criterion = nn.HuberLoss(delta=1.0, reduction='mean')
loss = criterion(predictions, targets)
```

**Smooth L1 vs Huber 的关系**：

```
HuberLoss(delta) = beta × SmoothL1Loss(beta=delta)
```

- **Smooth L1**：`0.5 x²/β` 和 `|x| - 0.5β`
- **Huber**：`0.5 x²` 和 `δ(|x| - 0.5δ)`
- 数值上相差一个 `beta/delta` 因子，但形状一致

### 4.4 手动实现

```python
def l1_loss_manual(pred, target, reduction='mean'):
    loss = torch.abs(pred - target)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def smooth_l1_manual(pred, target, beta=1.0, reduction='mean'):
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,   # L2 部分
        diff - 0.5 * beta          # L1 部分
    )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
```

---

## 5. 应用场景

### 5.1 目标检测的边界框回归

**最经典的应用**：Faster R-CNN、SSD、YOLO 等使用 Smooth L1 回归框坐标。

```python
class BBoxRegressionLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
    
    def forward(self, pred_boxes, target_boxes, weights=None):
        """
        Args:
            pred_boxes: (N, 4) 预测框 [dx, dy, dw, dh]
            target_boxes: (N, 4) 目标框偏移
            weights: (N,) 每个框的权重（正样本为 1，负样本为 0）
        """
        loss = self.smooth_l1(pred_boxes, target_boxes)  # (N, 4)
        loss = loss.sum(dim=1)  # (N,) 每个框的损失
        
        if weights is not None:
            loss = loss * weights
            return loss.sum() / (weights.sum() + 1e-6)
        return loss.mean()

# 使用
criterion = BBoxRegressionLoss(beta=1.0 / 9.0)  # RetinaNet 常用 beta
loss = criterion(pred_boxes, target_boxes, weights=pos_mask)
```

**为什么用 Smooth L1 而非 L2**：
- 框坐标可能有标注误差（异常值）
- Smooth L1 对大误差更鲁棒，训练更稳定

### 5.2 3D 检测的框回归（Qwen-Drive-1.0 风格）

```python
# 在 Qwen-Drive-1.0 的检测头中，框回归使用 L1 类损失
# 框编码：[x, y, w, l, cz, h, sin, cos, vx, vy] (10 维)

class Box3DRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred_bboxes, target_bboxes, code_weights=None):
        """
        Args:
            pred_bboxes: (N, 10) 预测框
            target_bboxes: (N, 10) 目标框
            code_weights: (10,) 每个维度的权重
        """
        loss = self.l1_loss(pred_bboxes, target_bboxes)  # (N, 10)
        
        # 对不同维度加权（如角度、速度权重不同）
        if code_weights is not None:
            loss = loss * code_weights.view(1, -1)
        
        return loss.sum(dim=1).mean()

# 使用（DETR3D / BEVFormer 常用配置）
code_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2])
criterion = Box3DRegressionLoss()
loss = criterion(pred_bboxes, target_bboxes, code_weights)
```

**注意**：
- 位置/尺寸维度权重为 1.0
- 角度（sin/cos）和速度维度权重较小（0.2），避免主导损失

### 5.3 图像重建/生成

```python
# 图像超分辨率、去噪、生成任务
class ImageReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred_img, target_img):
        """
        Args:
            pred_img: (B, C, H, W) 重建图像
            target_img: (B, C, H, W) 目标图像
        """
        return self.l1(pred_img, target_img)

# 使用（pix2pix、CycleGAN 等）
criterion = ImageReconstructionLoss()
loss = criterion(generated_img, real_img)
```

**为什么图像任务偏爱 L1**：
- L1 生成的图像更锐利（L2 倾向于产生模糊结果）
- L2 的平方惩罚会让模型输出"平均"结果，导致模糊
- L1 保留更多高频细节

### 5.4 深度估计

```python
class DepthEstimationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_depth, gt_depth, mask=None):
        """
        Args:
            pred_depth: (B, H, W) 预测深度
            gt_depth: (B, H, W) 真实深度
            mask: (B, H, W) 有效区域掩码
        """
        loss = torch.abs(pred_depth - gt_depth)
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

# 使用
criterion = DepthEstimationLoss()
loss = criterion(pred_depth, gt_depth, valid_mask)
```

### 5.5 关键点/姿态回归

```python
# 人体姿态估计、关键点检测
class KeypointLoss(nn.Module):
    def __init__(self, use_smooth=True):
        super().__init__()
        if use_smooth:
            self.loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred_kpts, target_kpts, visibility):
        """
        Args:
            pred_kpts: (N, K, 2) 预测关键点坐标
            target_kpts: (N, K, 2) 真实关键点
            visibility: (N, K) 关键点可见性
        """
        loss = self.loss(pred_kpts, target_kpts)  # (N, K, 2)
        loss = loss.sum(dim=-1)  # (N, K)
        
        # 只计算可见关键点
        loss = loss * visibility
        return loss.sum() / (visibility.sum() + 1e-6)
```

---

## 6. 超参数与技巧

### 6.1 Smooth L1 的 beta 选择

| beta | 行为 | 适用场景 |
|------|------|---------|
| **0.11 (1/9)** | 接近 L1 | RetinaNet 框回归 |
| **1.0** | 标准平衡 | 通用回归 |
| **3.0+** | 接近 L2 | 数据干净，需精确拟合 |

**调优建议**：
```python
# 目标检测常用配置
# Faster R-CNN: beta = 1.0
# RetinaNet: beta = 1.0 / 9.0
# SSD: beta = 1.0

# 根据误差分布调整
# 如果预测误差大多在 [0, β] 内 → 主要用 L2 行为
# 如果预测误差大多超过 β → 主要用 L1 行为
```

### 6.2 归一化目标值

**重要**：回归任务中，目标值的尺度影响损失大小。

```python
# 不同维度尺度差异大时需要归一化
# 例：3D 框 [x(±50m), y(±50m), z(±5m), w(1-20m), ...]

# 方法1：标准化目标
target_normalized = (target - mean) / std
pred_normalized = model(x)
loss = F.l1_loss(pred_normalized, target_normalized)

# 方法2：使用 code_weights 平衡不同维度
code_weights = torch.tensor([1.0, 1.0, 2.0, ...])  # z 权重更大
loss = (F.l1_loss(pred, target, reduction='none') * code_weights).mean()
```

### 6.3 与其他损失结合

```python
class CombinedRegressionLoss(nn.Module):
    """L1 + IoU Loss（目标检测常用）"""
    def __init__(self, l1_weight=1.0, iou_weight=2.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_boxes, target_boxes):
        # L1 损失（框坐标）
        l1 = self.l1_loss(pred_boxes, target_boxes)
        
        # GIoU 损失（框重叠度）
        iou = self.giou_loss(pred_boxes, target_boxes)
        
        return self.l1_weight * l1 + self.iou_weight * iou
    
    def giou_loss(self, pred, target):
        # 计算 GIoU（略）
        ...
```

**DETR 系列的配置**：
```python
# DETR/DETR3D 的框损失
loss = λ_L1 * L1_loss + λ_GIoU * GIoU_loss
# 典型权重：λ_L1 = 5.0, λ_GIoU = 2.0
```

---

## 7. 常见问题与调试

### 7.1 训练震荡

**原因**：L1 梯度恒定，接近最优时可能来回跳动。

**解决方案**：

```python
# 1. 使用 Smooth L1 替代纯 L1
criterion = nn.SmoothL1Loss(beta=1.0)

# 2. 学习率衰减
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. 后期切换到 L2
if epoch > threshold:
    criterion = nn.MSELoss()
```

### 7.2 收敛慢

**原因**：L1 梯度不随误差减小，精细调整能力弱。

**解决方案**：

```python
# 使用 Smooth L1（小误差时用 L2 加速收敛）
criterion = nn.SmoothL1Loss(beta=1.0)

# 或提高初始学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 7.3 不同维度尺度不平衡

**诊断代码**：

```python
# 检查各维度的损失贡献
with torch.no_grad():
    per_dim_loss = F.l1_loss(pred, target, reduction='none').mean(dim=0)
    print(f"各维度损失: {per_dim_loss}")
    # 如果某维度损失远大于其他，需要归一化或加权
```

---

## 8. 损失函数选择指南

### 8.1 回归任务决策树

```
回归任务
│
├─ 数据有异常值/噪声？
│   ├─ 是 → L1 或 Smooth L1
│   └─ 否 → 继续
│
├─ 需要精确拟合？
│   ├─ 是 → L2 (MSE)
│   └─ 否 → 继续
│
├─ 目标检测框回归？
│   └─ Smooth L1 + IoU Loss
│
├─ 图像生成/重建？
│   └─ L1（避免模糊）
│
└─ 通用回归 → Smooth L1（平衡选择）
```

### 8.2 对比总结表

| 损失 | 公式 | 优点 | 缺点 | 典型应用 |
|------|------|------|------|---------|
| **L1 (MAE)** | `\|y-ŷ\|` | 鲁棒，锐利 | 0点不可导，收敛慢 | 图像生成 |
| **L2 (MSE)** | `(y-ŷ)²` | 平滑，收敛快 | 对异常值敏感 | 干净数据回归 |
| **Smooth L1** | 分段 | 兼具两者优点 | 需调 beta | 目标检测 |
| **Huber** | 分段 | 同 Smooth L1 | 需调 delta | 鲁棒回归 |

---

## 9. 总结

### 核心要点

1. ✅ **L1 = 平均绝对误差**：`|y - ŷ|`，单位与数据一致
2. ✅ **对异常值鲁棒**：不像 L2 会被异常值主导
3. ✅ **梯度恒定**：`±1`，接近最优时可能震荡
4. ✅ **0 点不可导**：实际用次梯度（设为 0）
5. ✅ **Smooth L1 是改进**：小误差用 L2，大误差用 L1
6. ✅ **图像任务偏爱 L1**：生成结果更锐利

### 快速参考

```python
import torch.nn as nn
import torch.nn.functional as F

# L1 Loss
loss = F.l1_loss(pred, target)

# Smooth L1 Loss（推荐用于检测）
loss = F.smooth_l1_loss(pred, target, beta=1.0)

# Huber Loss
criterion = nn.HuberLoss(delta=1.0)
loss = criterion(pred, target)

# 带权重的框回归
loss = (F.l1_loss(pred, target, reduction='none') * code_weights).sum(-1).mean()

# 带掩码（只算有效样本）
loss = (F.l1_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
```

### 选择建议

| 场景 | 推荐损失 |
|------|---------|
| **目标检测框回归** | Smooth L1 (beta=1.0 或 1/9) |
| **3D 检测框回归** | L1 + code_weights |
| **图像生成/重建** | L1 |
| **深度估计** | L1 或 Smooth L1 |
| **数据有异常值** | L1 或 Huber |
| **干净数据精确拟合** | L2 (MSE) |

---

## 参考资源

- [PyTorch 官方文档：nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
- [PyTorch 官方文档：nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)
- [Fast R-CNN 论文（Smooth L1 来源）](https://arxiv.org/abs/1504.08083)
- **相关文档**：[cross_entropy_loss.md](./cross_entropy_loss.md)、[facal_loss.md](./facal_loss.md)

---
