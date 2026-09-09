# Focal Loss 原理与用途

## 一、问题背景：为什么需要 Focal Loss？

### 1.1 目标检测中的类别不平衡

在目标检测任务中（如 YOLO、Faster R-CNN、RetinaNet），存在严重的**正负样本不平衡**问题：

**传统检测器的样本统计**：
- **负样本（背景）**：数万个候选框，绝大多数是背景
- **正样本（目标）**：只有几十到几百个真实目标
- **不平衡比例**：1:1000 甚至更高

**使用标准交叉熵的问题**：

```python
# 假设 10,000 个候选框
# 正样本：50 个，负样本：9,950 个

# 即使负样本损失很小（易分类），累加起来仍然主导总损失
total_loss = sum(easy_negative_losses) + sum(hard_positive_losses)
           ≈ 9,950 × 0.01 + 50 × 1.0
           ≈ 99.5 + 50 = 149.5

# 负样本贡献 66%，模型被迫关注大量简单的负样本
```

**导致的后果**：
1. ❌ 训练效率低下：大量计算浪费在易分类样本上
2. ❌ 模型退化：过度关注简单样本，忽略困难样本
3. ❌ 收敛困难：梯度被大量简单样本主导

### 1.2 现有解决方案的局限

| 方法 | 原理 | 缺点 |
|------|------|------|
| **困难负样本挖掘 (OHEM)** | 选择损失最大的负样本 | 需要额外的采样步骤，训练复杂 |
| **两阶段检测器** | RPN + RCNN 分离前景/背景 | 速度慢，不是端到端 |
| **类别权重** | 手动设置正负样本权重 | 需要调参，无法处理样本内部难易差异 |

**Focal Loss 的创新**：
✅ **自动降权易分类样本**  
✅ **无需采样或两阶段**  
✅ **端到端训练**  
✅ **参数简单易调**

---

## 二、Focal Loss 的数学原理

### 2.1 从交叉熵到 Focal Loss

**标准交叉熵 (CE)**：

```
CE(p, y) = -log(p_t)
```

其中 `p_t` 定义为：

```
p_t = { p     如果 y = 1  (正样本)
      { 1-p   如果 y = 0  (负样本)
```

**问题分析**：

| 样本类型 | p_t | -log(p_t) | 问题 |
|---------|-----|-----------|------|
| 易分类正样本 | 0.99 | 0.01 | ✅ 损失小，合理 |
| 难分类正样本 | 0.51 | 0.67 | ✅ 损失大，合理 |
| 易分类负样本 | 0.99 | 0.01 | ❌ 虽然单个小，但数量巨大 |
| 难分类负样本 | 0.51 | 0.67 | ✅ 损失大，合理 |

**Focal Loss (FL)**：

```
FL(p_t) = -(1 - p_t)^γ log(p_t)
```

**调制因子 (Modulating Factor)**：`(1 - p_t)^γ`

- 当样本**易分类**（`p_t → 1`）：`(1 - p_t)^γ → 0`，损失被大幅降低
- 当样本**难分类**（`p_t → 0.5`）：`(1 - p_t)^γ ≈ 0.5^γ`，损失保持较大

**γ (gamma) 参数的作用**：

```
γ = 0  →  FL = CE         (无调制，等价于标准 CE)
γ = 1  →  (1-0.99)^1 = 0.01   (线性降权)
γ = 2  →  (1-0.99)^2 = 0.0001 (平方降权，推荐)
γ = 5  →  (1-0.99)^5 ≈ 1e-10  (极端降权)
```

### 2.2 数值示例

假设 `γ = 2`：

| p_t | CE Loss | (1-p_t)^2 | FL Loss | 降权比例 |
|-----|---------|-----------|---------|----------|
| 0.99 | 0.010 | 0.0001 | **0.000001** | **99.99%** ↓ |
| 0.9 | 0.105 | 0.01 | **0.001** | **99%** ↓ |
| 0.7 | 0.357 | 0.09 | **0.032** | **91%** ↓ |
| 0.5 | 0.693 | 0.25 | **0.173** | 75% ↓ |

**直观理解**：
- 当模型以 99% 置信度正确分类时，FL 几乎忽略该样本（降权 99.99%）
- 当模型只有 70% 置信度时，FL 仍保留较大损失（降权 91%）
- 难分类样本（p_t ≈ 0.5）的损失被保留，成为训练重点

### 2.3 α-balanced Focal Loss

**完整版本**：

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

**α (alpha) 参数**：类别平衡权重

```
α_t = { α       如果 y = 1  (正样本)
      { 1-α     如果 y = 0  (负样本)
```

**作用**：
- **α ∈ [0, 1]**：调整正负样本的相对重要性
- **推荐值**：α = 0.25（正样本权重 0.25，负样本权重 0.75）
- **原因**：负样本数量多但易分类，正样本少但重要

**示例**：

```python
# 假设 α = 0.25, γ = 2

# 正样本（y=1），p=0.9
α_t = 0.25
p_t = 0.9
loss = -0.25 × (1-0.9)^2 × log(0.9)
     = -0.25 × 0.01 × (-0.105)
     = 0.000263

# 负样本（y=0），p=0.1（即 p_t=0.9）
α_t = 0.75
p_t = 0.9
loss = -0.75 × (1-0.9)^2 × log(0.9)
     = -0.75 × 0.01 × (-0.105)
     = 0.000788
```

---

## 三、Focal Loss 的梯度分析

### 3.1 梯度推导

对于二分类（sigmoid 输出 p，logit 为 z）：

```
FL = -α(1-p)^γ log(p)           (y=1)
FL = -(1-α)p^γ log(1-p)         (y=0)
```

**梯度**（以 y=1 为例）：

```
∂FL/∂z = ∂FL/∂p × ∂p/∂z
       = -α × [γ(1-p)^(γ-1) × (-1) × log(p) + (1-p)^γ × 1/p] × p(1-p)
       = α(1-p)^γ × [γ(p-1)log(p)/p + (1-p)] × p(1-p)
       ≈ α(1-p)^(γ+1) × (p - y_true)  （简化形式）
```

**关键观察**：

1. **易分类样本（p → 1）**：梯度 ∝ `(1-p)^(γ+1) → 0`，几乎不更新
2. **难分类样本（p ≈ 0.5）**：梯度保持较大，持续优化
3. **误分类样本（p → 0）**：梯度 ∝ `1^(γ+1) × (-1)` = -1，强烈纠正

### 3.2 与 CE 梯度对比

| 损失 | 梯度（y=1） | 易分类样本（p=0.9） | 难分类样本（p=0.6） |
|------|------------|---------------------|---------------------|
| **CE** | `p - 1` | -0.1 | -0.4 |
| **FL (γ=2)** | `(1-p)^3 × (p-1)` | -0.001 | -0.026 |
| **降权比例** | - | **99%** ↓ | **93.5%** ↓ |

**结论**：FL 的梯度对易分类样本的下降更剧烈，训练自动聚焦于困难样本。

---

## 四、PyTorch 实现

### 4.1 二分类 Focal Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 正样本权重 (0-1)
            gamma: 调制指数 (>= 0)
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N,) logits (未经 sigmoid)
            targets: (N,) 标签 {0, 1}
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算 p_t
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算 alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用
criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
```

### 4.2 多分类 Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: (C,) 每个类别的权重，或标量
            gamma: 调制指数
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) 类别索引
        """
        # 标准交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 p_t
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        
        # Focal 调制因子
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用 alpha（如果提供）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Focal Loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用
num_classes = 10
criterion = FocalLoss(
    alpha=torch.ones(num_classes) * 0.25,
    gamma=2.0
)
loss = criterion(logits, targets)
```

### 4.3 高效实现（向量化）

```python
class FocalLossOptimized(nn.Module):
    """更高效的实现，避免 gather 操作"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) or (N, C) one-hot
        """
        p = F.softmax(inputs, dim=-1)
        
        # 支持整数索引或 one-hot
        if targets.ndim == 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:  # one-hot
            ce_loss = -(targets * F.log_softmax(inputs, dim=-1)).sum(dim=1)
            p_t = (p * targets).sum(dim=1)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

---

## 五、实际应用场景

### 5.1 目标检测（RetinaNet）

**RetinaNet 架构**：单阶段检测器，Focal Loss 的首次应用

```python
class RetinaNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = nn.SmoothL1Loss()
    
    def forward(self, cls_preds, box_preds, cls_targets, box_targets):
        """
        Args:
            cls_preds: (N, num_anchors, num_classes) 分类预测
            box_preds: (N, num_anchors, 4) 框回归预测
            cls_targets: (N, num_anchors) 类别标签（-1=ignore, 0=bg, >0=fg）
            box_targets: (N, num_anchors, 4) 框回归目标
        """
        # 分离前景、背景、忽略样本
        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        ignore_mask = cls_targets == -1
        
        # 分类损失（Focal Loss）
        valid_mask = ~ignore_mask
        cls_loss = self.focal_loss(
            cls_preds[valid_mask], 
            cls_targets[valid_mask]
        )
        
        # 回归损失（仅对前景）
        if pos_mask.sum() > 0:
            box_loss = self.box_loss(
                box_preds[pos_mask], 
                box_targets[pos_mask]
            )
        else:
            box_loss = torch.tensor(0.0, device=cls_preds.device)
        
        return cls_loss + box_loss

# 使用
loss_fn = RetinaNetLoss()
total_loss = loss_fn(cls_preds, box_preds, cls_targets, box_targets)
```

**效果**：
- nuScenes 2D 检测：mAP 提升 2-3%
- COCO 数据集：相比两阶段检测器（Faster R-CNN）速度快 5 倍，精度持平

### 5.2 语义分割（密集预测）

**问题**：边界像素困难，内部像素简单

```python
class SegmentationFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) 像素级分类 logits
            targets: (B, H, W) 像素级标签
        """
        B, C, H, W = logits.shape
        
        # 展平
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets = targets.reshape(-1)  # (B*H*W,)
        
        # 过滤 ignore_index
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]
        
        # Focal Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# 使用
criterion = SegmentationFocalLoss(alpha=0.25, gamma=2.0, ignore_index=255)
loss = criterion(seg_logits, seg_targets)
```

### 5.3 3D 目标检测

**Qwen-Drive-1.0 风格**（基于 query 的检测）：

```python
class QueryBasedFocalLoss(nn.Module):
    """用于 DETR 系列检测器的 Focal Loss"""
    def __init__(self, num_classes=7, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, cls_logits, targets, indices):
        """
        Args:
            cls_logits: (B, num_queries, num_classes) 分类 logits
            targets: list of dicts，每个包含 'labels'
            indices: 匈牙利匹配结果
        """
        # 构建目标张量
        target_classes = torch.full(
            cls_logits.shape[:2], 
            self.num_classes,  # 背景类
            dtype=torch.long, 
            device=cls_logits.device
        )
        
        # 填入匹配的正样本
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[batch_idx, src_idx] = targets[batch_idx]['labels'][tgt_idx]
        
        # Focal Loss（包括背景类）
        loss = self._focal_loss(cls_logits, target_classes)
        return loss
    
    def _focal_loss(self, inputs, targets):
        """内部 Focal Loss 计算"""
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(
            inputs.view(-1, self.num_classes + 1), 
            targets.view(-1), 
            reduction='none'
        )
        p_t = p.view(-1, self.num_classes + 1).gather(1, targets.view(-1, 1)).squeeze(1)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 5.4 文本分类（极端不平衡）

```python
class TextClassificationFocalLoss(nn.Module):
    """处理长尾分布的文本分类"""
    def __init__(self, class_freq, gamma=2.0):
        """
        Args:
            class_freq: (C,) 每个类别的频率
            gamma: 调制指数
        """
        super().__init__()
        self.gamma = gamma
        # 逆频率加权
        self.alpha = (1.0 / class_freq) / (1.0 / class_freq).sum()
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        
        alpha_t = self.alpha.to(logits.device)[targets]
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# 使用示例
class_freq = torch.tensor([10000, 1000, 100, 10])  # 类别频率
criterion = TextClassificationFocalLoss(class_freq, gamma=2.0)
```

---

## 六、超参数调优

### 6.1 γ (gamma) 的选择

| γ | 效果 | 适用场景 |
|---|------|---------|
| **0** | 无调制（= CE） | 类别平衡 |
| **0.5** | 轻度降权 | 轻微不平衡（1:10） |
| **1.0** | 中度降权 | 中度不平衡（1:100） |
| **2.0** | 强降权（推荐） | 严重不平衡（1:1000） |
| **5.0** | 极端降权 | 极端不平衡（1:10000+） |

**实验建议**：
1. 从 γ=2 开始（论文推荐）
2. 如果模型欠拟合（loss 降不下来），减小 γ 到 1.0
3. 如果模型过拟合（训练 loss 很低，验证 loss 高），增大 γ 到 3.0

### 6.2 α (alpha) 的选择

**经验公式**：

```python
# 方法1：根据正负样本比例
num_pos = 100
num_neg = 10000
alpha = num_neg / (num_pos + num_neg)  # ≈ 0.99

# 方法2：论文推荐值（适用于大多数场景）
alpha = 0.25  # 正样本权重 0.25，负样本权重 0.75

# 方法3：网格搜索
for alpha in [0.1, 0.25, 0.5, 0.75]:
    for gamma in [0.5, 1.0, 2.0, 3.0]:
        validate(alpha, gamma)
```

**调优建议**：
1. **目标检测**：α=0.25, γ=2.0（RetinaNet 原始设置）
2. **语义分割**：α=0.5, γ=2.0（前景背景更平衡）
3. **极端不平衡**：α=0.75-0.9, γ=3.0-5.0

### 6.3 与其他技巧结合

```python
class CombinedLoss(nn.Module):
    """Focal Loss + Label Smoothing + Class Weighting"""
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, 
                 label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets_smooth = self._smooth_labels(targets)
        else:
            targets_smooth = F.one_hot(targets, self.num_classes).float()
        
        # 计算损失
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        # Focal 调制
        probs = torch.exp(log_probs)
        p_t = (probs * targets_smooth).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma
        
        # 类别权重
        if self.class_weights is not None:
            class_weight = (self.class_weights.to(logits.device) * targets_smooth).sum(dim=-1)
            focal_weight = focal_weight * class_weight
        
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()
    
    def _smooth_labels(self, targets):
        """标签平滑"""
        smooth_targets = torch.full(
            (targets.size(0), self.num_classes),
            self.label_smoothing / (self.num_classes - 1),
            device=targets.device
        )
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return smooth_targets
```

---

## 七、与其他损失函数对比

### 7.1 性能对比表

| 损失函数 | mAP (COCO) | 训练时间 | 收敛速度 | 调参难度 |
|---------|-----------|---------|---------|---------|
| **CE** | 30.5 | 1.0× | 慢 | 简单 |
| **Weighted CE** | 32.1 | 1.0× | 慢 | 中等 |
| **OHEM** | 33.4 | 1.3× | 中 | 困难 |
| **Focal Loss** | **35.7** | **1.0×** | **快** | **简单** |

### 7.2 适用场景对比

| 场景 | CE | Focal Loss | 其他 |
|------|-------|-----------|------|
| **类别平衡** | ✅ 最佳 | ⚠️ 过杀 | - |
| **轻度不平衡 (1:10)** | ⚠️ 可用 | ✅ 推荐 | Weighted CE |
| **严重不平衡 (1:1000)** | ❌ 失效 | ✅ 最佳 | OHEM（次选） |
| **目标检测** | ❌ | ✅ 标准 | - |
| **语义分割** | ⚠️ | ✅ 推荐 | Dice Loss |
| **文本分类** | ✅ | ✅ | - |

### 7.3 代码对比

```python
# 标准 CE
loss = F.cross_entropy(logits, targets)

# Weighted CE（需要手动计算权重）
weights = compute_class_weights(train_labels)
loss = F.cross_entropy(logits, targets, weight=weights)

# OHEM（需要额外采样逻辑）
losses = F.cross_entropy(logits, targets, reduction='none')
hard_examples = losses.topk(k=batch_size // 2).indices
loss = losses[hard_examples].mean()

# Focal Loss（自动处理）
loss = focal_loss(logits, targets, alpha=0.25, gamma=2.0)
```

---

## 八、常见问题与调试

### 8.1 Loss 爆炸 / NaN

**原因**：
- logits 数值过大（如 z=1000）
- 学习率过高

**解决方案**：

```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 学习率预热
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.1, 
    total_iters=500
)

# 3. 检查数值稳定性
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Logits range: {logits.min()}, {logits.max()}")
    print(f"Targets: {targets}")
```

### 8.2 模型不收敛

**可能原因**：
- γ 过大，梯度过小
- α 设置不当

**诊断代码**：

```python
# 记录不同难度样本的损失
with torch.no_grad():
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    easy_mask = p_t > 0.9
    hard_mask = p_t < 0.5
    
    easy_loss = focal_loss[easy_mask].mean()
    hard_loss = focal_loss[hard_mask].mean()
    
    print(f"Easy samples: {easy_mask.sum()}, loss: {easy_loss}")
    print(f"Hard samples: {hard_mask.sum()}, loss: {hard_loss}")
    
    # 如果 easy_loss >> hard_loss，说明 γ 过大
```

### 8.3 训练不稳定

```python
# 使用混合精度训练时，Focal Loss 可能数值不稳定
# 解决方案：在 float32 下计算损失

from torch.cuda.amp import autocast

with autocast():
    logits = model(x)

# 损失计算不使用 autocast
with autocast(enabled=False):
    loss = focal_loss(logits.float(), targets)

scaler.scale(loss).backward()
```

---

## 九、代码库与工具

### 9.1 官方实现（detectron2）

```python
# 安装 detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# 使用
from detectron2.layers import sigmoid_focal_loss

loss = sigmoid_focal_loss(
    inputs,           # (N, num_classes) logits
    targets,          # (N, num_classes) {0, 1}
    alpha=0.25,
    gamma=2.0,
    reduction='mean'
)
```

### 9.2 第三方库

```python
# torchvision（内置）
from torchvision.ops import sigmoid_focal_loss

# kornia（视觉库）
from kornia.losses import focal_loss

# segmentation_models_pytorch（分割专用）
import segmentation_models_pytorch as smp
loss = smp.losses.FocalLoss(mode='multiclass')
```

---

## 十、总结

### 核心要点

1. ✅ **自动降权易分类样本**：`(1-p_t)^γ` 调制因子
2. ✅ **聚焦困难样本**：训练效率提升，模型性能提升
3. ✅ **无需采样/两阶段**：端到端训练，实现简单
4. ✅ **超参数简单**：α=0.25, γ=2.0 适用于大多数场景
5. ✅ **广泛应用**：目标检测、语义分割、3D 检测、文本分类

### 推荐配置

| 任务 | α | γ | 备注 |
|------|---|---|------|
| **目标检测** | 0.25 | 2.0 | RetinaNet 原始设置 |
| **语义分割** | 0.5 | 2.0 | 前景背景更平衡 |
| **3D 检测** | 0.25 | 2.0 | 与 2D 检测一致 |
| **极端不平衡** | 0.75 | 3.0-5.0 | 增大 γ 进一步降权 |

### 快速上手

```python
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = F.softmax(inputs, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

# 使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
```

---

## 参考资源

- **原始论文**：[Focal Loss for Dense Object Detection (ICCV 2017)](https://arxiv.org/abs/1708.02002)
- **RetinaNet 代码**：https://github.com/facebookresearch/detectron2
- **PyTorch 讨论**：https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification
- **相关文档**：[cross_entropy_loss.md](./cross_entropy_loss.md)

---

*最后更新：2026 年 9 月*
