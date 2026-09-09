# CrossEntropyLoss 原理与应用

## 一、什么是交叉熵损失

### 1.1 信息论基础

**熵 (Entropy)**：衡量信息的不确定性或混乱程度

```
H(P) = -∑ P(x) log P(x)
```

- 概率分布越均匀，熵越大（不确定性高）
- 概率分布越集中，熵越小（不确定性低）

**交叉熵 (Cross Entropy)**：衡量两个概率分布之间的差异

```
H(P, Q) = -∑ P(x) log Q(x)
```

- `P(x)`：真实分布
- `Q(x)`：预测分布
- 交叉熵越小，两个分布越接近

**KL 散度 (Kullback-Leibler Divergence)**：

```
D_KL(P || Q) = ∑ P(x) log(P(x) / Q(x))
             = ∑ P(x) log P(x) - ∑ P(x) log Q(x)
             = -H(P) + H(P, Q)
```

**关键关系**：最小化交叉熵 ≈ 最小化 KL 散度（因为 H(P) 是常数）

### 1.2 从概率角度理解

**最大似然估计 (MLE)**：

给定数据集 `D = {(x₁, y₁), ..., (xₙ, yₙ)}`，我们希望找到参数 θ 使得：

```
θ* = argmax_θ ∏ P(yᵢ | xᵢ; θ)
```

取对数（log-likelihood）：

```
θ* = argmax_θ ∑ log P(yᵢ | xᵢ; θ)
   = argmin_θ -∑ log P(yᵢ | xᵢ; θ)
```

这正是**交叉熵损失**的形式！

---

## 二、数学定义

### 2.1 二分类交叉熵 (Binary Cross Entropy)

**问题设定**：
- 真实标签：`y ∈ {0, 1}`
- 模型输出：`ŷ = sigmoid(z) ∈ (0, 1)`

**损失函数**：

```
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**直观理解**：
- 当 `y=1` 时，希望 `ŷ→1`，损失为 `-log(ŷ)`
- 当 `y=0` 时，希望 `ŷ→0`，损失为 `-log(1-ŷ)`

**PyTorch 实现**：

```python
import torch.nn.functional as F

# 方法1：手动计算
loss = -(y * torch.log(y_pred) + (1-y) * torch.log(1-y_pred))

# 方法2：PyTorch API
loss = F.binary_cross_entropy(y_pred, y)  # y_pred 已经过 sigmoid
# 或
loss = F.binary_cross_entropy_with_logits(logits, y)  # logits 未经 sigmoid
```

### 2.2 多分类交叉熵 (Categorical Cross Entropy)

**问题设定**：
- 类别数：C
- 真实标签：`y ∈ {0, 1, ..., C-1}` (整数) 或 one-hot 向量
- 模型输出（logits）：`z = [z₁, z₂, ..., z_C]`
- 概率分布：`p = softmax(z)`

**Softmax 函数**：

```
p_i = exp(z_i) / ∑ⱼ exp(z_j)
```

**交叉熵损失**：

```
CE = -∑ᵢ y_i log(p_i)
```

当标签是整数（类别索引 c）时：

```
CE = -log(p_c) = -log(exp(z_c) / ∑ⱼ exp(z_j))
   = -z_c + log(∑ⱼ exp(z_j))
   = -z_c + LogSumExp(z)
```

**PyTorch 实现**：

```python
# 方法1：分步计算
probs = F.softmax(logits, dim=-1)  # (N, C)
loss = F.nll_loss(torch.log(probs), target)  # target: (N,) 整数

# 方法2：PyTorch API（推荐，数值稳定）
loss = F.cross_entropy(logits, target)  # logits 未经 softmax
```

**重要**：`F.cross_entropy` 内部会自动应用 softmax，**不要**手动先 softmax！

---

## 三、数值稳定性

### 3.1 问题：直接计算的数值溢出

**朴素实现**：

```python
def naive_cross_entropy(logits, target):
    probs = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)
    return -torch.log(probs[range(len(target)), target]).mean()
```

**问题**：
- `exp(z)` 当 z 很大时会溢出（如 z=1000 → exp(1000) = inf）
- `log(0)` 当概率接近 0 时会出现 -inf

### 3.2 解决方案：LogSumExp 技巧

**数学技巧**：

```
log(∑ exp(z_i)) = log(∑ exp(z_i - z_max) * exp(z_max))
                = z_max + log(∑ exp(z_i - z_max))
```

其中 `z_max = max(z)`，使得 `exp(z_i - z_max)` 在 [0, 1] 范围内。

**稳定实现**：

```python
def stable_cross_entropy(logits, target):
    # logits: (N, C), target: (N,)
    max_logits = logits.max(dim=-1, keepdim=True)[0]
    logits_shifted = logits - max_logits
    
    log_sum_exp = max_logits.squeeze() + torch.log(
        torch.exp(logits_shifted).sum(dim=-1)
    )
    
    # 取出正确类别的 logit
    correct_logits = logits[range(len(target)), target]
    
    loss = -correct_logits + log_sum_exp
    return loss.mean()
```

**PyTorch 内置**：

```python
# PyTorch 的 F.cross_entropy 已经做了数值稳定优化
loss = F.cross_entropy(logits, target)
```

---

## 四、梯度推导

### 4.1 Softmax + CE 的梯度

**前向传播**：

```
p_i = exp(z_i) / ∑ⱼ exp(z_j)
L = -log(p_y)  （y 是真实类别）
```

**反向传播**：

```
∂L/∂z_i = p_i - 1{i=y}
```

其中 `1{i=y}` 是指示函数（当 i=y 时为 1，否则为 0）。

**推导过程**：

```
∂L/∂z_i = ∂L/∂p_j * ∂p_j/∂z_i  （链式法则，对 j 求和）

∂L/∂p_j = -1/p_y * 1{j=y}  （只有 j=y 时非零）

∂p_j/∂z_i = p_j(1{i=j} - p_i)  （softmax 求导）

代入得：
∂L/∂z_i = -1/p_y * p_y * (1{i=y} - p_i)
        = -(1{i=y} - p_i)
        = p_i - 1{i=y}
```

**直观理解**：
- 对于正确类别：梯度 = `p_y - 1`（预测概率 - 1）
- 对于错误类别：梯度 = `p_i - 0`（预测概率 - 0）

梯度 = 预测概率 - 真实概率（one-hot），非常优雅！

### 4.2 代码验证

```python
import torch
import torch.nn.functional as F

# 构造数据
logits = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
target = torch.tensor([0])  # 真实类别是 0

# 前向
loss = F.cross_entropy(logits, target)
loss.backward()

# 手动计算梯度
probs = F.softmax(logits, dim=-1)
manual_grad = probs.clone()
manual_grad[0, 0] -= 1  # 减去 one-hot

print("PyTorch 梯度:", logits.grad)
print("手动计算梯度:", manual_grad)
# 输出相同！
```

---

## 五、PyTorch API 详解

### 5.1 `F.cross_entropy`

**函数签名**：

```python
torch.nn.functional.cross_entropy(
    input,           # (N, C) 或 (N, C, d1, d2, ...) logits（未 softmax）
    target,          # (N,) 或 (N, d1, d2, ...) 类别索引
    weight=None,     # (C,) 每个类别的权重
    size_average=None,  # 已弃用
    ignore_index=-100,  # 忽略的类别索引（计算时跳过）
    reduce=None,     # 已弃用
    reduction='mean' # 'none' | 'mean' | 'sum'
)
```

**参数说明**：

- **input**：模型输出的 logits（**未经 softmax**）
- **target**：真实类别的整数索引
- **weight**：类别权重，用于处理类别不平衡
- **ignore_index**：忽略某些样本（如 padding）
- **reduction**：
  - `'none'`：返回每个样本的损失
  - `'mean'`：返回平均损失
  - `'sum'`：返回总损失

**示例**：

```python
import torch
import torch.nn.functional as F

# 3 个样本，4 个类别
logits = torch.randn(3, 4)
target = torch.tensor([0, 2, 1])

# 基本用法
loss = F.cross_entropy(logits, target)
print(f"平均损失: {loss.item()}")

# 不做平均
losses = F.cross_entropy(logits, target, reduction='none')
print(f"每个样本损失: {losses}")

# 类别权重（类别 0 权重 0.5，其他为 1.0）
weight = torch.tensor([0.5, 1.0, 1.0, 1.0])
loss_weighted = F.cross_entropy(logits, target, weight=weight)
print(f"加权损失: {loss_weighted.item()}")

# 忽略某个类别（如 padding）
target_with_pad = torch.tensor([0, -100, 1])  # -100 会被忽略
loss_ignore = F.cross_entropy(logits, target_with_pad, ignore_index=-100)
```

### 5.2 `nn.CrossEntropyLoss`

**类定义**（等价于函数式 API）：

```python
criterion = torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean',
    label_smoothing=0.0  # 标签平滑参数（PyTorch 1.10+）
)

loss = criterion(input, target)
```

**Label Smoothing**（标签平滑）：

```python
# 原始 one-hot: [0, 0, 1, 0] (假设类别 2 是正确的)
# 平滑后: [ε/3, ε/3, 1-ε+ε/3, ε/3]  （ε=0.1, C=4）

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, target)
```

**作用**：
- 防止模型过度自信（overconfident）
- 正则化效果，提升泛化能力
- 常用于 ImageNet 训练

### 5.3 二分类专用 API

```python
# BCELoss：输入已经过 sigmoid
bce_loss = nn.BCELoss()
loss = bce_loss(sigmoid(logits), target.float())

# BCEWithLogitsLoss：输入是 logits（推荐，数值稳定）
bce_logits_loss = nn.BCEWithLogitsLoss()
loss = bce_logits_loss(logits, target.float())
```

---

## 六、常见应用场景

### 6.1 图像分类

```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = ...  # ResNet, ViT, etc.
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)  # 不要加 softmax！
        return logits

model = ImageClassifier(num_classes=1000)
criterion = nn.CrossEntropyLoss()

# 训练循环
for images, labels in dataloader:
    logits = model(images)  # (N, 1000)
    loss = criterion(logits, labels)  # labels: (N,) 整数 0-999
    
    loss.backward()
    optimizer.step()
```

### 6.2 语义分割（像素级分类）

```python
class SegmentationModel(nn.Module):
    def forward(self, x):
        # x: (B, 3, H, W)
        logits = self.unet(x)  # (B, num_classes, H, W)
        return logits

model = SegmentationModel()
criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 通常是 void 类

# 训练
for images, masks in dataloader:
    logits = model(images)  # (B, C, H, W)
    # masks: (B, H, W) 每个像素是类别索引
    loss = criterion(logits, masks)
```

**注意**：`F.cross_entropy` 支持高维输入，自动展平后计算。

### 6.3 NLP 序列标注

```python
class SequenceTagger(nn.Module):
    def forward(self, x):
        # x: (B, seq_len, vocab_size)
        logits = self.lstm(x)  # (B, seq_len, num_tags)
        return logits

model = SequenceTagger()
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 是 <pad> 标记

# 训练
for sentences, tags in dataloader:
    logits = model(sentences)  # (B, L, C)
    # tags: (B, L)
    
    # 方法1：展平后计算
    loss = criterion(
        logits.view(-1, num_tags),  # (B*L, C)
        tags.view(-1)               # (B*L,)
    )
    
    # 方法2：直接传入（PyTorch 会自动处理）
    loss = criterion(logits.transpose(1, 2), tags)  # 需要 (B, C, L) 格式
```

### 6.4 3D 检测中的分类头

```python
# Qwen-Drive-1.0 风格
class DetectionHead(nn.Module):
    def forward(self, query_features):
        cls_logits = self.cls_branch(query_features)  # (B, num_queries, num_classes)
        return cls_logits

# 训练时使用 Focal Loss（CrossEntropy 的变体）
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# 或普通 CE（配合 top-k + sigmoid）
cls_scores = cls_logits.sigmoid()  # 多标签，每个 query 可能属于多个类
```

---

## 七、处理类别不平衡

### 7.1 类别权重

```python
# 统计训练集类别分布
class_counts = torch.tensor([1000, 500, 100, 50])  # 4 个类别
total = class_counts.sum()

# 方法1：逆频率权重
weights = total / class_counts
weights = weights / weights.sum() * len(weights)  # 归一化

# 方法2：有效样本数加权（Effective Number）
beta = 0.9999
weights = (1 - beta) / (1 - beta ** class_counts)

criterion = nn.CrossEntropyLoss(weight=weights)
```

### 7.2 Focal Loss（更适合极端不平衡）

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, target)
```

**Focal Loss 优势**：
- 自动降低易分类样本的权重
- 聚焦于难分类样本
- 目标检测（如 RetinaNet）中广泛使用

### 7.3 过采样/欠采样

```python
from torch.utils.data import WeightedRandomSampler

# 计算每个样本的采样权重
class_counts = torch.bincount(train_labels)
sample_weights = 1.0 / class_counts[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## 八、常见错误与调试

### 8.1 错误 1：对 softmax 后的输出再用 CE

```python
# ❌ 错误
logits = model(x)
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, target)  # probs 不是 logits！

# ✅ 正确
logits = model(x)
loss = F.cross_entropy(logits, target)  # 直接用 logits
```

### 8.2 错误 2：target 维度不匹配

```python
# ❌ 错误
logits = model(x)  # (B, C)
target = F.one_hot(labels, num_classes=C)  # (B, C) one-hot
loss = F.cross_entropy(logits, target)  # 报错！

# ✅ 正确
logits = model(x)  # (B, C)
target = labels    # (B,) 整数索引
loss = F.cross_entropy(logits, target)
```

### 8.3 错误 3：多标签分类用 CE

```python
# 多标签问题（一个样本可属于多个类）
# ❌ 错误：用 CrossEntropyLoss
loss = F.cross_entropy(logits, multi_labels)  # multi_labels 是多标签

# ✅ 正确：用 BCEWithLogitsLoss
loss = F.binary_cross_entropy_with_logits(logits, multi_labels.float())
```

### 8.4 调试技巧

```python
# 检查 loss 是否为 nan/inf
if torch.isnan(loss) or torch.isinf(loss):
    print("Loss is nan or inf!")
    print(f"Logits: {logits}")
    print(f"Target: {target}")
    
# 检查 logits 数值范围
print(f"Logits min/max: {logits.min()}/{logits.max()}")

# 检查 target 是否在合法范围
assert target.min() >= 0 and target.max() < num_classes

# 打印每个样本的损失
losses = F.cross_entropy(logits, target, reduction='none')
print(f"Per-sample loss: {losses}")
```

---

## 九、与其他损失函数的对比

| 损失函数 | 适用场景 | 公式 | 特点 |
|---------|---------|------|------|
| **CrossEntropyLoss** | 多分类（互斥类别） | `-log(p_y)` | 标准选择，数值稳定 |
| **NLLLoss** | 已计算 log-softmax | `-log(p_y)` | CE 的底层实现 |
| **BCELoss** | 二分类 | `-[y log(p) + (1-y)log(1-p)]` | 需要 sigmoid |
| **BCEWithLogitsLoss** | 二分类（多标签） | BCE + sigmoid | 数值稳定版本 |
| **FocalLoss** | 类别不平衡 | `-(1-p)^γ log(p)` | 聚焦难样本 |
| **MSELoss** | 回归 | `(y - ŷ)²` | 不适合分类 |
| **KLDivLoss** | 分布匹配 | `∑ p log(p/q)` | 知识蒸馏 |

---

## 十、实战技巧

### 10.1 训练稳定性

```python
# 1. 使用 logits 而非 probabilities
logits = model(x)
loss = F.cross_entropy(logits, target)  # ✅

# 2. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 学习率预热
scheduler = torch.optim.lr_scheduler.OneCycleLR(...)

# 4. Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 10.2 性能优化

```python
# 1. 使用 AMP（自动混合精度）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(x)
    loss = F.cross_entropy(logits, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. 高维输入自动展平（避免手动 reshape）
# logits: (B, C, H, W), target: (B, H, W)
loss = F.cross_entropy(logits, target)  # PyTorch 自动处理

# 3. ignore_index 优化（跳过 padding）
loss = F.cross_entropy(logits, target, ignore_index=pad_idx)
```

### 10.3 知识蒸馏

```python
# 教师模型输出（soft labels）
teacher_logits = teacher_model(x)
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

# 学生模型输出
student_logits = student_model(x)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

# 蒸馏损失（KL 散度）
distill_loss = F.kl_div(
    student_log_probs, 
    teacher_probs, 
    reduction='batchmean'
) * (temperature ** 2)

# 硬标签损失（原始 CE）
ce_loss = F.cross_entropy(student_logits, target)

# 总损失
loss = alpha * distill_loss + (1 - alpha) * ce_loss
```

---

## 十一、总结

### 核心要点

1. ✅ **直接用 logits**：`F.cross_entropy(logits, target)`，不要手动 softmax
2. ✅ **target 是整数索引**：`(N,)` 形状，不是 one-hot
3. ✅ **数值稳定性**：PyTorch 内部使用 LogSumExp 技巧
4. ✅ **梯度简洁**：`∂L/∂z = p - y_onehot`
5. ✅ **类别不平衡**：使用 `weight` 参数或 Focal Loss
6. ✅ **Label Smoothing**：防止过拟合，提升泛化

### 快速参考

```python
# 基本用法
loss = F.cross_entropy(logits, target)

# 类别权重
loss = F.cross_entropy(logits, target, weight=class_weights)

# 忽略某些样本
loss = F.cross_entropy(logits, target, ignore_index=-100)

# 标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, target)

# 高维输入（如分割）
loss = F.cross_entropy(logits, target)  # logits: (B,C,H,W), target: (B,H,W)

# 二分类
loss = F.binary_cross_entropy_with_logits(logits, target.float())
```

---

## 参考资源

- [PyTorch 官方文档：nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Deep Learning Book - Chapter 3.13](http://www.deeplearningbook.org/)
- [Stanford CS231n - Loss Functions](http://cs231n.stanford.edu/)
- [Focal Loss 论文](https://arxiv.org/abs/1708.02002)

---

*最后更新：2026 年 9 月*
