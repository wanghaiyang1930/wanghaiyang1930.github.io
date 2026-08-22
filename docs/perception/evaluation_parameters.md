<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-28 -->

# 感知评估参数详解

在自动驾驶感知系统中，准确评估模型性能至关重要。本文档详细介绍常用的评估指标及其计算方式。

## 1. 混淆矩阵基础

在介绍具体指标前，需要理解混淆矩阵的四个基本概念：

- **TP (True Positive)**：真正例 - 正确预测为正样本的数量
- **FP (False Positive)**：假正例 - 错误预测为正样本的数量（误检）
- **TN (True Negative)**：真负例 - 正确预测为负样本的数量
- **FN (False Negative)**：假负例 - 错误预测为负样本的数量（漏检）

|  | 预测为正 | 预测为负 |
|---|---|---|
| **实际为正** | TP | FN |
| **实际为负** | FP | TN |

## 2. Precision（精确率/查准率）

### 含义

精确率表示在所有预测为正样本的结果中，真正为正样本的比例。衡量模型预测的准确性。

**物理意义**：在自动驾驶中，表示检测到的目标中有多少是真实存在的目标，**精确率高意味着误检少**。

### 计算公式

```
Precision = TP / (TP + FP)
```

### 应用场景

- 当误检代价高时（如误识别行人导致急刹车）
- 需要减少虚警的场景

## 3. Recall（召回率/查全率）

### 含义

召回率表示在所有真实正样本中，被正确预测出来的比例。衡量模型发现正样本的能力。

**物理意义**：在自动驾驶中，表示实际存在的目标中有多少被检测到，**召回率高意味着漏检少**。

### 计算公式

```
Recall = TP / (TP + FN)
```

### 应用场景

- 当漏检代价高时（如漏检行人导致碰撞）
- 安全关键型检测任务

## 4. F1 Score

### 含义

F1 分数是 Precision 和 Recall 的调和平均数，综合考虑两个指标，平衡误检和漏检。

### 计算公式

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

或等价形式：

```
F1 = 2TP / (2TP + FP + FN)
```

### 特点

- 取值范围：[0, 1]，越接近 1 越好
- 当 Precision 和 Recall 都很高时，F1 才会高
- 对不平衡的 Precision 和 Recall 有惩罚作用

### F-beta Score 变体

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- β > 1：更重视 Recall（如 F2 Score）
- β < 1：更重视 Precision（如 F0.5 Score）

## 5. IoU（Intersection over Union）

### 含义

交并比，用于评估预测框与真实框的重叠程度。

### 计算公式

```
IoU = Area(Prediction ∩ Ground Truth) / Area(Prediction ∪ Ground Truth)
```

### 特点

- 取值范围：[0, 1]
- IoU 超过阈值（通常 0.5 或 0.7）**且预测类别与真实类别一致**时，该预测才计为 TP
- 在目标检测中是判断 TP 的重要标准
- 若多个预测框匹配到同一个真实框，通常只有置信度最高的那个计为 TP，其余计为 FP

## 6. AP（Average Precision）

### 含义

平均精度，是 Precision-Recall 曲线下的面积，综合评估不同召回率下的精确率。

### 计算步骤

1. **按置信度排序**：将所有预测结果按置信度从高到低排序
2. **计算 P-R 点**：遍历排序后的结果，逐步计算 Precision 和 Recall。注意每个预测被计为 TP 的前提是 **IoU 超过阈值且预测类别与真实类别一致**
3. **插值平滑**：
   ```
   P_interp(r) = max(P(r̃))  其中 r̃ ≥ r
   ```

不同标准的 AP 计算方式存在差异，务必区分：

#### PASCAL VOC 2007（11 点插值）
在 recall = {0, 0.1, 0.2, ..., 1.0} 共 11 个点上取插值后精确率的平均：
```
AP = (1/11) × Σ P_interp(r)  其中 r ∈ {0, 0.1, ..., 1.0}
```

#### PASCAL VOC 2010 及之后（全点插值 / all-point interpolation）
在所有 recall 变化点上计算 P-R 曲线下面积：
```
AP = Σ(R(n) - R(n-1)) × P_interp(R(n))
```

### COCO 数据集的 AP 计算

使用 101 点插值方法（可视为全点插值的密集采样版本）：
```
AP = (1/101) × Σ P_interp(r)  其中 r ∈ {0, 0.01, 0.02, ..., 1.0}
```

## 7. mAP（mean Average Precision）

### 含义

平均精度均值，是多个类别 AP 的平均值，用于多类别目标检测任务的整体评估。

### 计算公式

```
mAP = (1/N) × Σ AP_i  其中 i ∈ {1, 2, ..., N}
```

N 为类别数量。

### 常见变体

#### mAP@0.5

IoU 阈值设为 0.5 时的 mAP（PASCAL VOC 标准）

#### mAP@0.75

IoU 阈值设为 0.75 时的 mAP（更严格的标准）

#### mAP@[0.5:0.95]

COCO 标准，IoU 阈值从 0.5 到 0.95，步长 0.05，取平均值：
```
mAP = (1/10) × Σ mAP@IoU  其中 IoU ∈ {0.5, 0.55, ..., 0.95}
```

#### mAP_s, mAP_m, mAP_l

根据目标大小分类的 mAP：
- **mAP_s**：小目标（面积 < 32²）
- **mAP_m**：中等目标（32² < 面积 < 96²）
- **mAP_l**：大目标（面积 > 96²）

## 8. NDS（nuScenes Detection Score）

### 含义

nuScenes 数据集提出的综合评估指标，特别适用于 3D 目标检测。

### 计算公式

```
NDS = (1/10) × [5×mAP + Σ(1 - min(1, mTP))]
```

其中 mTP 表示 5 个 True Positive 误差指标（mean True Positive metrics）的集合：
- **ATE**（Average Translation Error）：平均平移误差，单位为米（m）
- **ASE**（Average Scale Error）：平均尺度误差，定义为 1 - IoU（对齐朝向和中心后计算），无量纲
- **AOE**（Average Orientation Error）：平均方向误差，单位为弧度（rad）
- **AVE**（Average Velocity Error）：平均速度误差，单位为 m/s
- **AAE**（Average Attribute Error）：平均属性误差，定义为 1 - acc（属性分类准确率），无量纲

> 注意：这 5 个误差量纲各不相同，`min(1, mTP)` 中的截断是对各误差**分别**取上界为 1，使其落入可与 mAP 组合的 [0, 1] 范围。因此 `1 - min(1, mTP)` 越接近 1 表示该维度误差越小。ATE/AOE/AVE 等有量纲的误差在计算时会先按各自的物理单位截断到 1。

### 特点

- 综合考虑检测精度（mAP）和定位质量（平移/尺度/朝向/速度/属性误差）
- 更全面评估 3D 检测性能，避免仅用 mAP 忽略定位精度的问题

## 9. 其他重要指标

### 9.1 Accuracy（准确率）

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

适用于类别平衡的分类任务。

### 9.2 FPR（False Positive Rate）

```
FPR = FP / (FP + TN)
```

假正例率，衡量负样本被错误预测的比例。

### 9.3 FNR（False Negative Rate）

```
FNR = FN / (TP + FN) = 1 - Recall
```

假负例率，衡量正样本被遗漏的比例。

### 9.4 AUC（Area Under Curve）

ROC 曲线（TPR vs FPR）下的面积，取值 [0, 1]，越接近 1 越好。

## 10. 自动驾驶感知中的实际应用

### 10.1 2D 目标检测

- **主要指标**：mAP@0.5, mAP@0.75
- **关注点**：不同类别（车辆、行人、自行车等）的 AP

### 10.2 3D 目标检测

- **主要指标**：NDS, mAP@[IoU thresholds]
- **关注点**：距离分段性能（0-30m, 30-50m, 50m+）

### 10.3 语义分割

- **主要指标**：mIoU（mean Intersection over Union）
- **计算公式**：
  ```
  mIoU = (1/N) × Σ (TP_i / (TP_i + FP_i + FN_i))
  ```

### 10.4 车道线检测

- **主要指标**：F1 Score, Accuracy
- **特定指标**：车道线检出率、误检率

## 11. 指标选择建议

| 场景 | 推荐指标 | 原因 |
|---|---|---|
| 行人检测 | Recall, F2 Score | 漏检代价高，宁可误检 |
| 静态障碍物检测 | F1 Score, mAP | 平衡误检和漏检 |
| 可行驶区域分割 | mIoU, Recall | 确保覆盖所有安全区域 |
| 3D 检测综合评估 | NDS, mAP@[0.5:0.95] | 全面评估定位和分类 |

## 12. 计算示例

假设在 100 个真实行人中：
- 检测到 90 个行人（TP = 90）
- 漏检 10 个行人（FN = 10）
- 误检 5 个非行人为行人（FP = 5）

则：
```
Precision = 90 / (90 + 5) = 0.947 (94.7%)
Recall = 90 / (90 + 10) = 0.900 (90.0%)
F1 Score = 2 × 0.947 × 0.900 / (0.947 + 0.900) = 0.923 (92.3%)
```

## 参考资料

- PASCAL VOC Challenge
- MS COCO: Common Objects in Context
- nuScenes: A multimodal dataset for autonomous driving
- KITTI Vision Benchmark Suite
