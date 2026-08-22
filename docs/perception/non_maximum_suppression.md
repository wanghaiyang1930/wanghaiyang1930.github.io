# Non-Maximum Suppression (非极大值抑制)

## 概述

Non-Maximum Suppression (NMS，非极大值抑制) 是目标检测中的经典后处理算法，用于解决**一个物体被多个边界框重复检测**的问题。目标检测网络（如R-CNN系列、YOLO、SSD等）在推理时往往会为同一个物体生成多个高度重叠的候选框，NMS通过保留置信度最高的框并抑制其他冗余框，确保每个物体只输出一个最佳检测结果。

NMS是目标检测pipeline中不可或缺的一环，直接影响最终检测结果的质量和准确率。

## 问题背景

**为什么需要NMS？**

在目标检测中：
- **密集采样策略**：检测器会在图像的不同位置和尺度生成大量候选框（Anchor boxes）
- **滑动窗口机制**：相邻窗口往往会对同一物体产生多个检测响应
- **特征金字塔**：多尺度检测会在不同层级产生重叠检测
- **回归偏差**：边界框回归的微小差异导致同一物体有多个略微不同的框

**结果**：单个物体可能被检测出数十个甚至上百个重叠框，必须通过后处理筛选出最优结果。

## 核心思想

NMS基于一个简单直观的假设：
> **对于检测同一物体的多个边界框，置信度最高的那个最可能是正确的检测结果。**

因此，NMS的策略是：
1. **保留**置信度最高的检测框
2. **抑制（删除）**与该框高度重叠的其他检测框
3. 重复该过程直到处理完所有候选框

"非极大值"指的是保留极大值（最高置信度），抑制非极大值（其他重叠框）。

## 算法流程

### 标准NMS算法

**输入**：
- 检测框集合 B = {b₁, b₂, ..., bₙ}，每个框包含：
  - 坐标：(x₁, y₁, x₂, y₂) 或 (x, y, w, h)
  - 置信度分数：s
  - 类别：c
- IoU阈值：N_t（通常0.3-0.7）
- 置信度阈值：S_t（通常0.5-0.7）

**输出**：
- 筛选后的检测框集合 D

**算法步骤**：

```
1. 初始化保留框列表 D = []

2. 按置信度分数 s 对检测框集合 B 进行降序排列

3. while B 不为空:
   a. 从 B 中取出置信度最高的框 M（即当前极大值框）
   
   b. 将 M 添加到保留列表 D 中
   
   c. 从 B 中移除 M
   
   d. 遍历 B 中剩余的所有框 b_i:
      - 计算 IoU(M, b_i)
      - 如果 IoU(M, b_i) > N_t:
          从 B 中移除 b_i（抑制冗余框）
   
4. 返回 D
```

### IoU计算

**交并比 (Intersection over Union, IoU)** 是衡量两个框重叠程度的指标：

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

其中：
- A ∩ B：两个框的交集面积
- A ∪ B：两个框的并集面积
```

**计算公式**：
```python
# 计算交集区域坐标
x1 = max(boxA[0], boxB[0])
y1 = max(boxA[1], boxB[1])
x2 = min(boxA[2], boxB[2])
y2 = min(boxA[3], boxB[3])

# 交集面积
intersection = max(0, x2 - x1) * max(0, y2 - y1)

# 各自面积
areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# 并集面积
union = areaA + areaB - intersection

# IoU
iou = intersection / union
```

**IoU值含义**：
- IoU = 1：两框完全重合
- IoU = 0：两框无重叠
- IoU > 0.5：通常认为检测同一物体
- IoU > 0.7：高度重叠

## 参数设置

### IoU阈值 (N_t)

**作用**：控制抑制的严格程度

**常用范围**：0.3 - 0.7

**影响**：
- **N_t过低（如0.3）**：
  - 抑制过于激进
  - 可能删除同一类别但不同物体的检测框
  - 漏检率上升（特别是密集场景）
  
- **N_t过高（如0.9）**：
  - 抑制不足
  - 同一物体保留多个检测框
  - 产生冗余检测

- **推荐设置**：
  - 通用检测：0.5
  - 密集场景（如人群检测）：0.3-0.4
  - 稀疏场景：0.6-0.7

### 置信度阈值 (S_t)

**作用**：过滤低质量检测框

**常用范围**：0.3 - 0.7

**影响**：
- **S_t过低**：保留更多低置信度框，增加误检
- **S_t过高**：可能过滤掉真实但模糊的检测

**处理时机**：
- **方式1**：NMS前过滤 - 先删除低置信度框再执行NMS
- **方式2**：NMS后过滤 - 先执行NMS再过滤低置信度框
- **推荐**：NMS前过滤，减少计算量

## 类别处理

NMS通常**按类别独立执行**：

**原因**：
- 不同类别的物体可能在空间上重叠（如"人"在"汽车"旁边）
- 类别间的框不应相互抑制

**实现方式**：
```python
for class_id in all_classes:
    # 筛选出当前类别的所有检测框
    boxes_of_class = filter_by_class(all_boxes, class_id)
    
    # 对该类别独立执行NMS
    kept_boxes = nms(boxes_of_class, iou_threshold)
    
    # 合并结果
    final_results.extend(kept_boxes)
```

**特殊情况：类别无关NMS**
- 某些应用中需要抑制不同类别但重叠的检测
- 例如：单目标跟踪中只保留最高置信度的检测，不管类别

## 代码实现示例

### Python/NumPy实现

```python
import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression
    
    参数:
        boxes: numpy array of shape (N, 4), 格式 [x1, y1, x2, y2]
        scores: numpy array of shape (N,), 置信度分数
        iou_threshold: float, IoU阈值
    
    返回:
        keep: 保留框的索引列表
    """
    # 边界情况
    if len(boxes) == 0:
        return []
    
    # 提取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算每个框的面积
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度降序排列，获取索引
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # 取出当前最高分的框
        i = order[0]
        keep.append(i)
        
        # 计算该框与其余框的IoU
        # 交集区域的左上角和右下角坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 交集面积
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        # IoU = 交集 / 并集
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # 保留IoU小于阈值的框（即与当前框不重叠或重叠较少的框）
        inds = np.where(iou <= iou_threshold)[0]
        
        # 更新order（注意inds是相对于order[1:]的索引，需要+1）
        order = order[inds + 1]
    
    return keep
```

### 使用示例

```python
# 示例检测结果
boxes = np.array([
    [100, 100, 200, 200],  # 框1
    [105, 105, 205, 205],  # 框2: 与框1高度重叠
    [300, 300, 400, 400],  # 框3: 独立检测
    [110, 110, 210, 210],  # 框4: 与框1高度重叠
])

scores = np.array([0.9, 0.85, 0.95, 0.75])

# 执行NMS
keep = nms(boxes, scores, iou_threshold=0.5)

print(f"保留的框索引: {keep}")
# 输出: [2, 0]  (框3置信度最高被保留；框1在同组中置信度最高被保留；框2和框4被抑制)

kept_boxes = boxes[keep]
kept_scores = scores[keep]
```

## 时间复杂度

**标准NMS**：O(N²)

**分析**：
- 外层循环：遍历N个检测框
- 内层循环：每次计算剩余框与当前框的IoU
- 最坏情况：所有框互不重叠，每次都要遍历剩余的所有框

**优化方法**：
1. **预过滤**：先用置信度阈值过滤，减少N
2. **空间索引**：使用空间哈希或R树加速邻域查找
3. **GPU加速**：并行计算IoU（如CUDA实现）
4. **近似算法**：只比较空间上邻近的框

## NMS的局限性

### 1. 密集场景漏检

**问题**：当多个物体高度重叠时，NMS会错误地抑制真实检测

**场景**：
- 人群检测：重叠的人体
- 货架商品：密集堆叠的物品
- 遮挡场景：部分遮挡的车辆

**原因**：NMS假设高IoU的框检测的是同一物体，但密集场景下不同物体的框也可能高IoU

**示例**：
```
两个部分重叠的行人A和B：
- 框1（检测A）：IoU=0.8 with 框2，置信度0.9
- 框2（检测B）：与框1重叠，置信度0.85
结果：框2被误删，行人B漏检
```

### 2. 固定阈值不适应性

**问题**：单一IoU阈值无法适应所有场景

- 不同类别的最佳阈值可能不同
- 同一图像中稀疏区域和密集区域需要不同阈值

### 3. 置信度不完全可靠

**问题**：NMS完全依赖置信度排序，但：
- 网络输出的置信度可能不准确（校准问题）
- 遮挡物体的真实框置信度可能低于部分框

### 4. 边界框定位误差

**问题**：保留的最高置信度框不一定是定位最准的框

**示例**：
- 框1：置信度0.9，IoU=0.7（与GT）
- 框2：置信度0.85，IoU=0.85（与GT）
- NMS保留框1，但框2定位更准

## NMS的改进方法

### 1. Soft-NMS

**论文**：Bodla et al., "Soft-NMS -- Improving Object Detection with One Line of Code", ICCV 2017

**核心思想**：不直接删除重叠框，而是**降低其置信度**

**算法修改**：
```python
# 标准NMS: 删除IoU > threshold的框
if iou > threshold:
    remove box

# Soft-NMS: 降低置信度
if iou > threshold:
    score = score * f(iou)  # 衰减函数
```

**衰减函数**：

**线性衰减**：
```
s_i = s_i * (1 - IoU(M, b_i))  if IoU > N_t
```

**高斯衰减**（推荐）：
```
s_i = s_i * exp(-(IoU(M, b_i)²) / σ)
```

**优点**：
- 保留了潜在的真实检测
- 在密集场景中效果更好
- 几乎无额外计算开销

**缺点**：
- 仍可能保留冗余框
- 需要调整额外参数σ

### 2. DIoU-NMS / CIoU-NMS

**论文**：Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020

**核心思想**：使用**DIoU（Distance-IoU）**或**CIoU（Complete-IoU）**替代标准IoU

**DIoU**考虑：
- 框的重叠度（IoU）
- 框中心点的距离

**CIoU**额外考虑：
- 长宽比的一致性

**公式**：
```
DIoU = IoU - (ρ²(b, b_gt) / c²)

其中：
- ρ(b, b_gt): 两框中心点距离
- c: 能包含两框的最小外接框的对角线长度
```

**抑制策略**：
```
if DIoU(M, b_i) > threshold:
    remove b_i
```

**优点**：
- 考虑了框的中心距离，密集场景表现更好
- 即使IoU相同，也会优先保留中心更接近的框

### 3. Adaptive NMS

**核心思想**：根据物体密度**自适应调整IoU阈值**

**方法**：
- 预测每个框周围的物体密度
- 密集区域使用较低的IoU阈值（允许更多重叠）
- 稀疏区域使用较高的IoU阈值（严格抑制）

**实现**：需要网络额外输出密度预测头

### 4. Learning NMS

**核心思想**：将NMS过程**参数化并学习**

**方法**：
- 用神经网络学习抑制策略
- 输入：候选框特征、IoU关系图
- 输出：每个框的保留概率

**代表工作**：
- GossipNet (ECCV 2018)
- Relation Networks for Object Detection (CVPR 2018)

**优点**：
- 端到端可学习
- 可以学习到复杂的抑制规则

**缺点**：
- 计算开销大
- 需要额外训练数据和标注

### 5. End-to-End检测器（无需NMS）

**代表方法**：
- **DETR** (Detection Transformer, ECCV 2020)
- **Sparse R-CNN** (CVPR 2021)

**核心思想**：
- 使用集合预测机制（Set Prediction）
- 匈牙利匹配（Hungarian Matching）分配检测到GT
- 天然避免重复检测，无需后处理NMS

**优点**：
- 完全端到端，无需手工设计后处理
- 理论上更优雅

**缺点**：
- 训练复杂，收敛慢
- 小目标检测性能仍有待提升

## 不同检测框架中的NMS

### Faster R-CNN

- **阶段1（RPN）**：对区域候选框执行NMS，通常IoU=0.7
- **阶段2（检测头）**：对最终检测框按类别执行NMS，IoU=0.5

### YOLO系列

- **YOLOv3/v4**：
  - 使用标准NMS，IoU阈值0.45-0.5
  - 按类别独立执行
  
- **YOLOv5**：
  - 默认使用标准NMS
  - 可选Soft-NMS、DIoU-NMS

- **YOLOv7/v8**：
  - 支持多种NMS变体
  - 提供类别无关NMS选项

### SSD

- **多尺度检测**：每个尺度独立执行NMS
- **IoU阈值**：通常0.45
- **Top-K筛选**：NMS前先保留前K个高置信度框（如K=200）

### RetinaNet

- **Focal Loss训练**：已缓解类别不平衡，输出置信度更可靠
- **NMS设置**：IoU=0.5，置信度阈值=0.05

## 实际应用建议

### 参数调优策略

1. **从标准值开始**：IoU=0.5, Score=0.5
2. **观察验证集表现**：
   - 冗余检测多 → 降低IoU阈值
   - 漏检多 → 提高IoU阈值或降低Score阈值
3. **针对特定场景调整**：
   - 密集场景：IoU=0.3-0.4
   - 稀疏场景：IoU=0.6-0.7
   - 高精度需求：提高Score阈值
   - 高召回需求：降低Score阈值

### 计算优化

**预过滤**：
```python
# 先过滤低置信度框
mask = scores > score_threshold
boxes = boxes[mask]
scores = scores[mask]

# 再执行NMS
keep = nms(boxes, scores, iou_threshold)
```

**Top-K筛选**：
```python
# 只保留前K个高分框
if len(scores) > topk:
    topk_indices = np.argsort(scores)[-topk:]
    boxes = boxes[topk_indices]
    scores = scores[topk_indices]
```

### 调试技巧

**可视化**：
```python
import cv2

def visualize_nms(image, boxes_before, boxes_after, scores):
    """可视化NMS前后的检测框"""
    # 绘制NMS前的所有框（红色）
    for box in boxes_before:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 
                      (0, 0, 255), 1)
    
    # 绘制NMS后保留的框（绿色，加粗）
    for box, score in zip(boxes_after, scores):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 
                      (0, 255, 0), 2)
        cv2.putText(image, f'{score:.2f}', (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image
```

## 总结

**NMS的核心价值**：
- 简单高效的冗余检测过滤方法
- 目标检测pipeline的标准组件
- 对最终检测性能影响显著

**选择建议**：
- **通用场景**：标准NMS，IoU=0.5
- **密集场景**：Soft-NMS或DIoU-NMS
- **追求极致性能**：考虑无NMS的端到端检测器（DETR）
- **工程实践**：先用标准NMS验证baseline，再尝试改进方法

**未来方向**：
- 更智能的自适应抑制策略
- 端到端可学习的抑制机制
- 无需后处理的检测架构成为主流

## 参考资料

- Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006
- Bodla et al., "Soft-NMS -- Improving Object Detection with One Line of Code", ICCV 2017
- Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020
- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020
- Hosang et al., "Learning Non-maximum Suppression", CVPR 2017
