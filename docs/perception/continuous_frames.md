# 连续帧（Continuous Frames）与障碍物检测标注

## 1. 什么是连续帧

### 1.1 基本概念

**连续帧（Continuous Frames / Sequential Frames）** 指传感器在时间上**连续采集**的一系列数据帧，帧与帧之间存在**时序关联**。

在自动驾驶感知中，连续帧通常指：
- **相机视频序列**：多相机以固定帧率（如 10-30 FPS）采集的图像序列
- **LiDAR 点云序列**：激光雷达连续扫描的点云帧
- **多模态时序数据**：图像 + 点云 + 毫米波雷达的同步序列

```
时间轴 →
帧 t-2      帧 t-1       帧 t        帧 t+1      帧 t+2
 │           │           │           │           │
[图像]      [图像]      [图像]      [图像]      [图像]
[点云]      [点云]      [点云]      [点云]      [点云]
 │           │           │           │           │
 └───────────┴───────────┴───────────┴───────────┘
         同一物体在不同帧中的连续运动
```

### 1.2 关键特征

| 特征 | 说明 |
|------|------|
| **时间连续性** | 帧间时间间隔固定（如 100ms @ 10FPS） |
| **空间关联性** | 同一物体在相邻帧中位置渐变 |
| **自车运动** | 相机/LiDAR 随车辆移动，坐标系变化 |
| **物体运动** | 动态障碍物有自己的运动轨迹 |

### 1.3 单帧 vs 连续帧

```
单帧（Single Frame）：
┌─────────────┐
│  帧 t        │  → 只有当前时刻的快照
│  [检测结果]  │  → 无法判断运动状态
└─────────────┘

连续帧（Continuous Frames）：
┌─────┐ ┌─────┐ ┌─────┐
│帧t-1│→│帧 t │→│帧t+1│  → 完整运动信息
│     │ │     │ │     │  → 速度、加速度、轨迹
└─────┘ └─────┘ └─────┘
```

---

## 2. 为什么障碍物检测需要连续帧标注

### 2.1 核心动机总览

障碍物检测（尤其是自动驾驶）需要连续帧标注，主要因为**单帧信息不足以支撑安全驾驶决策**。连续帧提供了以下单帧无法获得的关键信息：

1. **运动状态估计**（速度、加速度）
2. **物体跟踪与身份关联**（Tracking ID）
3. **遮挡与重现处理**
4. **轨迹预测的监督信号**
5. **时序模型的训练需求**
6. **标注一致性与质量提升**

下面逐一展开。

---

### 2.2 运动状态估计（速度/加速度）

**核心问题**：速度标注本身需要连续帧才能生成。

#### 标注生成阶段的必要性

**单帧的根本限制**：
```python
# 标注员面对单帧图像
frame_t = show_image(image_t)
# 可见：车辆的位置、尺寸、朝向
# ❌ 无法判断：这辆车是静止的还是以 60km/h 行驶？
```

**连续帧是速度标注的唯一来源**：
```python
# 标注系统通过连续帧计算速度真值
pos_t = annotate_box(frame_t)['position']       # (x_t, y_t)
pos_t_prev = annotate_box(frame_t_prev)['position']  # (x_{t-1}, y_{t-1})
dt = 0.1  # 100ms

velocity = (pos_t - pos_t_prev) / dt
# ✅ 得到速度真值标签 (vx, vy)
```

```
帧 t-1        帧 t
车辆位置A ──→ 车辆位置B
     │          │
     └──────────┘
     位移 / Δt = 速度标注值
```

**关键区分**：
- ✅ **有了速度标注后**，模型可以从单帧学习"什么样的物体应该有什么速度"
- ⚠️ **但标注员生成速度标签时**，必须看连续帧才能测量运动
- 📌 **结论**：连续帧是产生速度标注的**前置条件**，而非模型训练的必需品

#### 为什么速度标注至关重要

- **碰撞时间估计（TTC）**：需要相对速度
- **路径规划**：预测物体未来位置
- **安全决策**：判断是否需要刹车/避让

#### 实际数据集的做法

- **nuScenes**：标注 2Hz 关键帧，速度通过相邻关键帧差分计算
- **Waymo**：10Hz 全标注，每帧速度同样依赖前后帧位置
- **Qwen-Drive-1.0**：3D 框输出 `[x, y, z, w, l, h, yaw, vx, vy]`，训练时的速度真值来自连续帧标注

---

### 2.3 物体跟踪与身份关联（Tracking）

**问题**：单帧检测无法回答"这一帧的车 A 是不是上一帧的车 A"。

**多目标跟踪（MOT）的核心**：为每个物体分配持续的 **Tracking ID**。

```
帧 t-1:  [车#1] [车#2] [行人#3]
           ↓      ↓       ↓      ← 需要连续帧标注来建立关联
帧 t:    [车#1] [车#2] [行人#3]  (相同 ID)
```

**连续帧标注提供**：
- **身份一致性**：同一物体跨帧保持相同 ID
- **数据关联真值**：训练跟踪算法的监督信号

**应用场景**：
- **行为分析**：跟踪车辆是否变道、加塞
- **意图预测**：基于历史轨迹预测未来动作
- **多帧融合**：累积多帧观测提升检测稳定性

**标注方式**：
```json
{
  "frame_t-1": {
    "objects": [
      {"id": "vehicle_001", "box": [...], "type": "car"},
      {"id": "pedestrian_002", "box": [...], "type": "pedestrian"}
    ]
  },
  "frame_t": {
    "objects": [
      {"id": "vehicle_001", "box": [...], "type": "car"},      // 相同 ID
      {"id": "pedestrian_002", "box": [...], "type": "pedestrian"}
    ]
  }
}
```

#### 模型训练：如何使用连续帧学习跟踪

跟踪模型训练有两种主流范式：

##### 1. 联合检测+跟踪（Joint Detection and Tracking）

**代表方法**：CenterTrack、QDTrack、MUTR3D

**训练输入**：
```python
# 多帧作为输入
input = {
    'image_t': current_frame_images,      # 当前帧图像
    'image_t-1': previous_frame_images,   # 历史帧图像
    'heatmap_t-1': previous_detections,   # 历史帧检测结果（可选）
}
```

**网络输出**：
```python
output = {
    'detection': [...],           # 当前帧的检测框
    'tracking_offset': [...],     # 从当前帧指向历史帧的位移向量
    'tracking_id_embedding': [...] # 用于关联的特征向量
}
```

- `tracking_id_embedding`：可以理解为每个检测到的物体会被网络提取一个高维特征向量，如果两个帧中的物体是同一个，它们的 embedding 应该非常相似。

**监督信号（来自连续帧标注）**：
```python
# 1. 检测损失（标准的 3D 框回归）
loss_det = compute_detection_loss(pred_boxes, gt_boxes)

# 2. 跟踪偏移损失（关键！）
# 对于帧 t 中的物体 A，如果它在帧 t-1 也存在（通过 Tracking ID 确认）
# 则监督网络学习预测从当前位置指向历史位置的偏移
# 网络检测到的障碍物通过一系列的方法完成与真值障碍物的匹配绑定，然后检测结果与标注真值就可以计算 Loss。
if object_exists_in_previous_frame(tracking_id):
    offset_gt = position_t_minus_1 - position_t  # 真值偏移
    offset_pred = model.tracking_offset[object_idx] # 网络直接预测出偏移量，而不是本帧与上一帧预测结果的差值
    loss_offset = L1Loss(offset_pred, offset_gt)

# 3. ID 分类损失（可选，某些方法使用）
loss_id = CrossEntropyLoss(pred_id_embedding, gt_tracking_id) # 伪代码，真实两者并不能直接计算交叉熵

total_loss = loss_det + lambda_offset * loss_offset + lambda_id * loss_id
```

**推理时如何关联**：
```python
# 当前帧检测到物体 A 在位置 (x_t, y_t)
detection_t = {'position': (x_t, y_t), 'offset': (dx, dy)}

# 预测的偏移指向历史位置
predicted_prev_position = (x_t + dx, y_t + dy)

# 在历史帧的检测中找最接近的物体
for obj_prev in detections_t_minus_1:
    if distance(obj_prev.position, predicted_prev_position) < threshold:
        # 匹配成功，继承历史 ID
        tracking_id = obj_prev.tracking_id
```

**关键点**：
- ✅ 模型**显式学习帧间关联**（通过 offset 回归）
- ✅ 训练时**必须输入连续帧**
- ✅ 监督信号**完全依赖 Tracking ID 标注**

##### 2. 两阶段方法（Detection + Data Association）

**代表方法**：DeepSORT、ByteTrack、StrongSORT

**训练流程**：
```python
# 阶段 1：单帧检测器训练（标准检测任务）
detector = train_detector(single_frame_data)

# 阶段 2：特征提取器训练（Re-ID 任务）
# 输入：连续帧中同一物体的裁剪图像块
for tracking_id in all_tracking_ids:
    crops = []
    for frame in continuous_frames:
        if tracking_id in frame:
            crops.append(crop_object(frame, tracking_id))
    
    # 训练 Re-ID 网络：同一 ID 的特征应该接近
    embeddings = reid_network(crops)
    loss = triplet_loss(embeddings, tracking_id)  # 或 contrastive loss
```

**监督信号**：
```python
# Triplet Loss 示例
anchor = embedding[object_in_frame_t]       # 锚点（当前帧）
positive = embedding[same_id_in_frame_t-1]  # 正样本（历史帧，同 ID）
negative = embedding[diff_id_in_frame_t-1]  # 负样本（历史帧，不同 ID）

loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)
```

**推理时如何关联**：
```python
# 检测当前帧
detections_t = detector(frame_t)

# 提取每个检测的特征
for det in detections_t:
    embedding = reid_network(det.crop)
    
    # 与历史轨迹特征匹配
    best_match = find_most_similar(embedding, historical_tracks)
    if similarity > threshold:
        det.tracking_id = best_match.id
    else:
        det.tracking_id = new_id()  # 新物体
```

**关键点**：
- ✅ Re-ID 网络训练**必须使用连续帧中的同 ID 样本对**
- ✅ 没有 Tracking ID 标注就无法构建训练样本对
- ⚠️ 检测器本身可以单帧训练，但关联模块必须用连续帧

##### 3. 实际案例对比

| 方法 | 范式 | 输入帧数 | 关键损失 | 连续帧依赖 |
|------|------|---------|---------|-----------|
| **CenterTrack** | Joint | 2 帧 | offset L1 | 强（训练+推理都需要） |
| **QDTrack** | Joint | 2 帧 | offset + contrastive | 强 |
| **MUTR3D** | Joint | 2 帧 | query matching | 强 |
| **DeepSORT** | Two-stage | 单帧检测器 | triplet loss（Re-ID） | 中（仅 Re-ID 训练需要） |
| **ByteTrack** | Two-stage | 单帧检测器 | 无额外训练 | 弱（仅推理时用 IoU 关联） |

##### 4. 连续帧标注的关键作用总结

```
连续帧 Tracking ID 标注
         ↓
┌────────┴────────┐
│                 │
▼                 ▼
构建训练样本对     计算监督信号
（同 ID vs 异 ID） （offset 真值）
│                 │
└────────┬────────┘
         ↓
    训练跟踪模型
  （学习帧间关联）
         ↓
  推理时自动分配 ID
```

**没有连续帧 Tracking ID 标注，跟踪模型无法训练**，因为：
- ❌ 无法知道哪些检测框应该匹配
- ❌ 无法计算 offset 真值
- ❌ 无法构建 Re-ID 的正负样本对

---

### 2.4 遮挡与重现处理

**问题**：物体可能被暂时遮挡，单帧检测会"丢失"它。

**场景示例**：
```
帧 t-1:  [车A]  [车B]           ← 都可见
              ↓
帧 t:    [车A] (车B被卡车遮挡)   ← 车B消失
              ↓
帧 t+1:  [车A]  [车B]           ← 车B重新出现
```

**单帧的问题**：
- 帧 t 检测不到车 B → 误认为车 B 消失
- 帧 t+1 重新检测到 → 误认为是"新"物体

**连续帧标注的价值**：
- **遮挡标记**：标注物体在某帧被遮挡但仍存在
- **轨迹补全**：即使不可见也维持轨迹连续性
- **重识别（Re-ID）**：帧 t+1 的车 B 关联回帧 t-1 的车 B

**标注属性**：
```json
{
  "id": "vehicle_B",
  "visibility": "occluded",  // fully_visible | partially_occluded | occluded
  "box": [...],               // 即使遮挡也标注估计位置
  "interpolated": true        // 是否为插值标注
}
```

---

### 2.5 轨迹预测的监督信号

**问题**：自动驾驶需要预测障碍物**未来的运动轨迹**，这本质上需要连续帧。

**轨迹预测任务**：
```
输入：历史轨迹（过去 2 秒，连续帧）
     [位置_{t-20}, ..., 位置_{t-1}, 位置_t]
输出：未来轨迹（未来 3-6 秒）
     [位置_{t+1}, ..., 位置_{t+60}]
```

**连续帧标注是唯一来源**：
- 历史轨迹 = 过去连续帧的位置序列
- 未来轨迹真值 = 未来连续帧的位置序列（训练时使用）

**应用**：
- **运动规划**：避开障碍物的预测路径
- **交互建模**：预测车辆间的博弈（如汇入车流）
- **风险评估**：识别潜在碰撞

```
      预测轨迹
         ╱
  车辆 ──●─ ─ ─ ─→ ?
     历史  未来
 (连续帧)  (需预测)
```

---

### 2.6 时序模型的训练需求

**问题**：现代感知模型（如 BEVFormer）显式利用时序信息，训练时必须有连续帧。

**注意**：这里说的时序模型训练并不是指真值标注数据（历史帧）进入模型，而是历史传感器数（如：图像、位姿）据会进入模型，历史真实标注数据只会参与 Loss 计算，不进入模型。

**BEVFormer 的时间自注意力（TSA）**：
```python
# BEVFormer 融合历史 BEV 特征
def temporal_self_attention(bev_query_t, bev_feature_t_minus_1, ego_motion):
    # 需要历史帧的 BEV 特征
    bev_aligned = align_by_ego_motion(bev_feature_t_minus_1, ego_motion)
    output = deformable_attn(bev_query_t, bev_aligned)
    return output
```

**训练需求**：
- **多帧输入**：模型输入连续 2-4 帧
- **帧间标注一致**：每帧都需要标注，且物体 ID 关联
- **自车运动信息**：帧间的 ego pose 变化（用于坐标对齐）

**性能提升证据**（BEVFormer）：

| 版本 | mAP | NDS | mAVE（速度误差）|
|------|-----|-----|-----------------|
| BEVFormer-S（单帧） | 37.5 | 44.8 | 0.788 |
| BEVFormer（时序） | **41.6** | **51.7** | **0.394** |

- 时序版速度误差**降低 50%**（0.788 → 0.394）
- 整体性能显著提升，**证明连续帧的价值**

> **注意**：Qwen-Drive-1.0 虽然支持 TSA，但推理时采用单帧模式（TSA 退化）。
> 然而其训练仍可受益于连续帧标注，尤其是速度估计维度。
> 详见 [bevformer_qwen_drive_1_0.md](./bevformer_qwen_drive_1_0.md)。

---

### 2.7 标注一致性与质量提升

**连续帧标注还能提升标注质量本身**：

1. **时序插值**：
   - 标注关键帧（如每 5 帧标注 1 帧）
   - 中间帧通过插值生成，减少人工成本

```python
# 线性插值示例
box_t = interpolate(box_keyframe_1, box_keyframe_2, ratio=0.5)
```

2. **一致性校验**：
   - 利用运动连续性检查标注错误
   - 如果物体位置突变（不符合物理运动），标记为可疑

```python
# 检查运动合理性
velocity = (pos_t - pos_t_prev) / dt
if velocity > MAX_REASONABLE_SPEED:
    flag_annotation_error(frame_t, object_id)
```

3. **遮挡推断**：
   - 前后帧可见时，中间遮挡帧可推断位置

4. **减少漏标/误标**：
   - 前后帧都有的物体，当前帧漏标会被发现

---

## 3. 连续帧标注的关键要素

### 3.1 标注内容

| 要素 | 说明 | 示例 |
|------|------|------|
| **3D 边界框** | 每帧的位置/尺寸/朝向 | `[x, y, z, w, l, h, yaw]` |
| **Tracking ID** | 跨帧身份关联 | `vehicle_001` |
| **速度** | 通过帧间差分或直接标注 | `[vx, vy]` |
| **可见性** | 遮挡状态 | `visible / occluded` |
| **类别** | 物体类型 | `car / pedestrian / cyclist` |
| **时间戳** | 帧的采集时间 | `1234567890.123` |
| **自车位姿** | ego pose（用于坐标对齐） | `[translation, rotation]` |

### 3.2 坐标系对齐

**关键挑战**：自车在运动，不同帧的坐标系不同。

```python
# 将历史帧的物体位置转换到当前帧坐标系
def align_to_current_frame(pos_history, ego_pose_history, ego_pose_current):
    # 全局坐标系
    pos_global = ego_to_global(pos_history, ego_pose_history)
    # 当前帧坐标系
    pos_current = global_to_ego(pos_global, ego_pose_current)
    return pos_current
```

**标注要求**：
- 记录每帧的 **ego pose**（自车位姿）
- 支持坐标系间的相互转换

### 3.3 标注频率

| 策略 | 说明 | 权衡 |
|------|------|------|
| **全帧标注** | 每帧都人工标注 | 精确但成本高 |
| **关键帧 + 插值** | 关键帧标注 + 中间插值 | 平衡成本与质量 |
| **自动标注 + 人工校验** | 模型预标注 + 人工修正 | 高效，主流方案 |

---

## 4. 实际数据集示例

### 4.1 nuScenes

- **帧率**：关键帧 2Hz（用于标注），传感器 10-20Hz
- **连续性**：每个场景 20 秒连续序列
- **标注**：3D 框 + Tracking ID + 速度 + 可见性
- **速度真值**：通过连续关键帧位置差分计算

```
场景（20秒）
│
├─ 关键帧 @ 2Hz（40 帧标注）
│   ├─ 每帧：3D 框 + ID + 速度
│   └─ 跨帧：ID 关联
│
└─ 中间帧（传感器数据，用于时序模型输入）
```

### 4.2 Waymo Open Dataset

- **帧率**：10Hz 全标注
- **连续性**：20 秒序列（200 帧）
- **特点**：每帧全标注，Tracking ID 完整
- **优势**：高帧率支持精细运动分析

### 4.3 标注格式对比

| 数据集 | 标注帧率 | 序列长度 | Tracking | 速度标注 |
|--------|---------|---------|----------|---------|
| **nuScenes** | 2Hz | 20s | ✅ | ✅（差分） |
| **Waymo** | 10Hz | 20s | ✅ | ✅ |
| **KITTI** | 10Hz | 变长 | ✅（tracking 子集） | ⚠️ 部分 |
| **OpenScene/nuPlan** | 10Hz | 长序列 | ✅ | ✅ |

---

## 5. 连续帧带来的技术能力

### 5.1 能力对比

| 能力 | 单帧 | 连续帧 |
|------|------|--------|
| **物体检测** | ✅ | ✅ |
| **速度估计** | ❌ | ✅ |
| **多目标跟踪** | ❌ | ✅ |
| **轨迹预测** | ❌ | ✅ |
| **遮挡处理** | ⚠️ 弱 | ✅ |
| **运动分类**（静止/移动） | ❌ | ✅ |
| **行为理解**（变道/转弯） | ❌ | ✅ |
| **时序模型训练** | ❌ | ✅ |

### 5.2 感知-预测-规划链路

```
连续帧标注
    │
    ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  感知      │──→│  预测      │──→│  规划      │
│ (检测+跟踪)│   │ (轨迹预测) │   │ (路径规划) │
└───────────┘   └───────────┘   └───────────┘
    │               │               │
需要连续帧      需要历史轨迹      需要未来预测
（速度/ID）    （连续帧序列）    （基于预测）
```

**结论**：整个自动驾驶链路都依赖连续帧标注。

---

## 6. 总结

### 核心要点

1. ✅ **连续帧 = 时序关联的数据序列**，帧间有运动连续性
2. ✅ **速度估计**：单帧无法测速，连续帧通过位置差分获得
3. ✅ **物体跟踪**：Tracking ID 建立跨帧身份关联
4. ✅ **遮挡处理**：维持被遮挡物体的轨迹连续性
5. ✅ **轨迹预测**：历史+未来轨迹的唯一数据来源
6. ✅ **时序模型训练**：BEVFormer 等模型需要连续帧输入
7. ✅ **标注质量提升**：插值、一致性校验、遮挡推断

### 为什么障碍物检测需要连续帧标注（一句话总结）

> **安全驾驶决策不仅需要知道障碍物"在哪里"（单帧可得），更需要知道它"往哪去、有多快、下一步做什么"（必须连续帧）。**

### 关键数据支撑

- BEVFormer 时序版相比单帧版，**速度误差降低 50%**（mAVE: 0.788 → 0.394）
- nuScenes、Waymo 等主流数据集均采用连续帧标注 + Tracking ID
- 感知-预测-规划全链路都依赖连续帧信息

---

## 参考资源

- [nuScenes 数据集](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)
- **相关文档**：
  - [bevformer.md](./bevformer.md)（时序自注意力原理）
  - [bevformer_qwen_drive_1_0.md](./bevformer_qwen_drive_1_0.md)（Qwen-Drive 时序实践）
  - [detr3d_qwen_drive_1_0.md](./detr3d_qwen_drive_1_0.md)（速度维度的框回归）

---

