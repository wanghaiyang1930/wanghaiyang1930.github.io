# BEVFormer 技术要点

## 概览

**BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**

- **论文**：BEVFormer (ECCV 2022)
- **作者**：Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, Jifeng Dai (南京大学、上海AI Lab、香港大学、商汤)
- **核心创新**：通过时空Transformer从多相机图像学习统一的BEV（鸟瞰图）表示，同时利用空间和时间信息

---

## 1. 核心思想

### 1.1 设计理念

BEVFormer 构建一个显式的 **BEV 特征表示**，作为多任务感知的统一中间表示：

1. **BEV Queries**：在鸟瞰图空间定义网格状的可学习queries
2. **空间交叉注意力（Spatial Cross-Attention）**：从多相机图像聚合空间信息
3. **时间自注意力（Temporal Self-Attention）**：从历史BEV特征聚合时序信息
4. **统一BEV特征**：生成的BEV特征可同时支持3D检测和地图分割

### 1.2 与其他方法的区别

| 方法类型 | 特征表示 | 时序建模 | 代表方法 |
|---------|---------|---------|---------|
| **Query方法** | 无显式表示 | 弱 | DETR3D, PETR |
| **深度投影** | BEV（深度估计） | 无 | LSS, BEVDet |
| **BEVFormer** | **显式BEV网格** | **强（时空注意力）** | BEVFormer |

**关键优势**：
- ✅ **显式BEV表示**，可复用于多个下游任务
- ✅ **时序融合**，利用历史帧提升检测（尤其对遮挡和速度估计）
- ✅ **可变形注意力**，计算高效且聚焦关键区域
- ✅ **多任务统一**，同一BEV特征支持检测和分割

---

## 2. 架构详解

### 2.1 整体架构

```
多视图图像 (N cameras, T frames)
    ↓
图像骨干网络 (ResNet-101-DCN / VoVNet)
    ↓
多尺度特征图 (FPN)
    ↓
BEV Queries (H×W 网格，如 200×200)
    ↓
BEVFormer Encoder (6层)
    │
    ├─ Temporal Self-Attention (融合历史BEV)
    ├─ Spatial Cross-Attention (聚合多相机特征)
    └─ FFN
    ↓
统一 BEV 特征 (H×W×C)
    ↓
    ├─→ 检测头 (Deformable DETR head)
    └─→ 分割头 (Map segmentation)
```

### 2.2 核心组件

#### **A. BEV Queries**

- **形状**：`Q ∈ ℝ^(H×W×C)`，如 200×200×256
- **物理含义**：每个query对应BEV平面上的一个网格单元
- **网格分辨率**：如每格 0.512m，覆盖 [-51.2m, 51.2m]
- **位置编码**：可学习的位置嵌入

```python
bev_queries = nn.Embedding(bev_h * bev_w, embed_dims)  # 200*200, 256
bev_pos = positional_encoding(bev_h, bev_w)
```

#### **B. 空间交叉注意力（Spatial Cross-Attention, SCA）**

**核心操作**：每个BEV query从相关的相机视图采样特征

**流程**：
1. 将BEV query对应的3D位置（多个高度）投影到各相机
2. 在投影的2D位置使用**可变形注意力**采样特征
3. 聚合所有命中相机的特征

```python
# 伪代码
def spatial_cross_attention(bev_query, image_features, cameras):
    # 每个BEV位置(x,y)对应柱状的多个3D点
    for (x, y) in bev_grid:
        # 沿高度方向采样 N_ref 个3D参考点
        ref_points_3d = [(x, y, z) for z in height_anchors]  # 如4个高度
        
        total_feat = 0
        hit_count = 0
        for cam in cameras:
            for pt_3d in ref_points_3d:
                pt_2d = project(pt_3d, cam)
                if pt_2d in view:
                    # 可变形注意力采样
                    feat = deformable_attn(bev_query[x,y], pt_2d, image_features[cam])
                    total_feat += feat
                    hit_count += 1
        
        bev_query[x, y] = total_feat / hit_count  # 平均命中相机
```

**关键点**：
- 使用**Deformable Attention**，只采样少量关键点（如8个），计算高效
- 每个BEV位置沿高度采样多个3D点（pillar-like），覆盖不同高度物体
- 只从视野内相机采样，自动处理相机重叠区域

#### **C. 时间自注意力（Temporal Self-Attention, TSA）**

**核心操作**：当前BEV query与历史BEV特征交互

**流程**：
1. 将历史BEV特征根据自车运动对齐到当前坐标系
2. 当前BEV query与对齐后的历史BEV做可变形注意力

```python
# 伪代码
def temporal_self_attention(bev_query_t, bev_feature_t_minus_1, ego_motion):
    # 根据自车运动将历史BEV对齐到当前帧
    bev_aligned = align_by_ego_motion(bev_feature_t_minus_1, ego_motion)
    
    # 当前query与 [自身 + 对齐历史] 做可变形注意力
    for (x, y) in bev_grid:
        # 采样当前和历史特征
        feat_current = deformable_attn(bev_query_t[x,y], bev_query_t)
        feat_history = deformable_attn(bev_query_t[x,y], bev_aligned)
        bev_query_t[x, y] = (feat_current + feat_history) / 2
```

**关键点**：
- **运动补偿**：根据ego pose将历史BEV warp到当前坐标系
- **递归融合**：t时刻融合t-1，t-1融合t-2，隐式聚合长时序信息
- **速度估计增强**：时序信息显著提升物体速度预测精度

#### **D. BEVFormer Encoder层**

每层的执行顺序：

```
BEV Query
    ↓
Temporal Self-Attention (与历史BEV交互)
    ↓
Add & Norm
    ↓
Spatial Cross-Attention (与多相机特征交互)
    ↓
Add & Norm
    ↓
FFN (Feed-Forward Network)
    ↓
Add & Norm
    ↓
输出 BEV Query
```

堆叠6层，逐步细化BEV表示。

#### **E. 下游任务头**

**检测头**（基于Deformable DETR）：
- Object queries + BEV特征
- 输出：3D框 `[x, y, z, w, l, h, θ, vx, vy]`
- 使用匈牙利匹配 + Focal Loss + L1 Loss

**分割头**（地图分割）：
- 对BEV特征上采样
- 逐像素分类：道路、车道线、人行道等

---

## 三、关键技术细节

### 3.1 空间交叉注意力的3D参考点

**BEV到3D的映射**：
```python
# 每个BEV网格 (i, j) 对应真实世界坐标
x_real = (i - bev_w/2) * grid_size  # 米
y_real = (j - bev_h/2) * grid_size

# 沿高度方向锚定多个点（覆盖不同高度物体）
z_anchors = [-5, -3, -1, 1]  # 4个高度锚点

ref_points_3d = [(x_real, y_real, z) for z in z_anchors]
```

### 3.2 可变形注意力（Deformable Attention）

**动机**：全局注意力计算量大（O(N²)），可变形注意力只关注少量关键点

```python
def deformable_attention(query, reference_point, features):
    # 预测采样偏移
    offsets = linear_offset(query)  # [num_heads, num_points, 2]
    
    # 预测注意力权重
    attn_weights = softmax(linear_weight(query))  # [num_heads, num_points]
    
    # 在 reference_point + offsets 处采样
    sampling_locations = reference_point + offsets
    sampled = bilinear_sample(features, sampling_locations)
    
    # 加权聚合
    output = sum(attn_weights * sampled)
    return output
```

**优势**：
- 计算复杂度线性，适合高分辨率BEV
- 自适应聚焦关键区域

### 3.3 时序对齐（Ego-Motion Alignment）

**问题**：自车在移动，历史BEV和当前BEV坐标系不一致

**解决**：
```python
# 已知自车从 t-1 到 t 的位姿变化
T_ego = get_ego_pose_transform(t-1, t)  # 4x4变换矩阵

# 将历史BEV特征warp到当前坐标系
bev_history_aligned = grid_sample(
    bev_history, 
    warp_grid(bev_grid, T_ego)
)
```

### 3.4 两种时序策略

**BEVFormer（静态版）**：
- 仅使用当前帧
- Temporal Self-Attention退化为普通Self-Attention

**BEVFormer（时序版）**：
- 使用历史 2-4 帧
- 递归融合，显著提升速度估计和遮挡处理

---

## 4. 训练策略

### 4.1 损失函数

**检测损失**（与Deformable DETR一致）：
```python
L_det = λ_cls * FocalLoss + λ_bbox * L1Loss + λ_giou * GIoULoss
```

**分割损失**（可选）：
```python
L_seg = CrossEntropyLoss(pred_map, gt_map)
```

**总损失**：
```python
L_total = L_det + λ_seg * L_seg
```

### 4.2 训练细节

- **优化器**：AdamW
- **学习率**：2e-4，余弦衰减
- **骨干网络**：ResNet-101-DCN（可变形卷积）/ VoVNet-99
- **BEV分辨率**：200×200
- **训练轮数**：24 epochs（nuScenes）
- **时序帧数**：3-4帧历史
- **梯度裁剪**：max_norm=35

### 4.3 数据增强

- 图像级：随机缩放、翻转、颜色抖动
- BEV级：随机旋转、平移（需同步变换GT）
- 注意：时序训练时需保持帧间一致性

---

## 五、性能表现

### 5.1 nuScenes 验证集/测试集

| 方法 | Backbone | mAP ↑ | NDS ↑ | mAVE ↓ |
|------|----------|-------|-------|--------|
| DETR3D | R101 | 34.9 | 43.4 | 0.845 |
| PETR | R101 | 37.0 | 44.2 | 0.808 |
| **BEVFormer-S**（静态） | R101 | 37.5 | 44.8 | 0.788 |
| **BEVFormer**（时序） | R101 | **41.6** | **51.7** | **0.394** |
| BEVFormer | V2-99 | 48.1 | 56.9 | 0.378 |

**关键观察**：
- ✅ 时序版相比静态版 **mAVE（速度误差）大幅降低**（0.788→0.394）
- ✅ 时序信息显著提升整体NDS（44.8→51.7）
- ✅ 更强骨干（VoVNet）进一步提升性能

### 5.2 地图分割性能

| 方法 | IoU (道路) | IoU (车道) |
|------|-----------|-----------|
| Lift-Splat | 72.9 | 20.0 |
| BEVFormer | **80.1** | **25.7** |

---

## 六、优势与局限

### 6.1 优势

1. ✅ **显式BEV表示**：可复用于多任务（检测、分割、规划）
2. ✅ **强时序建模**：历史信息提升速度估计和遮挡处理
3. ✅ **计算高效**：可变形注意力降低复杂度
4. ✅ **多相机自然融合**：空间交叉注意力处理相机重叠
5. ✅ **成为行业基线**：后续大量工作基于BEVFormer

### 6.2 局限

1. ⚠️ **计算量仍较大**：BEV网格分辨率高时开销大
2. ⚠️ **高度信息压缩**：BEV表示丢失部分垂直细节
3. ⚠️ **依赖精确标定**：相机内外参误差影响投影质量
4. ⚠️ **时序内存开销**：需缓存历史BEV特征
5. ⚠️ **推理速度**：相比纯query方法（DETR3D）较慢

---

## 七、后续改进与变种

| 方法 | 主要改进 |
|------|---------|
| **BEVFormer v2** | 引入透视监督（perspective supervision），两阶段检测 |
| **BEVFusion** | 融合LiDAR和相机的BEV特征 |
| **BEVDet4D** | 时序BEV拼接，简化时序融合 |
| **PolarFormer** | 极坐标BEV表示，更适合环视相机 |
| **OccNet/OccFormer** | 基于BEVFormer扩展到3D占据预测 |

---

## 八、实现要点

### 8.1 空间交叉注意力核心代码

```python
class SpatialCrossAttention(nn.Module):
    def forward(self, bev_query, image_features, reference_points_3d, cameras):
        B, num_query, _ = bev_query.shape
        
        # 3D参考点投影到各相机
        ref_points_2d, mask = self.project_to_cameras(
            reference_points_3d, cameras
        )  # mask标记点是否在视野内
        
        # 可变形注意力采样
        output = self.deformable_attn(
            query=bev_query,
            reference_points=ref_points_2d,
            value=image_features,
            mask=mask
        )
        return output
```

### 8.2 时间自注意力核心代码

```python
class TemporalSelfAttention(nn.Module):
    def forward(self, bev_query, prev_bev, ego_motion):
        if prev_bev is not None:
            # 运动补偿对齐
            prev_bev = self.align_prev_bev(prev_bev, ego_motion)
            # 拼接当前和历史
            value = torch.stack([bev_query, prev_bev], dim=1)
        else:
            value = torch.stack([bev_query, bev_query], dim=1)
        
        output = self.deformable_attn(
            query=bev_query,
            value=value,
            reference_points=self.bev_reference_points
        )
        return output
```

---

## 九、与 DETR3D 对比

| 维度 | DETR3D | BEVFormer |
|------|--------|-----------|
| **中间表示** | 无（直接query） | 显式BEV网格 |
| **特征采样** | 单点3D-to-2D投影 | 可变形注意力（多点） |
| **时序建模** | 弱 | 强（Temporal Self-Attn） |
| **多任务** | 主要检测 | 检测+分割 |
| **速度估计** | 较弱 | 强（时序增强） |
| **推理速度** | 快（26 FPS） | 较慢（~15 FPS） |
| **精度（NDS）** | 43.4 | 51.7 |
| **可复用性** | 低 | 高（BEV可复用） |

---

## 参考资源

- **论文**：[BEVFormer (ECCV 2022)](https://arxiv.org/abs/2203.17270)
- **官方代码**：https://github.com/fundamentalvision/BEVFormer
- **数据集**：nuScenes, Waymo Open Dataset
- **相关综述**：见 `occ_summary.md`（占据网络综述，BEVFormer作为核心编码器）

---

*文档整理：DETR3D 系列感知方法技术要点*
*相关文档：[detr3d.md](./detr3d.md)、[occ_summary.md](./occ_summary.md)*
