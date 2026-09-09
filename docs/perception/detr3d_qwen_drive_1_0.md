# Qwen-Drive-1.0 如何实践 DETR3D

> 本文基于对工程 `Qwen-Drive-1.0/src/qwen_drive_perception/` 源码的逐文件分析整理，
> 所有结论均可在代码中找到对应位置（文件名 + 行号）。

---

## 〇、一句话结论（重要的准确性说明）

**Qwen-Drive-1.0 的 3D 检测头并不是"原版 DETR3D"，而是 DETR3D 血统的 query-based 集合预测解码器，被嫁接到 BEVFormer 的显式 BEV 特征图之上。**

- **DETR3D 的核心范式**（可学习 object query + 3D 参考点 + 迭代细化 + set prediction + 无 NMS）→ **完整保留**，见 `bev_encoder.py` 的 `DetectionTransformerDecoder` 与 `heads.py`。
- **DETR3D 的标志性操作**（3D 参考点直接投影到多相机 2D 图像特征采样）→ **未用在检测解码器上**。Qwen-Drive 的解码器 query 采样的是 **BEV 平面特征**（`value=bev_embed`），这正是 BEVFormer 检测头的做法。
- "3D-to-2D 投影到多相机图像"这一 DETR3D 精髓，在 Qwen-Drive 中出现在 **BEV 编码器**（`SpatialCrossAttention` + `point_sampling`），而非检测头。

因此本文的定位是：**梳理 Qwen-Drive 检测头中继承自 DETR3D 的那条技术血脉，以及它相对原版做了哪些工程化改造。**

相关文档：[detr3d.md](./detr3d.md)（DETR3D 原理）、[bevformer.md](./bevformer.md)（BEV 编码器）。

---

## 一、DETR3D 范式回顾（用于对照）

原版 DETR3D 的检测流程：

1. 一组可学习的 object queries；
2. 每个 query 通过一个线性层预测一个 **3D 参考点**；
3. 3D 参考点投影到各相机 → 采样图像特征 → 更新 query；
4. 多层 decoder **迭代细化**参考点；
5. 回归头输出 3D 框，分类头输出类别；
6. **集合预测**（匈牙利匹配训练）、**无 NMS**。

Qwen-Drive 保留了其中的 1、2、4、5、6，而把第 3 步的"投影到图像"替换成"在 BEV 平面上做可变形注意力"。

---

## 二、DETR3D 血统组件在 Qwen-Drive 中的映射

| DETR3D 概念 | Qwen-Drive 对应实现 | 位置 |
|-------------|---------------------|------|
| Object Queries | `query_embedding = nn.Embedding(900, 256*2)` | `heads.py:223` |
| Query 拆分 (pos/content) | `torch.split(object_query_embed, embed_dims, dim=1)` | `perception_transformer.py:225` |
| 3D 参考点生成 | `reference_points = nn.Linear(256, 3)` → `sigmoid()` | `perception_transformer.py:76, 228` |
| Decoder（6 层迭代细化） | `DetectionTransformerDecoder` | `bev_encoder.py:366` |
| Decoder 层结构 | 自注意力 + 可变形交叉注意力 + FFN | `bev_encoder.py:350-363` |
| 参考点迭代更新 | `tmp + inverse_sigmoid(ref)` → `sigmoid()` | `bev_encoder.py:406-413` |
| 回归分支（逐层） | `reg_branches` (ModuleList × 6) | `heads.py:212-220` |
| 分类分支（逐层） | `cls_branches` (ModuleList × 6) | `heads.py:206-219` |
| 框编码 code_size=10 | `[x,y,z,w,l,h,sin,cos,vx,vy]` | `heads.py:37-50` |
| 无 NMS 解码 | `NMSFreeCoder`（top-k） | `heads.py:53-86` |
| 集合预测输出 | top-300 框，`labels = idx % num_classes` | `heads.py:62-65` |

**关键差异**：

| DETR3D 概念 | 原版做法 | Qwen-Drive 做法 | 位置 |
|-------------|---------|-----------------|------|
| 交叉注意力的 value | 多相机图像特征 | **BEV embedding** | `perception_transformer.py:233, 240` |
| 参考点维度 | 3D (x,y,z) 投影到图像 | 用 **2D (x,y)** 索引 BEV 平面 | `bev_encoder.py:394` |
| 特征采样 | 3D→2D 相机投影 | BEV 平面可变形采样 | `attention.py:288 CustomMSDeformableAttention` |

---

## 三、检测解码器详解（DETR3D 血脉的主战场）

### 3.1 Object Queries 与 3D 参考点

在 `BEVFormerHead` 中定义 900 个 object query，每个维度为 `embed_dims * 2 = 512`：

```python
# heads.py:223
self.query_embedding = nn.Embedding(num_query, embed_dims * 2)  # (900, 512)
```

在 `PerceptionTransformer.forward` 中，把 512 维拆成 **位置部分**和**内容部分**，
并从位置部分线性映射出 **3D 参考点**（DETR3D 的标志动作）：

```python
# perception_transformer.py:225-229
query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)  # 各 256
query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
query = query.unsqueeze(0).expand(bs, -1, -1)
reference_points = self.reference_points(query_pos).sigmoid()   # Linear(256, 3) → (bs, 900, 3)
init_reference_out = reference_points
```

- `reference_points` 是 `nn.Linear(embed_dims, 3)`（`perception_transformer.py:76`），
  即每个 query 预测一个归一化的 `(x, y, z)` 参考点，`sigmoid` 到 `[0,1]`。
- 这与原版 DETR3D 完全一致：**参考点由 query 学习得到，而非手工网格。**

### 3.2 Decoder 层：自注意力 + 可变形交叉注意力

每个 decoder 层（`DetrTransformerDecoderLayer`，`bev_encoder.py:350`）由两种注意力组成：

```python
# bev_encoder.py:355-363
attn_modules=[
    MultiheadAttention(embed_dims=256, num_heads=8, dropout=0.1, batch_first=False),  # query 间自注意力
    CustomMSDeformableAttention(embed_dims=256, num_heads=8, num_levels=1, num_points=4),  # 交叉注意力 → BEV
]
```

- **自注意力**：object query 之间交互（去重、建模物体关系）——与 DETR3D 一致。
- **交叉注意力**：`CustomMSDeformableAttention`，但它的 `value` 是 **BEV embedding**，不是图像。
  这是与原版 DETR3D 最本质的区别：

```python
# perception_transformer.py:233, 237-248
bev_embed_dec = bev_embed_for_decoder.permute(1, 0, 2)   # BEV 作为 value
...
inter_states, inter_references = self.decoder(
    query=query,
    key=None,
    value=bev_embed_dec,          # ← 采样源是 BEV 平面，而非多相机图像
    reference_points=reference_points,
    reg_branches=reg_branches,
    spatial_shapes=bev_spatial_shapes,   # [[200, 200]]
    ...
)
```

参考点在进入可变形注意力时**只取 (x, y) 两维**去索引 200×200 的 BEV 平面：

```python
# bev_encoder.py:394
reference_points_input = reference_points[..., :2].unsqueeze(2)   # 只用 x,y 索引 BEV
```

> 换言之：原版 DETR3D 把 3D 点投影到图像取特征；Qwen-Drive 把 3D 点的水平分量 (x,y)
> 直接落在 BEV 网格上取特征，z 分量只参与后续框回归。

### 3.3 迭代参考点细化（DETR3D / Deformable DETR 精髓）

每层预测偏移量，叠加到上一层参考点（在 `inverse_sigmoid` 空间累加），再 `sigmoid` 回来：

```python
# bev_encoder.py:406-413
if reg_branches is not None:
    tmp = reg_branches[lid](output)
    assert reference_points.shape[-1] == 3
    new_reference_points = torch.zeros_like(reference_points)
    new_reference_points[..., :2]  = tmp[..., :2]  + inverse_sigmoid(reference_points[..., :2])  # x,y
    new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])  # z 来自 tmp 的第4维
    new_reference_points = new_reference_points.sigmoid()
    reference_points = new_reference_points.detach()   # 逐层细化，detach 稳定训练
```

- 6 层解码器逐步逼近真实框位置——这正是 DETR3D 的迭代 refinement。
- 注意 z 的偏移取自回归向量的第 4 维（`tmp[..., 4:5]`），对应框中心高度 `cz`。

### 3.4 逐层框解码

`BEVFormerHead.forward` 对每一层输出做分类 + 回归，并把回归结果叠加到该层参考点后反归一化到真实尺度：

```python
# heads.py:273-292
for lvl in range(hs.shape[0]):
    reference = init_reference if lvl == 0 else inter_references[lvl - 1]
    reference = inverse_sigmoid(reference)
    outputs_class = self.cls_branches[lvl](hs[lvl])
    tmp = self.reg_branches[lvl](hs[lvl])

    tmp[..., 0:2] += reference[..., 0:2]          # x,y 叠加参考点
    tmp[..., 0:2]  = tmp[..., 0:2].sigmoid()
    tmp[..., 4:5] += reference[..., 2:3]          # z(cz) 叠加参考点
    tmp[..., 4:5]  = tmp[..., 4:5].sigmoid()
    # 反归一化到 det_pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.4)
    tmp[..., 0:1] = tmp[..., 0:1] * (pc[3]-pc[0]) + pc[0]
    tmp[..., 1:2] = tmp[..., 1:2] * (pc[4]-pc[1]) + pc[1]
    tmp[..., 4:5] = tmp[..., 4:5] * (pc[5]-pc[2]) + pc[2]
```

### 3.5 框编码：10 维 + sin/cos 航向角

回归向量 `code_size = 10`，航向角用 `(sin, cos)` 双分量编码（DETR3D 标准做法）：

```python
# heads.py:37-50  denormalize_bbox
rot = torch.atan2(rot_sine, rot_cosine)     # sin,cos → yaw
cx, cy, cz = ...
w = normalized[..., 2:3].exp()               # 尺寸用指数解码
l = normalized[..., 3:4].exp()
h = normalized[..., 5:6].exp()
vx, vy = ...                                 # 速度
# 输出 [cx, cy, cz, w, l, h, yaw, vx, vy]  (9 维)
```

- 内部 10 维顺序：`[x, y, w, l, cz, h, sin, cos, vx, vy]`（注意 `cz` 在第 4 索引）。
- 解码后 9 维：`[x, y, z, w, l, h, yaw, vx, vy]`。

### 3.6 无 NMS 的 top-k 解码（集合预测）

`NMSFreeCoder` 完全不用 NMS，直接对所有 query×类别 的 sigmoid 分数取 top-300：

```python
# heads.py:62-79  NMSFreeCoder.decode_single
cls_scores = cls_scores.sigmoid()
scores, indexs = cls_scores.view(-1).topk(self.max_num)   # max_num = 300
labels = indexs % self.num_classes          # 7 类
bbox_index = indexs // self.num_classes
bbox_preds = bbox_preds[bbox_index]
final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
# 用 post_center_range = (-61.2,-61.2,-10, 61.2,61.2,10) 过滤越界框
mask  = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)
```

- `labels = index % num_classes`、`bbox_index = index // num_classes`：一个 query 可对多个类别打分，取全局 top-k——这是 DETR3D/Deformable-DETR 的经典无 NMS 解码。

### 3.7 坐标系转换：ego → lidar

解码在 ego 系进行，最后转回 lidar 系并把 z 从重心移到底面：

```python
# heads.py:319-329
if img_metas[i].get("box_coord_system") == "ego":
    bboxes = geometry.ego_to_lidar_boxes(bboxes, lidar2ego)
bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5   # z: 重心 → 底面
```

---

## 四、整体数据流（检测分支）

```
Qwen3.5 VLM 双流特征 (LLM tokens + ViT patches)
        │
        ▼
BEVFormer Encoder（6 层，200×200 BEV）      ← 空间交叉注意力在此把 3D 点投影到多相机图像
        │  bev_embed  (显式 BEV 平面特征)
        ▼
┌──────────────────────────────────────────┐
│  DETR3D 血统检测解码器（6 层）              │
│                                            │
│  900 object queries ──split──▶ query_pos  │
│                                └▶ Linear(256,3)+sigmoid ▶ 3D 参考点
│  ┌─ 每层: ─────────────────────────────┐  │
│  │  1) MultiheadAttention (query 自注意力)│  │
│  │  2) CustomMSDeformableAttention        │  │
│  │       value = BEV embedding            │  │  ← 差异点：采样 BEV 而非图像
│  │       ref_pt = reference_points[:,:2]  │  │
│  │  3) FFN                                │  │
│  │  4) reg_branch → 更新参考点(inv_sigmoid)│  │
│  └────────────────────────────────────────┘  │
│                                            │
│  cls_branches → 类别打分                    │
│  reg_branches → 10 维框(sin/cos yaw)        │
└──────────────────────────────────────────┘
        │
        ▼
NMSFreeCoder: sigmoid → top-300 → 反归一化 → post_center 过滤
        │
        ▼
ego → lidar 坐标转换, z 重心→底面
        │
        ▼
输出: boxes (N,9), scores, labels
```

---

## 五、关键配置一览（`configuration_perception.py`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `NUM_QUERY` | 900 | object query 数量（DETR3D 风格） |
| `CODE_SIZE` | 10 | 框回归维度 `[x,y,w,l,cz,h,sin,cos,vx,vy]` |
| `NUM_DECODER_LAYERS` | 6 | 解码器层数（迭代细化） |
| `DECODER_NUM_POINTS` | 4 | 可变形注意力采样点数 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `EMBED_DIM` | 256 | 特征维度 |
| `MAX_NUM_BOXES` | 300 | top-k 输出框上限 |
| `DET_PC_RANGE` | (-51.2,-51.2,-5.0, 51.2,51.2,5.4) | 检测空间范围（ego 系，米） |
| `POST_CENTER_RANGE` | (-61.2,-61.2,-10, 61.2,61.2,10) | 越界框过滤范围 |
| `DET_CLASS_NAMES` | 7 类 | vehicle/czone_sign/bicycle/generic_object/pedestrian/traffic_cone/barrier |
| `BEV_H = BEV_W` | 200 | BEV 网格分辨率 |

---

## 六、与原版 DETR3D / BEVFormer 的对照

| 维度 | 原版 DETR3D | BEVFormer 检测头 | **Qwen-Drive-1.0** |
|------|-------------|------------------|---------------------|
| 中间表示 | 无（直接 3D→2D 图像） | 显式 BEV | **显式 BEV（同 BEVFormer）** |
| 交叉注意力 value | 多相机图像特征 | BEV 特征 | **BEV 特征** |
| Object query | ✅ | ✅ | ✅ 900 个 |
| 3D 参考点 | ✅ Linear 预测 | ✅ | ✅ `Linear(256,3)` |
| 迭代细化 | ✅ | ✅ | ✅ 6 层 inverse_sigmoid 累加 |
| 无 NMS 集合预测 | ✅ | ✅ | ✅ `NMSFreeCoder` top-300 |
| 航向角编码 | sin/cos | sin/cos | ✅ sin/cos |
| 特征来源 | CNN 图像特征 | CNN 图像特征 | **VLM 双流 (LLM + ViT)** |

**结论**：Qwen-Drive 的检测头 = **BEVFormer 检测头**（本身就是 DETR3D 检测头的 BEV 变体），
特征输入换成了 Qwen3.5 VLM 的双流特征。它继承了 DETR3D 的**查询式集合预测范式**，
但没有沿用 DETR3D 的**直接 3D→2D 图像采样**——后者被 BEVFormer 的空间交叉注意力取代（放在了编码器里）。

---

## 七、DETR3D 精髓"3D-to-2D 投影"其实在哪里？

虽然检测解码器不做图像投影，但 DETR3D 那套"3D 参考点投影到多相机图像"的思想
**被搬到了 BEV 编码器的空间交叉注意力**里（`bev_encoder.py:231-286` `point_sampling`）：

```python
# bev_encoder.py:246-263 (point_sampling, 简化)
# BEV 3D 参考点(ego 系) → lidar → 图像像素
reference_points[..., 0:1] = ref * (pc[3]-pc[0]) + pc[0]   # 反归一化到真实坐标
...
ego2lidar = torch.linalg.inv(lidar2ego)
lidar2img = lidar2img @ ego2lidar
reference_points_cam = (lidar2img @ reference_points)      # 投影到各相机
# 归一化到像素平面，落在视野内的才参与采样
```

- 每个 BEV 网格沿高度取 4 个点（`POINTS_IN_PILLAR = 4`），投影到 6/8 个相机；
- 只有落在相机视野内的投影点才被 `SpatialCrossAttention` 采样（`attention.py:211-285`）；
- 这正是 DETR3D "3D-to-2D queries" 思想的体现，只不过对象是 **BEV query** 而非 **object query**。

> 因此完整地说：Qwen-Drive **在编码阶段用 DETR3D 式的 3D→2D 投影构建 BEV**，
> **在检测阶段用 DETR3D 式的查询解码器输出框**——两处都流着 DETR3D 的血。

---

## 八、总结

1. **Qwen-Drive-1.0 的 3D 检测严格来说是 BEVFormer 架构**，而 BEVFormer 的检测头
   本身脱胎于 DETR3D / Deformable-DETR。
2. **DETR3D 的查询式集合预测范式被完整继承**：900 个可学习 query、Linear 生成 3D 参考点、
   6 层迭代细化、sin/cos 航向角、`NMSFreeCoder` 无 NMS top-k 解码。
3. **与原版 DETR3D 的本质区别**：检测解码器的交叉注意力采样的是**显式 BEV 平面**
   （`value=bev_embed`，2D 参考点索引），而非直接投影到多相机图像。
4. **DETR3D 的 3D→2D 投影精髓**被前移到 **BEV 编码器的空间交叉注意力**，
   服务于 BEV 特征构建（`point_sampling`）。
5. **最大工程创新**：检测头的输入特征来自 **Qwen3.5 VLM 的双流特征**
   （LLM 图像 token + ViT patch），而非传统 CNN，从而把 3D 检测统一进 VLM 框架。

---

## 附：涉及的源码文件

| 文件 | 作用 |
|------|------|
| `configuration_perception.py` | 冻结的超参数（query 数、code_size、范围、类别等） |
| `heads.py` | `BEVFormerHead`、`NMSFreeCoder`、`denormalize_bbox`、逐层解码、ego→lidar |
| `perception_transformer.py` | query 拆分、参考点生成、调用解码器、occ 分支 |
| `bev_encoder.py` | `DetectionTransformerDecoder`、`DetrTransformerDecoderLayer`、迭代细化、`point_sampling` |
| `attention.py` | `CustomMSDeformableAttention`（检测交叉注意力）、`SpatialCrossAttention`（编码器 3D→2D） |

---

*文档整理：Qwen-Drive-1.0 感知头源码分析*
*相关文档：[detr3d.md](./detr3d.md)、[bevformer.md](./bevformer.md)、[occ_summary.md](./occ_summary.md)*
