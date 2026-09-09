# Qwen-Drive-1.0 如何实践 BEVFormer

> 本文基于对工程 `Qwen-Drive-1.0/src/qwen_drive_perception/` 源码的完整分析整理，
> 所有结论均附带**文件名+行号**引用，可直接在代码中验证。

---

## 0. 核心结论

**Qwen-Drive-1.0 的感知头是一个完整的 BEVFormer 架构实现，但做了关键改造：**

1. ✅ **BEV 编码器**：6 层 BEVFormerLayer（时间自注意力 + 空间交叉注意力），完整保留
2. ✅ **显式 BEV 表示**：200×200 网格，0.512m 分辨率，覆盖 102.4m × 102.4m
3. ✅ **时空注意力机制**：单帧推理时退化为空间注意力（历史 BEV 为 None）
4. 🔧 **特征骨干改造**：**从 ResNet/VoVNet CNN 换成 Qwen3.5 VLM 双流特征**（LLM tokens + ViT patches）
5. 🔧 **多任务扩展**：在 BEV 上统一实现 3D 检测 + occupancy 预测 + BEV 地图分割

相关文档：[bevformer.md](./bevformer.md)（原理）、[detr3d_qwen_drive_1_0.md](./detr3d_qwen_drive_1_0.md)（检测头）。

---

## 1. 整体架构对照

### 1.1 原版 BEVFormer vs Qwen-Drive

| 模块 | 原版 BEVFormer | Qwen-Drive-1.0 | 文件位置 |
|------|----------------|----------------|----------|
| **骨干网络** | ResNet-101-DCN / VoVNet | **Qwen3.5 VLM** | `modeling_perception.py:124-178` |
| **特征提取** | CNN 多尺度特征 (FPN) | **VLM 双流**：LLM tokens + ViT patches | `modeling_perception.py:138, 152` |
| **BEV Queries** | 可学习嵌入 (200×200) | ✅ 同样 | `heads.py:222` |
| **BEV 编码器** | 6 层 BEVFormerLayer | ✅ 6 层 BEVFormerLayer | `bev_encoder.py:185-347` |
| **时间自注意力** | TSA（融合历史 BEV） | ✅ 支持（推理时退化） | `attention.py:71-147` |
| **空间交叉注意力** | SCA（3D→2D 投影采样） | ✅ 完整实现 | `attention.py:211-285`, `bev_encoder.py:231-286` |
| **检测头** | Deformable DETR decoder | ✅ 6 层 decoder + NMS-free | `bev_encoder.py:366-419`, `heads.py:143-307` |
| **额外任务** | 地图分割 | ✅ + **occupancy 预测** | `perception_transformer.py:78-84, 262-264` |

---

## 2. BEV 编码器：BEVFormer 核心

### 2.1 BEV Queries 初始化

**可学习的网格状 BEV queries**（BEVFormer 的基础）：

```python
# heads.py:222
self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)  # (40000, 256)
# 200×200 = 40,000 个 query，每个 256 维
```

**位置编码**（可学习，区别于 sinusoidal）：

```python
# heads.py:202-204
self.positional_encoding = LearnedPositionalEncoding(
    num_feats=embed_dims // 2,   # 128
    row_num_embed=bev_h,         # 200
    col_num_embed=bev_w          # 200
)
```

**融合 UVTR 特征**（Qwen-Drive 扩展）：

```python
# heads.py:244-247
if uvtr_bev_feat is not None:
    bev_queries = bev_queries + uvtr_bev_feat.to(dtype)
elif vit_bev_feat is not None:
    bev_queries = bev_queries + vit_bev_feat.to(dtype)
```

- `uvtr_bev_feat` 来自 UVTR 体素池化的 BEV 投影（`view_transform.py`）
- 这是 Qwen-Drive 对 BEVFormer 的增强：额外的几何先验

### 2.2 BEVFormer 编码器层结构

每层（`BEVFormerLayer`, `bev_encoder.py:91-168`）包含：

```python
# bev_encoder.py:95-103
attn_modules=[
    TemporalSelfAttention(embed_dims=256, num_heads=8, num_levels=1),     # TSA
    SpatialCrossAttention(pc_range=pc_range, embed_dims=256),             # SCA
],
ffn=FFN(embed_dims=256, feedforward_channels=512, ffn_drop=0.1)
```

**执行顺序**（严格按 BEVFormer 论文）：

```python
# bev_encoder.py:129-166 (简化)
for layer in operation_order:  # ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
    if layer == "self_attn":
        query = TemporalSelfAttention(query, prev_bev, ...)  # 时间维度
    elif layer == "cross_attn":
        query = SpatialCrossAttention(query, image_features, ...)  # 空间维度
    elif layer == "ffn":
        query = FFN(query)
```

### 2.3 时间自注意力（Temporal Self-Attention, TSA）

**作用**：融合历史 BEV 特征，提升运动物体检测和速度估计。

```python
# attention.py:71-147  TemporalSelfAttention
def forward(self, query, key=None, value=None, ...):
    if value is None:  # 单帧推理
        bs, len_bev, c = query.shape
        value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        # 历史为 None，退化为 [当前, 当前] 双份，等价于普通自注意力
```

**关键点**：
- 多帧推理时，`value` 包含当前帧和对齐后的历史 BEV
- 单帧推理（Qwen-Drive 默认），`value = [query, query]`，**退化为普通可变形自注意力**
- 采样点数：4 个（`TSA_NUM_POINTS = 4`，`configuration_perception.py:116`）

**可变形采样**（效率优化）：

```python
# attention.py:116-141
sampling_offsets = self.sampling_offsets(query)  # 预测偏移
attention_weights = self.attention_weights(query).softmax(-1)
sampling_locations = reference_points + sampling_offsets / offset_normalizer
output = multi_scale_deformable_attn_cuda(value, spatial_shapes, ...)
```

### 2.4 空间交叉注意力（Spatial Cross-Attention, SCA）

**作用**：BEV query 从多相机图像特征聚合信息——**BEVFormer 的核心创新**。

#### A. 3D 参考点生成与投影

每个 BEV query 对应一个平面位置 `(x, y)`，沿高度 z 采样多个锚点（pillar-like）：

```python
# bev_encoder.py:198-220  get_reference_points (dim="3d")
zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, ...)  # 4 个高度
xs = torch.linspace(0.5, W - 0.5, W, ...) / W                  # 归一化 x
ys = torch.linspace(0.5, H - 0.5, H, ...) / H                  # 归一化 y
ref_3d = torch.stack((xs, ys, zs), -1)  # (4, 200, 200, 3)
```

**投影到各相机**（`point_sampling`, `bev_encoder.py:231-286`）：

```python
# bev_encoder.py:246-268 (核心投影逻辑)
# 1. 反归一化到真实坐标（ego 系）
reference_points[..., 0:1] = ref * (pc_range[3] - pc_range[0]) + pc_range[0]
reference_points[..., 1:2] = ref * (pc_range[4] - pc_range[1]) + pc_range[1]
reference_points[..., 2:3] = ref * (pc_range[5] - pc_range[2]) + pc_range[2]

# 2. 齐次坐标
reference_points = torch.cat((reference_points, torch.ones_like(...)), -1)  # (x,y,z,1)

# 3. ego → lidar → 图像
ego2lidar = torch.linalg.inv(lidar2ego)
lidar2img = lidar2img @ ego2lidar
reference_points_cam = torch.matmul(lidar2img, reference_points)  # 投影

# 4. 透视除法 + 归一化到 [0,1]
reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3]
reference_points_cam[..., 0] /= img_shape[1]  # 宽度归一化
reference_points_cam[..., 1] /= img_shape[0]  # 高度归一化

# 5. 生成可见性掩码
bev_mask = (depth > eps) & (x > 0) & (x < 1) & (y > 0) & (y < 1)
```

**输出**：
- `reference_points_cam`: `(N_cam, N_query, N_height, 2)` 投影后的 2D 坐标
- `bev_mask`: `(N_cam, N_query, N_height)` 是否在视野内

#### B. 相机感知的特征采样

`SpatialCrossAttention` (`attention.py:211-285`) 使用**向量化 top-k rebatch**优化：

```python
# attention.py:251-273 (核心采样逻辑)
# 1. 找出每个相机命中的 BEV query（有至少一个高度点在视野内）
mask_any = bev_mask[:, 0].any(dim=-1)  # (N_cam, N_query)
valid_counts = mask_any.sum(dim=-1)     # 每个相机命中多少 query
max_len = int(valid_counts.max().item())
rebatch_indices = mask_any.topk(max_len, dim=-1).indices  # top-k 有效 query

# 2. 根据 rebatch_indices 重新组织 query 和参考点
queries_rebatch = query.gather(2, query_index)              # (bs, N_cam, max_len, C)
reference_points_rebatch = reference_points_cam.gather(...) # (bs, N_cam, max_len, D, 2)

# 3. 可变形注意力采样（8 个采样点）
queries = deformable_attention(
    query=queries_rebatch.view(bs * N_cam, max_len, embed_dims),
    value=image_features,  # 多相机图像特征
    reference_points=reference_points_rebatch,
    spatial_shapes=spatial_shapes,  # 多尺度 [(H1,W1), (H2,W2), ...]
)

# 4. scatter 回原始 BEV 位置并平均
slots.scatter_add_(1, scatter_index, queries)
count = bev_mask.sum(-1) > 0  # 每个 query 被多少相机命中
slots = slots / count[..., None]
```

**关键点**：
- 采样点数：8 个（`SCA_NUM_POINTS = 8`）
- 多尺度：4 层 FPN 特征（`NUM_FEATURE_LEVELS = 4`）
- 自动处理相机重叠区域（平均命中相机的特征）

---

## 3. 特征输入：VLM 双流替换 CNN

### 3.1 原版 BEVFormer 的特征流

```
多相机图像 → ResNet-101-DCN → FPN → 多尺度特征 {C2, C3, C4, C5}
```

### 3.2 Qwen-Drive 的 VLM 双流

```python
# modeling_perception.py:124-178  BEVFormerModelV2
# 主流：LLM 的图像 token 隐藏状态
self.adaptor = SimpleFPN(
    dim=config.llm_dim,        # 2560
    out_channels=config.embed_dim,  # 256
    scale_factors=(4.0, 2.0, 1.0, 0.5)
)

# UVTR 流：ViT 的 patch 特征
self.vit_neck = SimpleFPN(
    dim=config.vit_dim,        # 1024
    out_channels=config.embed_dim,  # 256
    scale_factors=(1.0,)
)
```

**LLM 流处理**（主流，用于 SCA 采样）：

```python
# modeling_perception.py:130-142
feat_main = img_llm_feats.permute(0, 3, 1, 2)  # (N_cam, C, H/2, W/2)
mlvl_feats = self.adaptor(feat_main)           # 4 个尺度
mlvl_feats_reshaped = []
for feat in mlvl_feats:
    _, C_f, H_f, W_f = feat.shape
    mlvl_feats_reshaped.append(feat.view(B, N, C_f, H_f, W_f))
```

**ViT 流处理**（辅助流，用于 UVTR 深度估计）：

```python
# modeling_perception.py:144-156
feat_vit = img_vit_feats.permute(0, 3, 1, 2)  # (N_cam, C, H, W)
vit_mlvl_feats = self.vit_neck(feat_vit)       # 1 个尺度

# 深度估计
vit_depths = []
for feat in vit_mlvl_feats_reshaped:
    depth_logits = self.depth_net(feat.view(-1, *feat.shape[-3:]))
    vit_depths.append(depth_logits.softmax(dim=1))
```

### 3.3 特征捕获（VLM 前向传播）

在 `QwenDrivePerception.infer` 中通过 hook 捕获 VLM 的中间特征：

```python
# modeling_perception.py:242-266
captured = {}
def _hook(module, args, output=None):
    captured["patches"] = args[0]  # ViT 预融合 patch

visual = self._vlm.model.visual
handle = visual.merger.register_forward_hook(_hook)
try:
    outputs = self._vlm(input_ids=..., pixel_values=..., ...)
finally:
    handle.remove()

# 后处理
hidden_states = self._vlm.model.language_model.norm(outputs.hidden_states[-1])  # LLM 流
patches = visual.merger.norm(captured["patches"])  # ViT 流
vit_feats = self._premerge_grids(patches, image_grid_thw)
```

**关键点**：
- LLM 流：最后一层 decoder 输出，经过 final norm
- ViT 流：在 merger 之前的 patch 特征（896×512 → 28×16 patch grid）
- 不需要额外训练 VLM，直接复用预训练权重

---

## 4. 检测头：DETR3D 风格解码器

详见 [detr3d_qwen_drive_1_0.md](./detr3d_qwen_drive_1_0.md)，这里简述与 BEVFormer 的关系。

### 4.1 解码器输入

```python
# perception_transformer.py:237-248
bev_embed_dec = bev_embed_for_decoder.permute(1, 0, 2)  # BEV 编码器输出

inter_states, inter_references = self.decoder(
    query=query,                # 900 个 object query
    value=bev_embed_dec,        # ← 从 BEV 特征解码，而非图像
    reference_points=reference_points,  # 3D 参考点
    spatial_shapes=bev_spatial_shapes,  # [[200, 200]]
    ...
)
```

### 4.2 与 BEVFormer 原版的一致性

| 组件 | BEVFormer 原版 | Qwen-Drive | 位置 |
|------|----------------|-----------|------|
| 解码器类型 | Deformable DETR | ✅ 6 层 | `bev_encoder.py:366` |
| value 来源 | BEV 特征 | ✅ BEV 特征 | `perception_transformer.py:240` |
| 参考点 | 2D (x,y) | ✅ 2D | `bev_encoder.py:394` |
| 集合预测 | 匈牙利匹配 + 无 NMS | ✅ | `heads.py:53-86` |

---

## 5. Occupancy 预测：基于 BEV 的扩展

**BEVFormer 未包含 occupancy，这是 Qwen-Drive 的多任务扩展。**

### 5.1 从 BEV 到 Occupancy Volume

```python
# perception_transformer.py:253-260
# 1. 将 BEV 特征(200×200)重塑为体素(200×200×16)
bev_feat = self._adapt_bev_for_occ(
    bev_feat_ego, bev_h, bev_w, 
    occ_pc_range=active_occ_pc_range,
    occ_voxel_size=active_occ_voxel_size
)

# 2. 融合 UVTR 体素特征
if uvtr_occ_feat is not None:
    uvtr_occ_feat = self._adapt_volume_for_occ(uvtr_occ_feat, ...)
    bev_feat = self._fuse_uvtr_occ_feat(bev_feat, uvtr_occ_feat)

# 3. 3D U-Net 精细化
occ_feat = self.occ_decoder(bev_feat).permute(0, 4, 3, 2, 1)  # (B,X,Y,Z,C)
occ_pred = self.occ_pred_head(occ_feat)  # (B,X,Y,Z,10类)
```

**体积适配**（裁剪 + 重采样）：

```python
# perception_transformer.py:186-189
bev_feat_3d = bev_feat.view(bs, -1, self.occ_pillar_h, bev_h, bev_w)  # (B,C/16,16,200,200)
# 使用 grid_sample 裁剪到 occ_pc_range 并重采样到 (200,200,16)
return F.grid_sample(bev_feat_3d, grid, mode='bilinear', ...)
```

### 5.2 3D U-Net Refiner

```python
# occ_refiner.py:61-107  OccVoxelUNetRefiner
# 下采样（仅在 XY 平面，保留 Z）
self.enc1 = ResidualStage3D(c0, c1, stride=(1, 2, 2), num_blocks=2)
self.enc2 = ResidualStage3D(c1, c2, stride=(1, 2, 2), num_blocks=2)
self.enc3 = ResidualStage3D(c2, c3, stride=(1, 2, 2), num_blocks=2)

# 上采样 + 跳跃连接
self.dec2 = ResidualStage3D(c3 + c2, c2, num_blocks=2)
self.dec1 = ResidualStage3D(c2 + c1, c1, num_blocks=2)
self.dec0 = ResidualStage3D(c1 + c0, c0, num_blocks=2)
```

---

## 6. 坐标系统与变换

### 6.1 三大坐标系

| 坐标系 | 定义 | 用途 |
|--------|------|------|
| **lidar** | LiDAR 传感器坐标系 | 3D 框输出坐标系 |
| **ego** | 车辆坐标系（X前 Y左 Z上） | BEV 网格、occupancy 坐标系 |
| **image** | 像素坐标（896×512） | 图像特征采样 |

### 6.2 关键变换

**lidar ↔ ego**：

```python
# geometry.py:75-79  build_lidar2ego
mat = np.eye(4)
mat[:3, :3] = quaternion_rotation(rotation)  # 四元数 → 旋转矩阵
mat[:3, 3] = translation
```

**3D 点 → 图像**：

```python
# geometry.py:43-53  build_lidar2img
lidar2cam_rt = compose(sensor2lidar_rotation, sensor2lidar_translation)
lidar2img = cam_intrinsic @ lidar2cam_rt
# 再经过 apply_image_scale 处理 resize (896×512)
```

**框变换**（旋转中心 + 航向角 + 速度）：

```python
# geometry.py:82-94  _transform_boxes
transformed[..., :3] = (centers_homo @ rt.T)[..., :3]         # 平移+旋转中心
transformed[..., 6] += torch.atan2(rt[1,0], rt[0,0])          # 航向角叠加
transformed[..., 7:9] = velocity @ rt[:2,:2].T                # 速度旋转
```

---

## 7. 配置参数详解

### 7.1 BEV 空间配置

```python
# configuration_perception.py:88-93
DET_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.4)  # ego 系，米
DET_VOXEL_SIZE = (102.4/200, 102.4/200, 10.4)         # (0.512, 0.512, 10.4)
BEV_H = BEV_W = 200                                    # 网格分辨率
```

- 覆盖范围：102.4m × 102.4m × 10.4m（X前后 × Y左右 × Z高度）
- 网格尺寸：200×200，每格 0.512m（约 50cm）
- BEV 实际是 2D + 高度锚点（pillar 结构）

### 7.2 Occupancy 空间配置

```python
# configuration_perception.py:90-93
NUSCENES_OCC_PC_RANGE = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
NUSCENES_OCC_VOXEL_SIZE = (80/200, 80/200, 6.4)  # (0.4, 0.4, 0.4)
OCC_PILLAR_H = 16                                 # Z 方向体素数
```

- 覆盖范围：80m × 80m × 6.4m
- 体素网格：200×200×16，每格 0.4m
- 从 BEV 的 200×200 垂直扩展到 200×200×16

### 7.3 注意力配置

```python
# configuration_perception.py:114-117
POINTS_IN_PILLAR = 4        # 每个 BEV 位置沿 Z 的参考点数
SCA_NUM_POINTS = 8          # 空间交叉注意力采样点数
TSA_NUM_POINTS = 4          # 时间自注意力采样点数
DECODER_NUM_POINTS = 4      # 检测解码器采样点数
```

### 7.4 图像配置

```python
# configuration_perception.py:98-99, 120
FRUSTUM_RANGE = (0, 0, 1.0, 896, 512, 60.0)  # (x,y,z_min, x,y,z_max) UVTR 视锥
FRUSTUM_SIZE = (16.0, 16.0, 0.5)              # (dx, dy, dz) 视锥网格
IMAGE_SIZE = (896, 512)                       # 固定输入分辨率（宽×高）
```

---

## 8. 推理流程完整追踪

### 8.1 端到端数据流

```
1. 多相机图像 (6-8 cameras, 896×512)
        │
        ▼
2. Qwen3.5 VLM 前向传播
        ├─→ LLM hidden states (N_cam, H/2, W/2, 2560)
        └─→ ViT patches (N_cam, H, W, 1024)
        │
        ▼
3. SimpleFPN 特征金字塔
        ├─→ LLM 流: 4 尺度 {(H/8,W/8), (H/4,W/4), (H/2,W/2), (H,W)}
        └─→ ViT 流: 1 尺度 (H,W)
        │
        ▼
4. UVTR 深度估计 + 体素池化
        └─→ 体素特征 (B, C, 16, 200, 200)
        │
        ▼
5. BEV Queries 初始化 (200×200)
        └─→ 叠加 UVTR BEV 投影
        │
        ▼
6. BEVFormer Encoder (6 层) ◄─────── 核心
        │
        ├─→ 时间自注意力 (单帧退化)
        └─→ 空间交叉注意力 (3D→2D 投影到图像)
        │
        ▼
7. BEV 特征 (B, 200×200, 256)
        │
        ├────────────────────┬────────────────────┐
        ▼                    ▼                    ▼
    检测解码器            Occupancy 分支       地图分割
    (900 queries)         (3D U-Net)          (ResNet18)
        │                    │                    │
        ▼                    ▼                    ▼
    3D 框 (300)          体素 (200×200×16)    地图 (400×200)
```

### 8.2 关键文件调用链

```
QwenDrivePerception.infer (modeling_perception.py:227)
  └─→ VLM forward + hook capture (modeling_perception.py:250)
      └─→ BEVFormerModelV2.forward (modeling_perception.py:124)
          ├─→ SimpleFPN (fpn.py:18)
          ├─→ DepthNet + Uni3DVoxelPoolDepth (view_transform.py:79, 103)
          └─→ BEVFormerHead.forward (heads.py:230)
              └─→ PerceptionTransformer.forward (perception_transformer.py:197)
                  ├─→ BEVFormerEncoder.forward (bev_encoder.py:288)
                  │   └─→ BEVFormerLayer × 6 (bev_encoder.py:91)
                  │       ├─→ TemporalSelfAttention (attention.py:71)
                  │       └─→ SpatialCrossAttention (attention.py:211)
                  │           └─→ point_sampling (bev_encoder.py:231)
                  ├─→ DetectionTransformerDecoder (bev_encoder.py:366)
                  └─→ OccVoxelUNetRefiner (occ_refiner.py:61)
```

---

## 9. 与原版 BEVFormer 的差异总结

### 9.1 保留的核心组件 ✅

| 组件 | 实现文件 | 行号 |
|------|---------|------|
| BEV Queries (可学习嵌入) | `heads.py` | 222 |
| 可学习位置编码 | `layers.py` | 189-214 |
| BEVFormer Encoder (6 层) | `bev_encoder.py` | 185-347 |
| 时间自注意力 (TSA) | `attention.py` | 71-147 |
| 空间交叉注意力 (SCA) | `attention.py` | 211-285 |
| 3D→2D 投影 (point_sampling) | `bev_encoder.py` | 231-286 |
| 可变形注意力 | `attention.py` | 48-68, 224-242 |
| Deformable DETR 检测头 | `bev_encoder.py` | 366-419 |
| 无 NMS 解码 | `heads.py` | 53-86 |

### 9.2 关键改造 🔧

| 改造点 | 原版 | Qwen-Drive | 影响 |
|--------|------|-----------|------|
| **特征骨干** | ResNet-101-DCN | Qwen3.5 VLM 双流 | 统一 VLM 框架 |
| **特征维度** | CNN 256 维 | LLM 2560 → 256, ViT 1024 → 256 | 需要 SimpleFPN 适配 |
| **时序策略** | 支持多帧历史 | 单帧推理（TSA 退化） | 简化部署 |
| **多任务** | 检测 + 地图 | + occupancy 预测 | 3D U-Net 扩展 |
| **UVTR 集成** | 无 | 显式深度 + 体素池化 | 几何先验增强 |

### 9.3 工程优化 ⚡

1. **BFloat16 CUDA 内核**：`ops/ms_deform_attn_bf16.cu`，加速可变形注意力
2. **相机感知 rebatch**：`attention.py:251-273`，向量化处理不同相机的有效 query
3. **固定分辨率**：896×512，标定信息融入 `lidar2img`，无需运行时缩放
4. **权重复用**：VLM 冻结，只训练感知头

---

## 十、性能与限制

### 10.1 计算开销

| 模块 | FLOPs (估算) | 瓶颈 |
|------|-------------|------|
| VLM 前向 | ~80% | Qwen3.5-4B 推理 |
| BEV 编码器 | ~15% | 空间交叉注意力（多相机投影） |
| 检测/Occ/地图 | ~5% | 可变形注意力 |

### 10.2 限制

1. ⚠️ **固定相机配置**：训练时的相机数量、朝向、内参需一致
2. ⚠️ **固定分辨率**：896×512，改变分辨率需重新标定
3. ⚠️ **单帧推理**：未利用时序信息，速度估计弱于多帧版本
4. ⚠️ **BEV 压缩**：高度信息压缩到 pillar，垂直细节丢失

---

## 十一、总结

**Qwen-Drive-1.0 是一个工程化的 BEVFormer 实现，核心创新在于：**

1. ✅ **完整保留 BEVFormer 的时空注意力机制**（6 层编码器 + TSA/SCA）
2. 🔧 **用 VLM 双流特征替换 CNN**，实现感知与 VLM 的统一
3. 🔧 **单帧推理简化**，TSA 退化为普通自注意力，适合实时部署
4. 🔧 **多任务扩展**，在统一 BEV 表示上实现检测 + occupancy + 地图
5. ⚡ **UVTR 几何增强**，显式深度估计提供额外监督

**技术血脉**：BEVFormer (encoder) + DETR3D (decoder) + UVTR (view transform) + VLM (backbone)

相比原版 BEVFormer，Qwen-Drive 更像一个**感知 SDK**，通过统一的 BEV 特征支持多任务，
并把传统 CNN 升级到 VLM，为后续与语言、规划的联合训练铺平道路。

---

## 附录：关键文件索引

| 文件 | 作用 | 核心类/函数 |
|------|------|-----------|
| `modeling_perception.py` | 顶层模型 | `QwenDrivePerception`, `BEVFormerModelV2` |
| `bev_encoder.py` | BEV 编码器 | `BEVFormerEncoder`, `BEVFormerLayer`, `point_sampling` |
| `attention.py` | 注意力模块 | `TemporalSelfAttention`, `SpatialCrossAttention` |
| `perception_transformer.py` | Transformer 主体 | `PerceptionTransformer` (集成 encoder/decoder/occ) |
| `heads.py` | 任务头 | `BEVFormerHead`, `NMSFreeCoder` |
| `view_transform.py` | UVTR 视图转换 | `DepthNet`, `Uni3DVoxelPoolDepth` |
| `occ_refiner.py` | Occupancy 精细化 | `OccVoxelUNetRefiner` (3D U-Net) |
| `fpn.py` | 特征金字塔 | `SimpleFPN` (适配 VLM 特征) |
| `geometry.py` | 坐标变换 | `build_lidar2img`, `ego_to_lidar_boxes` |
| `configuration_perception.py` | 配置常量 | 所有超参数 |

---

*文档整理：Qwen-Drive-1.0 感知头源码分析·BEVFormer 实践*  
*相关文档：[bevformer.md](./bevformer.md)、[detr3d_qwen_drive_1_0.md](./detr3d_qwen_drive_1_0.md)、[occ_summary.md](./occ_summary.md)*
