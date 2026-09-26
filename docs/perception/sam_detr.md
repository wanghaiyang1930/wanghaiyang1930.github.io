# DETR 风格分割与 BEV Seg 迁移方案

> 本文整理自 SAM 技术方案讨论，重点解释 query、Transformer decoder、动态 mask head、Hungarian matching，以及如何将这些机制迁移到 BEV segmentation。
>
> 核心建议：迁移“DETR 的集合预测思想 + MaskFormer/Mask2Former 的分割 head”，而不是完整照搬 SAM 3 的文本、presence、box 和视频记忆模块。以下 BEV 配置是实验建议，不是 SAM 3 官方架构或预训练配方。

## 1. 先明确迁移的目标

假设已有 BEV encoder，输出：

```text
bev_features: [batch, channels, height, width]
```

第一阶段保持相机/LiDAR 到 BEV 的特征生成、时序融合不变，只替换或增加 segmentation head。

Query-based mask head 既可以做实例分割，也可以做语义分割。关键区别不在于是否用了 Transformer，而在于：

1. GT 是按类别合并，还是按对象拆开；
2. 每个 query 被分配什么监督；
3. 最终保留独立 mask，还是聚合成语义类别图。

| 任务 | GT 组织方式 | 最终输出 |
|---|---|---|
| BEV 语义分割 | 同一类别的网格组成一张 mask | 每个网格的类别或多层语义分数 |
| BEV 实例分割 | 每个对象单独一张 mask | 每个对象的类别、分数、独立 mask |

只有 semantic GT、没有实例标注时，不能仅通过换成 query head，就期待模型学会可靠地区分同类的不同对象。

## 2. DETR 风格到底是什么

核心不是“用了 Transformer”，而是：

1. 使用一组可学习 query，预测一个无序目标集合；
2. 训练时通过一对一匹配，将 GT 分配给不同 query；
3. 未匹配 query 学习 no-object，减少重复和无效预测。

原始 DETR 输出类别与框；迁移到分割时，可以改为输出“类别 + mask”，不必保留框分支。

例如 BEV 中有三辆汽车，模型设置 100 个 query：

```text
100 个 query
    ↓
共同读取 BEV 特征
    ↓
query 7  → 汽车 A 的类别 + mask
query 25 → 汽车 B 的类别 + mask
query 63 → 汽车 C 的类别 + mask
其余 query → no-object
```

编号只是示意。Query 7 不预先绑定汽车 A，也不必固定负责左前方，更不自动等于跨帧 track ID。每张样本的 GT 分配由匹配过程决定。

### 2.1 与 SAM 2 / SAM 3 的关系

- SAM 2 的提示分割重点是“用户指定对象 → 输出该对象 mask”；多个候选可以是同一提示的不同解释。
- SAM 3 概念检测分支进一步使用实例 queries，发现概念对应的实例集合。
- 本文建议的 BEV 方案更接近固定类别的 MaskFormer/Mask2Former：保留集合预测和动态 mask head，不默认引入概念提示。

### 2.2 Anchor、box 与 NMS

本方案第一版可以不使用传统密集 anchor，也不使用 box head。每个 query 直接生成 mask，不要求先预测框、裁剪 ROI、再做分割。

这不意味着所有 DETR 变体都没有参考框：SAM 3 等实现可以使用 reference box 进行解码。本文选择的是纯 mask 集合预测方案。

一对一匹配使模型不必依赖传统 box NMS 才能工作，但不保证绝对没有重复实例。是否增加去重后处理，应由实例任务的验证结果决定；语义输出则主要通过 query 聚合得到。

## 3. 适合 BEV 的整体结构

```text
相机 / LiDAR
      ↓
现有 BEV Encoder + 时序融合
      ↓
BEV 特征
      ├───────────────────────────────┐
      ↓                               ↓
用于 attention 的多尺度特征       Pixel Decoder
      ↑                               ↓
可学习 queries                 高分辨率 mask features
      ↓                               │
Transformer Decoder                   │
      ↓                               │
更新后的 queries                      │
      ├─ Linear → 类别预测            │
      │                               │
      └─ MLP → mask embedding ────────┘
                         ↓
                       通道点积
                         ↓
                 每个 query 的 mask
```

### 3.1 两种 query 不要混淆

如果前端使用 BEVFormer，需要区分：

| Query | 含义 | 作用 |
|---|---|---|
| BEV grid query | BEV 空间网格位置 | 从相机和历史信息生成 BEV 表征 |
| Mask / object query | 待预测的区域或实例槽位 | 读取已有 BEV 表征，产生分割结果 |

它们不是同一组 query，数量和语义也不一样。添加 query segmentation head，不意味着必须重新设计 BEV grid query。

## 4. Query 如何读取 BEV 特征

### 4.1 初始化可学习向量

示例设置：

```text
queries: [batch, 100, 256]
```

100 是输出槽位数量，256 是特征维度，不是 100 个类别或 100 个预设框。这些向量可学习，但尚未读取当前图像时，并不代表已经定位好的对象。

### 4.2 Cross-attention：读取空间信息

将用于 attention 的 BEV 特征按空间位置展开：

```text
BEV:    [batch, 256, height, width]
memory: [batch, height * width, 256]
```

Query 与不同位置的特征计算相关性，聚合相关位置的信息。位置编码使模型能区分空间位置，而不只比较特征内容。

```text
query：我应该关注哪些 BEV 位置？
memory：这些位置有什么特征，位于哪里？
```

Decoder 同时通过 query self-attention 交换信息，再用 FFN 更新表示。具体模块顺序依架构而定；例如 Mask2Former 使用 cross-attention、self-attention、FFN 的顺序。

Self-attention 本身不等于去重约束，减少重复实例还依赖集合匹配和训练监督。

### 4.3 Masked attention：聚焦预测区域

Mask2Former 使用上一层预测 mask 限制下一层 cross-attention 的可访问区域：

```text
上一层 query 的 mask
          ↓
转换成 attention 区域限制
          ↓
读取相关区域特征
          ↓
更新 query 和 mask
```

这形成“粗分 → 聚焦 → 细化”的迭代过程。实现需要处理所有位置都被屏蔽的情况，避免 query 无位置可读。

迁移建议：先跑通普通 cross-attention，再评估 masked attention，避免第一版同时改变过多机制。

## 5. Head 如何输出类别和 mask

### 5.1 类别 head

设业务类别数为 `num_classes`：

```text
query features
      ↓ Linear
[batch, num_queries, num_classes + 1]
```

额外的 1 类是 no-object，表示这个 query 没有负责一个有效输出区域。

**No-object 不等于 BEV 的 background 或 free-space。**前者是输出槽位无效；后者若是业务需要预测的类别，应作为正常类别建模。未知或未标注区域也不能直接等同于 no-object。

### 5.2 动态 mask head

每个 query 经 MLP 得到通道权重，pixel decoder 提供具有空间分辨率的像素特征：

```text
mask_embedding: [batch, num_queries, mask_channels]
mask_features:  [batch, mask_channels, height, width]
```

沿通道点积：

```python
mask_logits = torch.einsum(
    "bqc,bchw->bqhw",
    mask_embedding,
    mask_features,
)
```

输出：

```text
mask_logits: [batch, num_queries, height, width]
```

可以将其理解为“query 动态生成一组 1×1 分类器权重”，再用这组权重判断每个 BEV 网格是否属于该区域：

- 像素特征负责位置和局部细节；
- query embedding 负责当前区域的选择；
- 每个 query 输出独立的前景/背景 logits；
- mask 通道是区域槽位，不是固定类别通道。

动态权重作用于所有位置，不意味着 mask 只能表达简单几何形状；空间变化来自各位置不同的 pixel features。

### 5.3 与传统 dense head 的对照

```text
传统语义 head：
BEV features → Conv → [batch, num_classes, height, width]

Query mask head：
BEV features → queries → 类别分数
                       + [batch, num_queries, height, width]
                                 ↓
                        按任务保留实例或聚合语义
```

Query 方案把“区域是什么”和“区域覆盖哪里”拆成两个预测，而不是直接对每个网格输出固定类别 logits。

## 6. BEV GT 如何组织

### 6.1 语义分割：按类别生成区域 mask

假设类别包括道路、车道线、人行横道、车辆区域：

```text
GT 1：类别 = 道路，mask = 所有道路网格
GT 2：类别 = 车道线，mask = 所有车道线网格
GT 3：类别 = 人行横道，mask = 所有人行横道网格
GT 4：类别 = 车辆，mask = 所有车辆网格
```

同类别不连通的区域也可以放在同一张 mask 中，不要求每张 mask 都是一个连通区域。MaskFormer/Mask2Former 的语义分割数据组织支持按出现的类别构造二值 mask。

此时 query 学到的是语义区域，使用 Hungarian matching 不会自动把任务变成实例分割。

### 6.2 实例分割：每个对象一张 mask

```text
GT 1：类别 = 车辆，mask = 车辆 A
GT 2：类别 = 车辆，mask = 车辆 B
GT 3：类别 = 车辆，mask = 车辆 C
```

匹配后不同 query 分别负责不同对象。相同类别可以有多个 GT，query 数量需要覆盖业务中的实例数量分布。

### 6.3 互斥语义与多层语义

BEV 标签可能有两种定义：

- 互斥标签：一个网格只属于一类；
- 多层标签：同一网格可以同时属于道路区域和车道线等不同图层。

必须先明确数据语义。多层标签不能在推理末尾强行执行单类别 argmax；GT 构造、聚合方式和阈值也要与多层定义一致。

## 7. Hungarian matching 与训练 loss

### 7.1 为什么要匹配

模型输出固定数量 query，但每张样本的 GT 数量不同，且 query 没有预设输出顺序。

Hungarian matching 根据当前预测，选择总代价较小的一对一分配：

```text
预测 query 集合 + GT mask 集合
              ↓
           代价矩阵
              ↓
      Hungarian assignment
              ↓
    确定每个 query 对应哪个 GT
```

### 7.2 匹配代价

纯 mask BEV 方案可从以下代价开始：

```text
matching_cost = class_weight * class_cost
              + mask_weight * mask_bce_cost
              + dice_weight * mask_dice_cost
```

- 分类代价：预测类别是否符合 GT；
- BCE 代价：网格级前景/背景是否符合 GT；
- Dice 代价：预测区域与 GT 的重叠是否良好。

Mask2Former 的匹配分类代价使用负类别概率，不必与训练时的 CE 完全同形。匹配代价的权重和最终 loss 权重也应作为两个配置概念区分。

### 7.3 匹配后的监督

```text
匹配到 GT 的 query：
    分类 CE
    mask BCE
    mask Dice

未匹配 query：
    no-object 分类 CE
```

通常不需要把未匹配 query 的 mask 强制监督成全零。Mask2Former 的 mask loss 只针对匹配项；no-object 分类有单独权重，避免大量未匹配 query 压倒有效目标监督。

总损失可概括为：

```text
loss = class_weight * class_ce
     + mask_weight * mask_bce
     + dice_weight * mask_dice
     + configured_auxiliary_losses
```

**Matching 是离散的监督分配步骤，不是再加一个可微的 `L_match`。**中间 decoder 层也可以增加辅助监督，而不只监督最终层。

### 7.4 BCE 与 Dice 的分工

- BCE 约束每个有效网格的前景/背景预测；
- Dice 约束整体区域重叠；
- 二者都不能替代 GT 质量或细结构所需的分辨率。

这里采用 Mask2Former 风格的 BCE + Dice，不要与 SAM 2 的 Focal + Dice 配置混成同一套默认配方。

## 8. 推理：如何得到 BEV 分割结果

### 8.1 语义输出

对类别 logits 做 softmax，去掉 no-object 通道；对 mask logits 做 sigmoid，再把每个 query 的类别概率与 mask 概率相乘并累加：

```python
class_probabilities = class_logits.softmax(dim=-1)[..., :-1]
mask_probabilities = mask_logits.sigmoid()
semantic_scores = torch.einsum(
    "bqk,bqhw->bkhw",
    class_probabilities,
    mask_probabilities,
)
```

得到：

```text
semantic_scores: [batch, num_classes, height, width]
```

解释：某 query 若较确信自己是道路，同时较确信某个网格属于其区域，就为该网格的道路分数贡献较大权重。

- 互斥类别可在类别维取 argmax；
- 多层标签需按图层定义设计输出与阈值；
- 聚合结果是分数，不应直接当成已归一化、已校准的概率。

### 8.2 实例输出

保留每个有效 query 的独立结果：

```text
类别 + 分数 + 实例 mask
```

根据类别分数和 mask 质量等策略选择候选。可先不加入 NMS，评估重复预测是否成为实际问题后，再决定是否需要去重策略。

Query 索引不是天然跨帧实例 ID；若需要输出轨迹，必须另行设计关联或时序 query 机制。

## 9. BEV 迁移的关键注意事项

以下是工程建议，不是官方默认配置。

### 9.1 Unknown 必须贯穿 matching 和 loss

区分已知负例/free-space 与未知/未标注区域。若某网格按业务定义是 unknown，应从相关匹配代价与 mask loss 中排除，而不是直接设为背景。

不能只在最终 BCE 中加入 `valid_mask`，却在匹配 BCE/Dice 中仍将 unknown 当负例：此时 GT 分配已经受错误信息影响。Dice 的分子、分母也要使用一致的有效区域定义。

### 9.2 Attention 分辨率与 mask 分辨率分开设计

建议：

```text
较粗 BEV 特征 → query attention
较细 BEV 特征 → 最终 mask 点积
```

控制 attention 成本，同时保留 mask 边界细节。对于车道线、路沿等细结构，先检查 GT 栅格化和下采样是否已经抹掉目标；最后的插值不能恢复不存在的输入细节。

若采用 mask loss 点采样，也要评估是否充分覆盖细小前景，而不能只根据显存收益选择采样策略。

### 9.3 保留 dense head 基线

固定类别较少、只需要语义栅格时，传统 `Conv + 上采样` 仍值得作为基线。Query head 是否更优，应由精度、细结构召回、延迟和显存共同决定。

### 9.4 不要同时引入所有 SAM 3 模块

| 模块 | 第一版建议 |
|---|---|
| 现有 BEV encoder、时序融合 | 保持不变 |
| BEV pixel decoder | 加入 |
| Query decoder | 加入 |
| 分类 + no-object head | 加入 |
| 动态 mask head | 加入 |
| Hungarian matching | 加入 |
| 文本 encoder / presence | 固定类别任务先不加入 |
| Box head / reference box | 纯 mask 任务先不加入 |
| SAM 视频 memory | 先不加入 |

## 10. 第一轮实验建议

1. 保留现有 BEV encoder 和时序融合，新增可切换 query-mask head。
2. 以 256 维 query、3 层 decoder 作为可调的实验起点。
3. 根据 GT 区域/实例数分布设置 query 数，不盲目照抄 100。
4. 使用分类 CE + mask BCE + Dice，正确处理 no-object 与 unknown。
5. 第一版不增加 box、anchor、NMS、文本和 presence。
6. 对照原有 dense head，记录精度、延迟和显存。
7. 在基本方案有效后，再评估多尺度、masked attention、辅助监督及采样策略的增益。

| 任务 | 重点评估 |
|---|---|
| BEV 语义分割 | mIoU、分类别 IoU、细结构质量、有效区域内误报漏报 |
| BEV 实例分割 | 实例级精度与召回、重复实例、密集区域分离、小目标 |
| 工程指标 | Head 额外延迟、峰值显存、输出数量与分辨率变化的成本 |

## 11. 总结

迁移的核心是：**将 BEV 分割表示为区域集合预测，而不是把 BEV 特征送入一个框检测器。**

- Query 读取 BEV 信息，形成区域相关表示；
- 分类 head 回答“这是什么区域”；
- 动态 mask head 回答“覆盖哪些网格”；
- Hungarian matching 决定“这个 query 学哪个 GT”；
- GT 按类别合并，得到语义分割；按对象拆开，得到实例分割；
- Anchor、box、NMS、文本、presence 都不是这版纯 mask 方案的必需模块。

## 12. 参考材料与源码入口

### 本地材料

- `docs/perception/sam2.md`
- `docs/perception/sam3.md`
- `docs/perception/sam_technical_solution.md`
- `docs/perception/maskformer.md`
- `docs/perception/mask2former.md`
- `docs/perception/bevformer.md`

### 论文

- DETR：*End-to-End Object Detection with Transformers*。
- MaskFormer：*Per-Pixel Classification is Not All You Need for Semantic Segmentation*。
- Mask2Former：*Masked-attention Mask Transformer for Universal Image Segmentation*。
- BEVFormer：*Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*。

### 官方实现

以下路径相对于对应官方仓库根目录。不同版本配置可能变化，复现时应锁定具体 commit。

| 仓库 | 路径 | 阅读重点 |
|---|---|---|
| `facebookresearch/detr` | `models/detr.py`、`models/matcher.py` | 集合预测、no-object、Hungarian matching |
| `facebookresearch/Mask2Former` | `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py` | Query decoder、masked attention、分类与 mask 点积 |
| `facebookresearch/Mask2Former` | `mask2former/modeling/matcher.py` | 分类与 mask 匹配代价 |
| `facebookresearch/Mask2Former` | `mask2former/modeling/criterion.py` | 分类、BCE、Dice、辅助监督 |
| `facebookresearch/Mask2Former` | `mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py` | 语义标签转类别 mask 集合 |
| `facebookresearch/Mask2Former` | `mask2former/maskformer_model.py` | 语义聚合和实例推理 |
