# MaskFormer：从逐像素分类到区域集合预测

> 本文面向 BEV segmentation 迁移，重点解释 MaskFormer 的 query、mask head、GT 分配与训练监督。配套阅读：`sam_detr.md` 和 `mask2former.md`。
>
> 论文：*Per-Pixel Classification is Not All You Need for Semantic Segmentation*，NeurIPS 2021。架构和训练事实以论文及官方实现为依据，文末列出源码索引；BEV 部分明确标为迁移建议，不是原论文实验结论。

## 1. 一句话理解

**MaskFormer 不直接让每个像素选择类别，而是预测一组“类别 + 二值区域 mask”，再按任务需要组合结果。**[MF1]

传统语义分割：

```text
图像 / BEV 特征
      ↓
固定类别分类 head
      ↓
[batch, num_classes, height, width]
```

MaskFormer：

```text
图像 / BEV 特征
      ↓
一组 query 读取特征
      ↓
每个 query 输出：这个区域是什么 + 这个区域覆盖哪里
      ↓
类别分数：[batch, num_queries, num_classes + 1]
区域 mask：[batch, num_queries, height, width]
```

这里的核心变化是从 **pixel classification** 转向 **mask classification**。输出槽位数量由 `num_queries` 决定，不要求等于类别数。[MF1][MF2]

## 2. 它是语义分割还是实例分割

MaskFormer 的区域表示可以服务不同任务，不能仅凭“有 query、有二值 mask”就判断它一定是实例分割。原论文主要展示统一语义分割和全景分割的能力。[MF1]

| 任务 | 一个 GT mask 的含义 | 举例 |
|---|---|---|
| 语义分割 | 某类别在图中的全部区域 | 所有车辆组成一张车辆 mask |
| 实例分割式目标组织 | 一个独立对象 | 每辆车各有一张 mask |
| 全景分割 | Thing 按实例，stuff 按语义区域组织 | 汽车分别输出，道路按对应区域输出 |

例如，语义 GT 有道路、车辆、人行横道三类：

```text
GT 1：道路类别 + 道路二值 mask
GT 2：车辆类别 + 所有车辆的联合 mask
GT 3：人行横道类别 + 所有人行横道的联合 mask
```

同一个 GT mask 可以包含多个不连通区域。这时即使用一对一 query 匹配，输出仍然是语义分割，不会凭空获得车辆实例身份。[MF6]

## 3. 整体结构：两个 decoder 各做什么

```text
输入图像或已有 BEV 特征
              ↓
        多尺度空间特征
              ├──────────────────────────┐
              ↓                          ↓
         Pixel Decoder          低分辨率特征 + 位置编码
              ↓                          ↑
      高分辨率 mask features         Learnable Queries
              │                          ↓
              │                 Transformer Decoder
              │                          ↓
              │                    query features
              │                     ├─ Linear → 类别
              │                     └─ MLP → mask embedding
              └──────────────────────────┘
                            ↓
                          通道点积
                            ↓
                     每个 query 的 mask
```

- **Pixel Decoder**：恢复高分辨率空间特征，负责形状与边界信息。
- **Transformer Decoder**：读取图像信息，为每个 query 形成区域相关表示。

二者不是重复模块：前者保留空间网格，后者维护固定数量的区域槽位。MaskFormer 使用 FPN 风格的像素解码设计，query decoder 则主要读取低分辨率图像特征。[MF1][MF2][MF7]

### 3.1 不要把输入特征与最终 mask 分辨率混为一谈

用于 query attention 的特征可以较粗；用于点积生成 mask 的特征可以较细。因此不必让每个 decoder 层都直接注意最高分辨率网格。[MF1]

对 BEV 的直观意义是：可以让 query 在较粗的 BEV 特征上聚合上下文，而从较细的 pixel features 生成边界。具体分辨率和上采样方式属于迁移设计。

## 4. Query 是如何工作的

### 4.1 Query 是输出槽位，不是类别或固定空间点

Query 不预先绑定某个类别，也不固定对应某个位置。一个 query 在不同输入上可以预测不同类别或 no-object；语义分割时，也不要求 query 0 永远对应道路。[MF1][MF2]

原始官方实现有一个容易忽略的细节：可学习的是 `query_embed`，作为 decoder 的 query positional embedding；初始 decoder content 使用全零张量。不能把后续 Mask2Former 的可学习 content query 设计直接写成 MaskFormer 的原始实现。[MF2]

### 4.2 Decoder 的主要运算

典型 decoder 层依次进行：[MF2]

1. Query self-attention：不同输出槽位交换信息；
2. Cross-attention：query 读取图像特征；
3. FFN：更新 query 表示；
4. 各步骤配合残差连接和归一化。

Cross-attention 可以理解为“这个区域槽位应该从哪些位置读取信息”。位置编码提供空间信息。Self-attention 并不天然保证不重复，输出去重的学习还依赖一对一监督。

## 5. Head 的精确形式

### 5.1 分类 head

```text
query_features: [batch, num_queries, hidden_dim]
        ↓ Linear
class_logits: [batch, num_queries, num_classes + 1]
```

最后一类是 no-object，即该 query 没有被分配有效区域。使用类别 softmax，包括 no-object 在内。[MF2][MF4]

**No-object 不是图像背景或 BEV free-space。**它描述 query 是否有效；free-space 若是业务类别，需要单独作为正常类别定义。

### 5.2 Mask head：三层 MLP + 点积

```text
query_features
        ↓ 三层 MLP
mask_embedding: [batch, num_queries, mask_dim]

pixel decoder
        ↓
mask_features: [batch, mask_dim, height, width]
```

沿通道做点积：[MF2]

```python
mask_logits = torch.einsum(
    "bqc,bchw->bqhw",
    mask_embedding,
    mask_features,
)
```

每个 query 的 mask embedding 相当于一组动态生成的通道分类权重。同一组像素特征配合不同 query，可以得到不同区域 mask。

它不是每个 query 输出 `height * width` 个独立 MLP 参数，也不是先检测 box、裁出 ROI、再调用一个独立分割器。[MF2]

### 5.3 为什么能表示复杂形状

虽然每个 query 最后只是一个向量，但不同网格的 pixel feature 不同，所以点积结果随位置变化。复杂形状来自空间特征与目标向量的组合，而不是向量本身存储全部轮廓点。[MF2]

### 5.4 与 SAM 的联系和区别

“目标向量 × 空间特征”的 mask 生成形式与 SAM 的动态解码思路相似，但不能认为两者训练方式相同：

- SAM 2 的提示指定一个对象，多个 mask 可以是同一提示的不同解释；
- MaskFormer 的 queries 预测区域集合，需要和 GT 区域集合匹配。

前者的候选选择不等于后者的 Hungarian assignment。SAM 部分可结合本目录 `sam_technical_solution.md` 阅读。

## 6. GT 分配：Hungarian matching

假设模型输出 100 个 query，但样本只有 3 个 GT 区域。不能硬性规定前 3 个 query 负责这 3 个 GT，因为输出集合没有固定顺序。[MF1][MF3]

### 6.1 先建立代价矩阵

```text
cost_matrix: [num_queries, num_gt_regions]

cost = class_weight * class_cost
     + focal_weight * mask_focal_cost
     + dice_weight * mask_dice_cost
```

原始 MaskFormer matcher 包含类别、Focal 和 Dice 代价；类别项使用 GT 类别的负预测概率。不要与 Mask2Former 的 BCE 匹配代价混淆。[MF3]

### 6.2 求一对一分配

```text
query 12 → GT 道路
query 38 → GT 车辆
query 71 → GT 人行横道
其余 query → unmatched
```

Matching 在无梯度上下文执行，决定后续监督分配，不是一个额外的可微 `L_match`。为完整覆盖 GT，query 数量需要足够容纳当前样本中的区域数量。[MF3]

## 7. 监督：哪些分支收到什么 loss

### 7.1 匹配 query

- 类别 CE：预测正确的区域类别；
- Mask sigmoid Focal：预测每个像素是否属于该区域；
- Mask Dice：优化区域重叠。

### 7.2 未匹配 query

- 仅分类到 no-object；
- 不需要把它们的所有 mask 强行监督为全零。

官方 criterion 通过 no-object 权重控制大量无效槽位的分类贡献，mask loss 只针对匹配项。[MF4]

### 7.3 损失组合

```text
loss = class_weight * class_ce
     + focal_weight * mask_focal
     + dice_weight * mask_dice
     + configured_auxiliary_losses
```

原始 criterion 在对齐预测与 GT 分辨率后，对匹配 mask 的空间位置计算分割损失；这里不是 Mask2Former 后来的不确定点采样方案。启用辅助监督时，中间层也产生类别和 mask 输出，官方 criterion 对辅助输出重新匹配并计算 loss。[MF2][MF4]

损失系数、归一化和 decoder 层数需要结合实验配置确认，不能把任一配置的数值当作所有实现的固定标准。

## 8. 推理：区域集合如何变成语义图

### 8.1 语义分割聚合

去掉 no-object 通道，将每个 query 的类别概率与 mask 概率相乘后求和：[MF5]

```python
class_probabilities = class_logits.softmax(dim=-1)[..., :-1]
mask_probabilities = mask_logits.sigmoid()
semantic_scores = torch.einsum(
    "bqk,bqhw->bkhw",
    class_probabilities,
    mask_probabilities,
)
```

得到 `[batch, num_classes, height, width]`。互斥类别可以取类别维 argmax；聚合分数不是自动归一化、校准后的概率。

这里不用先选择一个 query 再给每个像素赋类，也不需要对语义结果做 box NMS。[MF5]

### 8.2 全景分割

全景路径需要先过滤无效或低分候选，再处理 mask 之间的竞争、重叠和 stuff 合并等规则，最终形成一致的像素归属。不能把“类别概率 × mask 概率求和”当作所有任务统一的最终后处理。[MF5]

### 8.3 Anchor、box、NMS 的定位

本模型的基础 mask head 和 matcher 不依赖传统密集 anchor 或 box 回归。一对一分配不是 NMS；全景的像素归属处理也不等于传统 box NMS。集合预测不保证零重复或零重叠。[MF2][MF3][MF5]

## 9. 迁移到 BEV seg：建议保留什么

以下是基于架构的工程建议。

### 9.1 最小迁移版本

```text
现有 BEV encoder / 时序融合：保持不变
          ↓
BEV features
          ├─ 较粗特征 → query decoder
          └─ 较细特征 → pixel decoder → mask features
                                      ↑
query → 类别 head + mask embedding ────┘
```

先保留分类、mask、Hungarian matching，不增加文本、presence、box head 或 SAM memory。

### 9.2 先确认 GT，而不是先选 query 数

- 语义任务：按出现的类别构造 mask 集合；
- 实例任务：每个车辆、车位等对象独立构造 mask；
- 多层地图标签：允许不同语义图层重叠，不能末尾强制单标签 argmax；
- Unknown：与已知背景/free-space 分开，mask loss 和匹配代价都需要正确屏蔽。

官方图像数据路径不能直接当作已经支持 BEV 的有效性定义；尤其不能只修改训练 BCE/Focal，却遗漏匹配 Dice 中的无效网格。

### 9.3 适合用作什么实验

MaskFormer 适合作为 query mask head 的简明基线：比直接上完整 Mask2Former 更容易定位“集合预测本身”的收益。

建议对照：

1. 原有 dense segmentation head；
2. MaskFormer 风格 query head；
3. 后续增加 Mask2Former 的多尺度、masked attention 和点采样。

评估 mIoU、细结构质量、小目标、时延和显存。固定少类别的 BEV 语义任务中，query head 不一定优于成熟的卷积 head。

## 10. 针对车位与停车限位器的适配建议

用户场景是“框状车位 + 很小的停车限位器”。这不是简单的通用道路语义分割，建议先区分业务输出：车位更关心实例、角点和入口方向；限位器更关心物体是否存在、位置以及需要时的占据轮廓。

以下是定制方案建议，不是 MaskFormer 原有 head。

### 10.1 车位：不要把框状外观直接等同于轴对齐检测框

旋转或斜向车位需要表达四边形与入口语义。建议一个 slot query 对应一个车位实例，输出：

```text
slot query
    ├─ 存在性 / 类型
    ├─ 入口左、入口右、后侧右、后侧左：4 × 2 坐标
    ├─ 必要时的角点可见性 / 标注有效性
    └─ 可选：车位区域 mask
```

这里的“左/右”必须用统一的局部朝向定义，不能随图像旋转改变命名。例如按从入口朝向车位内部观察的左右定义，保持角点语义顺序一致。镜像增强也要更新角点角色。

按标注约定提取入口边，比仅输出一张矩形 mask 更容易保留“从哪边驶入”的语义。车位多边形表示有专门的研究，例如 HPS-Net；本文建议的 query + corner head 是迁移设计，并非直接复现其网络。

### 10.2 三种 mask GT 不要混用

| GT 类型 | 表达含义 | 限制 |
|---|---|---|
| 四角点填充的内部区域 | 车位几何范围 | 内部即使被车遮挡，也不等于观测到了地面 |
| 人工标注的可见地面标线 | 当前可见标线 | 不能默认知道完整车位范围和入口 |
| 由多边形生成的边界带 | 几何边界辅助监督 | 可能包含入口虚边等非真实标线 |

建议以有序角点作为结构化主输出，内部区域 mask 作为可选辅助。若业务只需要语义标线，则单独训练标线分支，不要把多边形边界自动当作真实油漆标线。

### 10.3 车位的匹配和监督

建议匹配代价包含分类与有序角点距离，可选择加入区域 mask 项。匹配后，对同一个 query 同时监督角点和 mask，避免两条独立匹配产生不同实例对应关系。

角点可用归一化坐标的 L1/Smooth L1；业务评估则恢复米制坐标，统计入口端点、角点和方向误差。不能仅依赖大面积内部 mask 的 IoU，因为较高 IoU 仍可能伴随不可接受的入口定位误差。

完整角点推断、仅可见角点检测和可见性预测是不同任务，应由 GT 定义决定。未知角点不能当作坐标零值监督，裁剪截断也不应直接伪造成新车位边界。

### 10.4 限位器：先核算短边还剩多少网格

令 BEV 网格大小为 `resolution_m`，限位器短边为 `short_side_m`，mask feature 相对原 BEV 网格下采样倍率为 `mask_stride`：

```text
short_side_cells = short_side_m / (resolution_m * mask_stride)
```

假设短边为 0.12 m、原 BEV 分辨率为 0.05 m、mask feature stride 为 4，则短边只有 0.6 个特征格。这只是计算示例，不是业务实测尺寸；如果这个量已经小于一个格，换 query decoder 很可能不是第一优先级。

优先比较：保留更细的 mask features、近场局部高分辨率 BEV，以及中心点 + 尺寸/方向的轻量检测 head。只输出位置/框时，未必需要 mask；需要占据轮廓时，再评估实例 mask。

### 10.5 两类目标共享什么、分开什么

建议共享 BEV encoder，并把车位 queries 与限位器 queries 或检测分支分开设置。这样可以分别选择输出几何、分辨率、匹配代价和 loss 权重，而不是要求框状大区域与极小物体使用完全相同的监督策略。

更详细的小目标采样与 masked attention 风险见 `mask2former.md` 第 8 节。

## 11. 它的不足，以及为什么要看 Mask2Former

原始 MaskFormer 的 query decoder 主要读取单尺度、较粗的特征；每层全局读取空间信息，缺少显式的预测区域约束；mask 监督也不是后续的高效不确定点采样路径。[MF1][MF2][MF4]

Mask2Former 沿这几个方向改进，而不是推翻“query 分类 + mask embedding 点积”的基础 head。详细见 `mask2former.md`。

## 12. 源码索引

官方仓库：`facebookresearch/MaskFormer`。以下路径相对于仓库根目录；核对日期为 2026-09-25，具体复现应锁定 commit 和配置。

| 标记 | 来源 | 重点 |
|---|---|---|
| MF1 | 论文 *Per-Pixel Classification is Not All You Need for Semantic Segmentation*，arXiv `2107.06278` | Mask classification、架构和任务定义 |
| MF2 | `mask_former/modeling/transformer/transformer_predictor.py`；`mask_former/modeling/transformer/transformer.py` | Query 初始化、decoder、MLP、点积 head |
| MF3 | `mask_former/modeling/matcher.py` | 类别、Focal、Dice 匹配代价 |
| MF4 | `mask_former/modeling/criterion.py` | CE、Focal、Dice、no-object、辅助监督 |
| MF5 | `mask_former/mask_former_model.py` | Semantic / panoptic inference |
| MF6 | `mask_former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py` | 按语义类别构造 GT masks |
| MF7 | `mask_former/modeling/heads/pixel_decoder.py` | Pixel decoder 与空间特征融合 |

车位结构化表示参考：*HPS-Net: Holistic Parking Slot Network Using Polygon-Shaped Representations*，arXiv `2310.11629`；本目录 `parking_slot_ref_hps_net.md`、`parking_slot_ref_dmpr_ps.md` 可作背景阅读，具体配方仍需以对应论文/源码为准。
