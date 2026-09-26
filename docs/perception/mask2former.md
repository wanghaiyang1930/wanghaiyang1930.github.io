# Mask2Former：Masked Attention、多尺度与小目标 BEV 迁移

> 本文配合 `maskformer.md` 阅读，重点解释 Mask2Former 相对 MaskFormer 的变化，以及如何适配“框状车位 + 很小的停车限位器”。
>
> 论文：*Masked-attention Mask Transformer for Universal Image Segmentation*，CVPR 2022。文末列出官方源码入口；车位角点 head、限位器分支及采样改造属于工程建议，不是官方默认模型。

## 1. 先给结论：它改进了什么

Mask2Former 保留了 MaskFormer 的基本输出形式：每个 query 预测类别及一张二值 mask。主要变化发生在“query 如何读空间特征”和“mask 如何训练”，而不是取消动态 mask head。[M1][M2]

| 维度 | MaskFormer | Mask2Former |
|---|---|---|
| 基本输出 | 类别 + mask | 仍是类别 + mask |
| Query cross-attention | 主要读取单尺度特征，全局注意力 | 根据预测 mask 限制读取区域 |
| 多尺度 | Pixel decoder 融合多尺度 | Query decoder 还逐层轮换读取多个尺度 |
| 初始 query content | 官方实现为零内容 + 可学习 query embedding | 可学习 content + 可学习 query embedding |
| Decoder 顺序 | Self-attention → cross-attention → FFN | Masked cross-attention → self-attention → FFN |
| Mask 损失 | Focal + Dice | 采样点上的 sigmoid BCE + Dice |
| 匹配采样 | 原实现采用空间 mask 代价 | 同一图共享随机点计算 mask 匹配代价 |
| Loss 采样 | 原实现对齐分辨率后计算 | 不确定性采样 + 随机点 |

MaskFormer 对照依据见 `maskformer.md`；Mask2Former 依据 [M1]—[M4]。表中不意味着所有后续衍生实现和训练配置都必须完全相同。

**对当前任务，最值得学习的是高分辨率特征、多尺度读取和迭代区域细化；但很小的限位器恰好也是随机采样与硬 attention mask 需要谨慎处理的对象。**后一判断是针对业务的迁移风险分析，不是原论文对该目标的实验结论。

## 2. 完整结构

```text
图像 backbone / 已有 BEV encoder
                  ↓
              多尺度特征
                  ↓
              Pixel Decoder
                  ├─ attention feature level 0
                  ├─ attention feature level 1
                  ├─ attention feature level 2
                  └─ 高分辨率 mask features ─────────────┐
                                                       │
可学习 query content + query positional embeddings     │
                  ↓                                    │
初始类别 / mask 预测 ←─────────────────────────────────┤
                  ↓                                    │
            Masked Cross-Attention                      │
                  ↓                                    │
             Self-Attention                             │
                  ↓                                    │
                 FFN                                   │
                  ↓                                    │
        分类 Linear + mask embedding MLP ──通道点积─────┘
                  ↓
       本层 mask，同时生成下一层 attention mask
```

官方实现中，初始 query 在进入第一层 decoder 前就产生一次预测，之后每层继续更新。这样第一次 masked attention 也有可用的初始 mask，而不需要先运行完整的第一层再开始约束。[M2]

## 3. Pixel Decoder：保留细节并提供多尺度 memory

### 3.1 它输出的不是最终类别图

Pixel decoder 将 backbone 特征融合成适合 query 读取的多尺度表示，以及用于最终 mask 点积的高分辨率 embedding。它自身的空间特征不绑定最终某个实例。[M5]

官方常用 `MSDeformAttnPixelDecoder` 通过多尺度可变形注意力融合特征，再结合 FPN 式路径提供更高分辨率 mask features。典型图像配置中，attention 特征对应 stride 8/16/32，mask features 对应 stride 4；具体取值仍由配置和 backbone 决定。[M5]

### 3.2 两种 attention 不要混淆

- Pixel decoder 的 **deformable attention**：通过采样参考位置附近的特征完成多尺度融合；
- Query decoder 的 **masked attention**：用当前 query 的预测 mask 限制可访问的空间位置。

二者位于不同模块，机制不同，不能把 masked attention 解释成“在 box 内只采几个点”。[M2][M5]

### 3.3 BEV 不能机械照抄图像 stride

BEV 输入本身可能已经经过投影和下采样。应计算最终 mask feature 的实际米/格，而不是只看 `stride=4` 这个相对数字。

迁移时可以使用更简单的 BEV FPN 作为第一版 pixel decoder；这属于简化变体，不应声称已经完整复现官方 Mask2Former。

## 4. Masked Attention 的实现细节

### 4.1 普通 cross-attention

每个 query 可以读取某尺度上所有空间位置。即使当前已经大致定位对象，后续层仍可能聚合大量不相关区域。[M1]

### 4.2 从预测 mask 构造 attention 限制

官方代码的流程可概括为：[M2]

```text
当前 query 的 mask logits
       ↓ resize 到下一层 attention 特征分辨率
      sigmoid
       ↓
低于 0.5 的位置设为不可访问
       ↓
布尔 attention mask，并 detach
       ↓
用于下一层 cross-attention
```

这里的 `True` 表示禁止访问，不是“保留前景”。`detach` 意味着不通过硬阈值产生的区域门控传播梯度；mask head 仍通过自己的分割 loss 学习，共享网络也有其他可微路径。

### 4.3 一层 decoder 的执行顺序

```text
query + 当前空间特征 + attention mask
                   ↓
          masked cross-attention
                   ↓
           query self-attention
                   ↓
                  FFN
                   ↓
          分类预测 + mask 预测
```

先读取图像，再让 queries 交换信息，是它相对原始 MaskFormer decoder 的一处调整。[M2]

### 4.4 全部位置都被屏蔽怎么办

如果某个 query 在整个 attention map 上都被屏蔽，官方代码会解除该行限制，使其可以重新全局读取，避免无位置可访问。[M2]

但这只是“全空 mask”的保护。**如果误预测了某个非空背景区域，却没覆盖小目标，就不会因为漏掉真实目标而自动触发全局恢复。**这是从代码逻辑推导出的风险，对限位器迁移需要单独验证。

### 4.5 Masked attention 不等于按前景面积线性省算力

官方 query decoder 将布尔 mask 传给常规多头注意力，并非显式将前景 token 压缩成一个稀疏序列。因此不能简单声称“前景只占 1%，attention FLOPs 就降到 1%”。[M2]

其主要意义是改变读取范围和学习行为；实际显存与延迟要测量。

## 5. 多尺度 query 解码：每层轮换，不是每层拼接全部尺度

官方 decoder 维护三个特征尺度，每层按 `layer_index % num_feature_levels` 选择一个尺度，后续继续循环。每层更新后的 mask 会 resize 到下一层所需尺度。[M2]

```text
初始 mask
    ↓
第 1 层：读取尺度 0 → 新 mask
    ↓
第 2 层：读取尺度 1 → 新 mask
    ↓
第 3 层：读取尺度 2 → 新 mask
    ↓
第 4 层：重新读取尺度 0
```

该设计使 query 在不同分辨率下获得上下文和细节，而不是始终只看最粗特征。对 BEV 的具体收益仍取决于小目标是否在较细特征中保留下来，不能靠多尺度模块恢复已在前端丢失的信息。

## 6. Head：仍然是类别与动态 mask

### 6.1 类别输出

```text
query features → Linear
class_logits: [batch, num_queries, num_classes + 1]
```

最后一类是 no-object。它是无效 query 标签，不是 free-space、unknown 或遮挡状态。[M2][M3]

### 6.2 Mask 输出

```text
query features → 三层 MLP → mask embedding
mask embedding × pixel features → mask logits
```

```python
mask_logits = torch.einsum(
    "bqc,bchw->bqhw",
    mask_embedding,
    mask_features,
)
```

每个 query 一张二值 mask，不需要先裁 ROI，也不需要核心 head 回归 box。分类和 mask 两个预测共享 query 表示。[M2]

### 6.3 Query 数量不是类别数

实例任务中同一类可以有很多 query；语义任务中 GT 可按类别合并。Query 是否跨帧对应同一实例，需要额外关联设计，不能直接把 query 编号当 ID。[M1][M4]

## 7. 监督：采样匹配与采样 loss 是两件事

### 7.1 Hungarian matching

以类别、mask BCE、Dice 构建代价，再求一对一匹配：[M4]

```text
matching_cost = class_weight * negative_class_probability
              + mask_weight * sampled_mask_bce
              + dice_weight * sampled_mask_dice
```

匹配不是额外 loss；纯分割路径不要求 box cost。

### 7.2 匹配阶段：全图共享随机点

对同一张图，matcher 随机产生一组空间点，让所有预测 mask 和所有 GT mask 在这组相同坐标上取值，再计算候选之间的代价。[M4]

这样不同 query-GT 对的代价建立在相同空间位置上。**此处不是为每个 query 单独做不确定性采样。**

### 7.3 Loss 阶段：不确定点 + 随机点

匹配完成后，只对匹配 mask 采样并计算分割损失。官方 criterion 使用预测 logit 的负绝对值衡量不确定性：[M3]

```text
uncertainty = -abs(mask_logit)
```

Logit 越接近 0，sigmoid 越接近 0.5，位置越不确定。采样流程为先随机过采样，选择部分高不确定位置，再补充随机位置；预测与 GT 在同一批点上插值取值。

这些点不等于 GT 边界，也不保证涵盖所有小物体。采样点的选择不参与梯度传播，采样位置的预测 logits 仍用于可微 loss。[M3]

### 7.4 最终监督项

| 输出 | 监督 |
|---|---|
| 匹配 query 的类别 | 正确类别 CE |
| 未匹配 query 的类别 | 降权的 no-object CE |
| 匹配 mask | 采样点上的 sigmoid BCE + Dice |
| 未匹配 mask | 不直接强制为全零 |
| 初始预测 / 中间层预测 | 配置启用时的辅助监督 |

官方实现对辅助输出也重新执行 matching。不要强行假设一个 query 必须从初始层到最终层始终匹配同一个 GT。[M3]

### 7.5 点采样到底节省了什么

它降低匹配与 mask loss 对稠密空间位置的计算/存储需求，但模型依然要产生 mask 特征和 query mask。不能把“loss 只采若干点”理解成整个网络不再生成稠密 mask。[M2][M3][M4]

## 8. 你的场景：车位 + 很小的停车限位器

以下是针对业务的定制建议，不是 Mask2Former 原始模块或已验证收益。

### 8.1 建议结构：共享 BEV，按输出需求拆 head

```text
共享 BEV encoder + 时序融合
             ↓
       多尺度 BEV features
             ├─ slot queries
             │      ├─ 存在性 / 类型
             │      ├─ 有序四角点 / 入口
             │      └─ 可选：车位内部区域 mask
             │
             └─ 高分辨率限位器分支
                    ├─ 方案 A：stopper queries + 实例 mask
                    └─ 方案 B：中心点热力图 + 偏移 / 尺寸 / 方向
```

A、B 是待对照的方案，不建议第一版必须同时部署。两类目标可以共享 pixel features，但可分别设置 query 数量、匹配和分辨率；不必强迫使用同一种输出表示。

### 8.2 车位的主输出应考虑几何，而不只是区域 IoU

建议让一个 slot query 对应一个车位，回归有序四角点，并从固定语义的两个入口角点确定入口边。角点顺序、左右定义、镜像变换规则必须统一。

可选 mask 用四边形内部填充作为几何辅助；不要将它解释为“全部可见地面”，也不要把生成的四条边全部当成真实车位标线。只有标线标注才能监督可见油漆线。

推荐的匹配/损失结构示意：

```text
slot matching cost:
    分类代价 + 有序角点距离 + 可选 mask 代价

matched slot loss:
    分类 CE + 角点 L1/Smooth L1 + 可选 mask BCE/Dice
```

同一匹配关系同时监督 corner 与 mask。可以试验边长、入口方向或非自交等软几何约束，但不要将所有斜车位强制成理想矩形。

若 GT 仅有入口两点，不应伪造另外两角为精确 GT；可以先训练入口端点和方向，再按有依据的几何模型表示深度。车位结构存在也不代表一定空闲，是否可停需要独立的占用/障碍物判断。

### 8.3 限位器首先是空间采样问题

先收集真实尺寸与投影后的网格覆盖分布，尤其关注短边：

```text
short_side_cells = short_side_m / (bev_resolution_m * effective_mask_stride)
```

纯假设示例：短边 0.12 m、BEV 网格 0.05 m、mask feature stride 4，则短边只占 0.6 个特征格；如果保持 stride 1，则约 2.4 格。

这不是可检测性的硬阈值，但提示了一个优先级：先保证前端 BEV 表征、目标投影和 mask features 中还保留足够信息，再讨论增加 query decoder 层数。单纯把粗 logits 插值放大不是恢复细节。

可以比较：高分辨率近场 ROI、细尺度 skip features，或直接在图像高分辨率特征上辅助定位再融合到 BEV。限位器有高度，若前端采用纯地面 IPM，还需要检查其投影误差，不要将几何误差全部归因于分割 head。

### 8.4 小目标可能被随机匹配采样漏掉

设 GT 在有效区域占比为 `foreground_fraction`，均匀独立采样 `num_points` 个点。理想化估计：

```text
expected_positive_points = num_points * foreground_fraction
probability_of_no_positive_points = (1 - foreground_fraction) ** num_points
```

这是对随机点策略的数学风险分析，不是实际 recall 公式；插值、采样域与分布会影响真实情况。GT 很小时，点采样的 mask 代价可能无法充分区分候选。

可评估的定制改造包括：

- 在同图共享随机点基础上，加入来自各 GT 的前景/边界点，再用这套共享点比较所有候选；
- 对小目标匹配增加中心点或有向框几何项，避免只依赖稀疏 mask 样本；
- 在紧凑的高分辨率局部区域计算更密集 loss；
- 对小目标使用显式的前景、边界、背景分层采样。

这些做法改变了采样分布，必须保留负例覆盖、调整归一化，并与未改造基线对照。GT 引导采样只在训练的 matching/loss 中使用，不能将 GT 位置当成推理可得的 attention 提示。

### 8.5 Masked attention 可能过早排除小目标

初始 query 的 mask 若只覆盖错误背景，后续可能持续读取错误区域。全空恢复不能覆盖这种“非空但错位”的情况。

可消融：

1. 前若干层全局读取，后续层启用 masked attention；
2. 对 attention 允许区域适当膨胀，减轻离散网格误差；
3. 保留部分全局读取层，或融合高分辨率局部特征。

这些都不是原始默认 Mask2Former。区域膨胀不能扩大 GT 目标本身，attention 读取范围与最终 mask 标注是两个概念。

### 8.6 不一定要让限位器使用 mask

| 下游需求 | 优先比较的表示 |
|---|---|
| 只要是否存在及中心位置 | 中心点热力图或 query 中心回归 |
| 需要长度、宽度、朝向 | 有向框 / 中心 + 尺寸 + 方向 |
| 需要占据区域、轮廓 | 高分辨率实例 mask，必要时辅助几何回归 |

中心点分支仍然依赖可辨认的输入特征，不能解决信息已消失的问题。它也需要相应的峰值选择/重复结果处理，不能把 query 方案的无 NMS 描述自动套到中心点检测器。

### 8.7 不要靠“车位里应该有”来补出限位器

车位与限位器可建立后续关联，但第一版建议独立检测，再基于几何关系关联。不能因为检测到车位就默认存在限位器，否则容易将没有、被遮挡或未标注的目标混为一谈。

遮挡/unknown、已知不存在与已知可见应有明确的标注规则。语义负例、query no-object 和对象可见性不是同一标签。

## 9. 推理与输出

### 9.1 标准语义输出

使用每个 query 的类别概率和 mask 概率相乘求和，形成类别分数图。与 MaskFormer 相同，结果不是自动归一化的类别概率；多层 BEV 语义也不应直接强制互斥 argmax。[M6]

### 9.2 标准实例输出

官方实例推理包含类别候选选择和 mask 质量分数计算。质量可以由预测 mask 内部的前景概率统计得到，不应将其称为另一个已经训练的 SAM 式 IoU head。[M6]

车位定制 head 则输出类别/分数与有序角点，限位器输出选定的 mask 或几何表示。是否需要几何去重、轨迹融合及生命周期管理，应在具体系统中另行设计。

## 10. 验证计划：先定位问题来源

| 对照实验 | 要回答的问题 |
|---|---|
| 原 dense head / 中心点 head | 不改 Transformer 是否已满足限位器需求 |
| MaskFormer 风格 query head | 区域集合预测是否有收益 |
| 加多尺度读取 | 细尺度信息是否改善小目标 |
| 加 masked attention | 区域约束是帮助还是压低召回 |
| 默认采样 vs 小目标采样改造 | GT 是否被匹配/loss 点覆盖 |
| 车位纯 mask vs corner + mask | 角点与入口误差是否改善 |
| 近场高分辨率 vs 仅增加 decoder | 瓶颈是表征分辨率还是推理能力 |

除常规 IoU，建议记录：

- 车位：实例召回、入口端点误差、四角点误差、方向误差、错误入口、重复车位；
- 限位器：固定误报水平下的召回、中心/轮廓误差、按距离与短边网格数分组的漏检；
- 两者：遮挡、磨损、反光、边界截断场景，以及延迟和显存；
- 数据划分：按停车场或序列隔离训练/验证，避免相邻帧泄漏造成虚高结果。

评估使用实际米制坐标和业务阈值，不只用大面积车位 mask 的平均 IoU 掩盖限位器漏检。

## 11. 最终选型建议

针对当前需求，我建议把方案理解为：

> 共享 BEV 特征；车位采用结构化 query 几何预测，mask 可辅助；限位器优先保证高分辨率与小目标监督，再比较 query-mask 和中心点/有向框 head。

Mask2Former 提供的是多尺度读取、区域约束和集合监督工具，不是必须同时用于两个目标的完整模板。它是否优于更简单方案，需要按这两类目标分别验证。

## 12. 源码与论文索引

官方仓库：`facebookresearch/Mask2Former`。路径相对于仓库根目录；核对日期为 2026-09-25，复现需锁定 commit 与配置。

| 标记 | 来源 | 重点 |
|---|---|---|
| M1 | 论文 *Masked-attention Mask Transformer for Universal Image Segmentation*，arXiv `2112.01527` | 架构和设计动机 |
| M2 | `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py` | Learnable queries、masked attention、尺度轮换、初始预测 |
| M3 | `mask2former/modeling/criterion.py` | 不确定性采样、CE/BCE/Dice、辅助输出 |
| M4 | `mask2former/modeling/matcher.py` | 同图共享随机点、Hungarian matching |
| M5 | `mask2former/modeling/pixel_decoder/msdeformattn.py` | 多尺度可变形 pixel decoder、FPN 路径 |
| M6 | `mask2former/maskformer_model.py` | 语义、实例、全景推理 |
| M7 | `mask2former/config.py` | Query、采样、损失与模型选项 |

车位几何表示参考：HPS-Net 论文，arXiv `2310.11629`。配套本地文档：`maskformer.md`、`sam_detr.md`、`parking_slot_ref_hps_net.md`、`parking_slot_ref_dmpr_ps.md`。
