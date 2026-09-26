# SAM 技术方案拆解：Mask Head、监督与视频记忆

> 阅读起点：本目录的 `sam2.md`、`sam3.md`。本文在两份概览基础上，补充官方代码核对，重点解释“head 到底如何输出 mask”“GT 如何分配”“哪些分支收到什么梯度”。
>
> 核对日期：2026-09-25。代码依据为官方仓库当时可访问的 `main`，不是锁定 commit 的复现说明；具体配置、训练阶段和权重需成套使用。文末给出源码索引 [S1]—[S13]。本文不把 SAM 3.1 的新增机制混入 SAM 3 基础方案，也不声称公开微调配置等于原始预训练配方。

## 1. 先区分两条任务链路

| 问题 | SAM / SAM 2 实例提示分割 | SAM 3 概念检测分割 |
|---|---|---|
| 用户想做什么 | 指出一个对象，把它分出来 | 给出概念，找出符合概念的实例集合 |
| 提示 | 点、框、mask | 文本、视觉/几何提示等 |
| token/query 的含义 | 指定对象的候选分割解释 | 概念对应的不同实例槽位 |
| 多个 mask 的含义 | 同一个提示存在歧义，不一定是不同对象 | 不同 query 可以代表不同对象 |
| 监督分配 | 提示已绑定一个 GT 对象 | 先将预测 query 与 GT 实例匹配 |
| 核心 head | mask、IoU、对象可见性 | query score、box、presence、实例 mask |

SAM 2 的 mask decoder 延续 SAM 的 token 条件解码设计，增加高分辨率特征及视频相关分支。SAM 3 的**概念检测分支**不是把文本直接塞进同一个 SAM 2 mask decoder，而是包含独立的检测查询、框预测和像素解码路径。[S1][S4][S7][S8]

**最重要的共同点：mask 不是直接用固定类别卷积通道输出，而是让“目标相关向量”与“像素特征”做通道点积。不同点是，目标向量如何得到、代表什么、如何分配监督。**[S1][S8]

## 2. SAM 2：从输入到 mask 的完整张量链路

### 2.1 提示不是标签，GT mask 才是主要分割监督

`PromptEncoder` 将点坐标编码后，加上正点/负点类型 embedding；框的两个角点有不同类型 embedding。点和框形成稀疏 token。mask 提示通过卷积下采样形成稠密特征，加到图像特征上；无 mask 提示时使用可学习的 no-mask embedding。[S2]

例如“点在汽车车门上”只是输入条件，监督可以是整辆车的 GT mask。**训练并非只要求点击位置预测正确，而是要求完整 mask 正确。**这也解释了为什么交互训练仍需要像素级标注。[S2][S6]

### 2.2 准备图像特征与输出 token

以模型维度 256、输入 1024×1024、主特征 stride 16 的 SAM 2 配置为示例，batch 维以下用 `batch` 表示提示/对象批次，不一定等于独立图像数量：[S1][S4][S5]

```text
图像 -> Hiera + FPN
                ├─ stride 4 高分辨率特征
                ├─ stride 8 高分辨率特征
                └─ stride 16 主特征 [batch, 256, 64, 64]
                                      │
                       视频：先经过 Memory Attention
                                      │
                            加 dense prompt embedding
                                      │
                                Two-Way Transformer
```

默认 `num_multimask_outputs=3` 时，内部实际有 **4 个 mask token**：token 0 用于单 mask 路径，token 1—3 用于多候选路径。此外还有 1 个 IoU token；开启 `pred_obj_scores` 时再增加 1 个对象分数 token。它们与点/框 token 拼接后进入 Transformer。[S1]

### 2.3 Two-Way Transformer 如何交互

`TwoWayAttentionBlock` 按顺序执行：[S3]

1. token self-attention：输出 token 与提示 token 交换信息；
2. token → image cross-attention：token 从空间特征读取对象信息；
3. token MLP；
4. image → token cross-attention：图像特征反过来读取提示/输出 token；
5. 堆叠这些 block 后，再做一次 token → image attention。

因此，不仅 mask token 被提示条件化，**用于生成 mask 的空间特征也会被更新**。SAM 2 的构建代码使用两层 Two-Way Transformer；这里的“双向”指 token 与图像相互读写，不是视频前向/后向传播。[S3][S4]

### 2.4 真正的 mask head：Hypernetwork MLP + 点积

设 Transformer 输出的第 `candidate` 个 mask token 为 `token_candidate`，上采样像素特征为 `pixel_features`。源码的等价数学表达为：[S1]

```text
mask_weights = MLP_candidate(token_candidate)
mask_logit[candidate, row, col]
    = sum_channel(mask_weights[channel] * pixel_features[channel, row, col])
```

在上述示例维度中，两个 2 倍反卷积把特征变成 `[batch, 32, 256, 256]`；每个 mask token 有自己的三层 MLP，将 256 维 token 映射为 32 维权重。4 个权重向量与空间特征矩阵相乘，得到 `[batch, 4, 256, 256]` 的低分辨率 logits。[S1]

可以把它理解为**每次提示动态生成的 1×1 分类器**：空间分辨率由像素特征保留，目标选择由 token 生成的通道权重控制。32 维不是 32 个类别，也不表示只能描述 32 个像素。[S1]

高分辨率 skip feature 在两级上采样过程中相加，而不是最后才粘贴边界。低分辨率 logits 再插值到输出分辨率，得到最终 mask；插值本身没有新增真实视觉细节。[S1][S4]

### 2.5 三个容易混淆的输出

| 输出 | 从哪里产生 | 代表什么 |
|---|---|---|
| mask logits | mask token → MLP → 像素特征点积 | 每个像素是否属于指定对象 |
| IoU prediction | IoU token → MLP → 每候选一个值 | 候选 mask 预计与 GT 重叠得多好 |
| object score | 对象分数 token → Linear 或 MLP → 1 个 logit | 指定对象在当前帧是否可见 |

IoU 不是像素概率，也不是类别置信度；对象不可见也不等于“某个类别在整张图不存在”。未开启对象分数分支时，代码使用固定高 logit，不能将其视为真正训练的可见性判断。[S1]

`object pointer` 则来自选定 mask token 的投影，供后续记忆使用；它不是输出一个离散 ID 的分类 head。对象 ID 与 pointer 特征应区分开。[S4]

## 3. SAM 2：如何监督 head

### 3.1 每个对象、每帧、每轮交互都有对应 GT

训练样本可理解为视频帧、指定对象的逐帧二值 mask，以及由 GT/预测误差生成的提示。模型在当前提示下产生候选 mask，训练器累加多个帧和交互步骤的损失。[S5][S6]

官方公开的 SAM 2.1 MOSE 微调配置给出了一个具体例子：[S5]

```text
loss = 20 * mask_focal
     +  1 * mask_dice
     +  1 * iou_regression
     +  1 * object_classification

supervise_all_iou = true
iou_use_l1_loss = true
```

这些数值只代表该配置，不是所有 SAM 版本、所有训练阶段统一的常数。

### 3.2 像素监督：Focal + Dice

原概览将像素损失概括为 BCE 类损失；更精确地说，核对的 SAM 2 loss 实现使用 **sigmoid focal loss + Dice**。设像素预测概率为 `prob`、GT 为 `target`，可抽象为：[S5]

```text
prob_correct = target * prob + (1 - target) * (1 - prob)
focal = alpha_target * (1 - prob_correct)^gamma * BCE(prob, target)
dice = 1 - (2 * sum(prob * target) + smooth)
             / (sum(prob) + sum(target) + smooth)
```

Focal 降低易分类像素的权重；Dice 衡量区域重叠。二者不是额外的边界 loss，也不能保证小目标和细边界一定正确。[S5]

### 3.3 多候选怎么训：只对最佳候选回传分割损失

给定一个 GT mask，对所有候选分别计算加权 Focal + Dice，选择损失最小的候选，只让它接收该轮的 mask 分割损失。这是 **best-of-K / winner-takes-all** 式分配，不是多个对象之间的 Hungarian matching。[S5]

这样不必强迫所有候选都拟合同一张 GT，给不同粒度的解释留下空间。训练时按 GT loss 选候选；推理时没有 GT，靠预测 IoU 或单 mask 路径选择。[S1][S5]

### 3.4 IoU head 的标签从哪里来

将每个预测 logit 以 0 为阈值二值化，计算其与 GT 的实际 IoU，再回归该数值。公开实现支持 MSE 或 L1；上述配置使用 L1，并监督所有候选的 IoU，而不只监督最佳候选。[S5]

实际 IoU 由硬阈值后的 mask 算出，不通过它向 mask logits 传播可微分割梯度。因此要区别 **IoU 分数回归**和 **Dice 分割优化**；IoU 回归仍可通过共享网络影响公共特征。[S5]

### 3.5 遮挡/离屏时监督什么

对象分数的 GT 来自“当前对象 GT mask 是否有前景”。启用对象分数监督时，空 mask 对应不可见；该对象的 mask、Dice、IoU 损失被置零，保留对象分类监督。上述配置的对象分类 Focal 参数退化为普通 BCE。[S5]

这不等于要求模型恢复遮挡后的完整物体，也不能因为本帧不可见就自动删除轨迹。是否终止跟踪、何时重新提示属于额外的推理策略。

### 3.6 交互训练为什么有效

训练器先由 GT 生成初始点、框或 mask 条件，然后根据预测与 GT 的误差采样纠错点：漏分区域可提供正点，多分区域可提供负点。前一次低分辨率预测作为下一次 mask prompt，代码会对这一路反馈执行 `detach()`。[S6]

因此它模拟的是“初始提示 → 预测 → 纠错 → 再预测”，而不是仅训练一次理想框提示。视频中还选择 conditioning frame 与需要纠错的帧；并非把所有帧 GT 都当作推理可见输入。[S6]

## 4. SAM 2 视频：memory 如何学会跟踪

```text
历史对象 memory + pointer
            ↓
当前帧特征 -> Memory Attention -> 提示分割 head -> 当前 mask
                                                    ↓
                      图像特征 + mask -> Memory Encoder
                                                    ↓
                                             后续帧 memory
```

Memory Encoder 将图像特征与下采样 mask 融合；Memory Attention 让当前帧读取历史空间记忆和对象 pointer，并利用位置/时间信息。它们是内部状态模块，不是分别输出类别的 head。[S4]

**有跟踪能力，不代表必须有一个独立的 `L_track` 或 pointer ID 分类 loss。**核对的 SAM 2 criterion 直接计算 mask、Dice、IoU、对象分数，多帧预测对这些损失的优化训练了视频分割链路。跨帧 GT 身份用于组织同一对象的监督，不能据此声称 pointer 有直接 ID 标签。[S4][S5][S6]

## 5. SAM 3：概念检测分支的 head 有何不同

### 5.1 先纠正“统一 Hiera 架构”的误解

SAM 2 使用 Hiera；核对的 SAM 3 图像模型构建代码使用 ViT 视觉骨干、文本编码器、视觉语言融合、检测 decoder 和像素 decoder。不能将两者画成同一条 Hiera + Two-Way Transformer 主干。[S7]

```text
图像 -> ViT + neck -> 视觉特征 ───────┐
文本/几何提示 -> prompt 特征 ────────┤
                                    ↓
                         视觉语言融合 + 检测 decoder
                             ├─ 实例 query -> score
                             ├─ 实例 query -> box
                             ├─ presence token -> presence
                             └─ 实例 query -> mask embedding ──┐
视觉/融合特征 -> pixel decoder -> 像素 embedding ───────────────┤
                                                              ↓
                                                        实例 mask 点积
```

核对的默认构建函数使用 256 维模型空间、6 层检测 decoder、200 个查询，开启 box refinement 和 presence token；这是该实现配置，不是架构必须固定的查询数量。训练中的辅助查询分支也不应与推理查询数混为一谈。[S7][S9]

### 5.2 Score 与 box head

query 经过 prompt 条件化后，score 分支用 `DotProductScoring` 与 prompt 表示打分，而不是固定类别集合的多通道 softmax。box 分支预测 reference box 的修正量：[S7][S9]

```text
box = sigmoid(inverse_sigmoid(reference_box) + box_head(query))
```

框输出与 mask 输出共享实例 query，但 mask 不是必须经过“预测框裁图 → RoIAlign → 独立分割器”的级联。实际 mask head 接收 query 与全图像素特征，不直接把最终预测框当裁剪操作。[S8][S9]

### 5.3 Presence 不等于另一个 no-object 分类器

presence 表示“当前概念在图中有没有可见实例”；query score 表示候选实例与概念的匹配程度。推理 processor 会组合这两种概率进行过滤；应确认调用路径是否已在模型内部组合，避免重复乘算。[S9][S10]

用概率分解帮助理解：`最终分数 ≈ 概念存在概率 × 条件实例分数`。这是一种解释，不意味着输出已经做过业务场景概率校准。SAM 3 的 query 负样本可以通过 score 分支处理，**不应再凭空添加一个独立 no-object head**。[S9][S10][S11]

### 5.4 实例 mask head：仍然是点积，但 query 语义变了

`MaskPredictor` 用三层 MLP 生成每个 query 的 mask embedding，再与 pixel embedding 做点积：[S8]

```text
query_embedding: [batch, queries, channels]
pixel_embedding: [batch, channels, height, width]
mask_logits = einsum("bqc,bchw->bqhw", query_embedding, pixel_embedding)
```

Pixel Decoder 自顶向上融合多尺度特征，包含上采样、相加、卷积与归一化；`UniversalSegmentationHead` 还包含实例特征投影和单通道 semantic 分支。实例通道对应不同 query，而非 SAM 2 的同一对象歧义候选。[S8]

该概念分支没有理由默认再加一个与 SAM 2 完全相同的 IoU-token head。SAM 3 的交互/跟踪分支和概念检测分支需分别看待，不能仅凭都叫 SAM 就混用输出解释。[S7][S8][S9]

## 6. SAM 3：监督如何分配给 queries

### 6.1 GT 以“图像—概念—实例集合”组织

简化训练样本可写成：图像、概念提示、该概念的实例框/有效 mask，以及标注是否穷尽的状态。**未标注不一定等于不存在。**公开 loss 中有 `is_exhaustive`、`is_valid_mask` 等处理，框标注可用但 mask 不可用时不能强造 mask GT。[S11]

举例：图中有两辆红车和一辆蓝车，提示为“红车”，则红车属于正实例集合；蓝车不属于这个集合。同类另一辆红车不应被当负例。负例始终相对于具体 prompt 定义。

### 6.2 Hungarian matching 是分配步骤，不是独立 loss

对于 `queries` 个预测和 `instances` 个 GT，先建立分类与框几何等代价，求一对一分配。官方 matcher 使用 `linear_sum_assignment`，匹配在无梯度上下文执行。[S12]

```text
assignment = Hungarian(cost_matrix)
matched_queries -> 对应 GT 的框、mask、score 监督
unmatched_queries -> 适用条件下的负 score 监督
```

不同 matcher/config 的代价项可以不同，不应默认匹配总是包含 mask cost。匹配结果决定监督落在哪个 query 上，**不是在总 loss 中再加一个可微 `L_match`**。训练辅助分支可能使用其他分配策略，一对一不代表所有辅助分支都一对一。[S12][S9]

### 6.3 各 head 的实际监督

| 分支 | GT / 标签来源 | 核对实现中的损失或注意事项 |
|---|---|---|
| query score | 匹配状态和定位质量 | `IABCEMdetr` 支持 IoU-aware soft target，不宜简化成所有正例恒为 1 |
| box | 匹配的 GT box | L1 + GIoU |
| instance mask | 匹配且有效的 GT mask | Focal + Dice；可配置采样点计算 |
| presence | 是否有可见的有效 GT 实例 | 独立的存在性分类监督 |
| semantic 分支 | 概念对应前景区域 | 是否启用相应 criterion 由训练配置决定 |

表格依据 [S11]。`IABCEMdetr` 的正例软标签使用预测分数与 box IoU 的组合并 detach；这不等于增加一个 SAM 2 式 mask-IoU 回归 head。

### 6.4 极重要的负样本细节

在 `IABCEMdetr(use_presence=True)` 路径中，若图中没有可见 GT，代码会屏蔽该样本的逐 query 分类损失，改由 presence 承担不存在监督。**不是“所有 query 都额外强压成负例，再加 presence loss”。**[S11]

图中有实例时，未匹配 query 才在适用的标注完整性规则下承担负例作用。因此，“无目标时只算 presence”与“有目标时部分 query 为负例”是两个不同情形。[S11]

### 6.5 一个不误导的 loss 示意

```text
SAM 2:
    sum_over_frames_and_interactions(
        weighted_focal + weighted_dice + weighted_iou + weighted_object_bce
    )

SAM 3 概念检测分支:
    weighted_query_score + weighted_presence
    + weighted_box_l1 + weighted_box_giou
    + weighted_mask_focal + weighted_mask_dice
    + configured_auxiliary_losses
```

上述省略了 loss gating、归一化和辅助分支细节，是结构示意而非可直接运行的 criterion。不要机械加上 `L_match`、pointer-ID loss 或统一 `L_track`；只有对应训练配置确实定义了某个监督项，才能写进复现方案。[S5][S11][S12]

## 7. 三种失败现象如何定位

以下是基于上述结构的排查建议，而非官方性能保证：

| 现象 | 优先检查 |
|---|---|
| 点在车上，却切出了车门 | 提示歧义、多候选策略、GT 对象粒度，不能只怪分辨率 |
| 找到了车但边界粗糙 | pixel feature 分辨率、skip 融合、GT 边界、mask loss |
| 概念不存在却输出实例 | presence 监督、负概念样本、score 组合与阈值 |
| 漏掉同一概念的其他实例 | 标注是否穷尽、query 分配、检测召回，而非只改 mask head |
| 两个 query 输出同一对象 | 匹配与辅助训练配置、去重策略 |
| 视频逐渐漂移 | 条件帧质量、memory 内容、可见性判断和重新提示机制 |

## 8. 最终记住四件事

1. **SAM 2 的 mask head 是“mask token → MLP 动态权重 → 与高分辨率像素特征点积”。**[S1]
2. **SAM 2 多候选围绕同一 GT 选最佳候选监督；SAM 3 多实例查询需要先做 GT 分配。**[S5][S12]
3. **IoU 质量、对象可见性、概念 presence、实例 score 是不同信号，不能混叫置信度。**[S1][S9][S11]
4. **matching、memory、pointer 是机制或中间表示，不意味着必然各有一个独立 loss。**[S4][S5][S12]

## 9. 源码与材料索引

本地背景材料：`docs/perception/sam2.md`、`docs/perception/sam3.md`。以下路径均相对于对应官方仓库根目录；建议沿“构建配置 → forward → criterion → 数据采样”的顺序阅读。

| 编号 | 官方仓库 / 源码路径 | 重点入口 |
|---|---|---|
| S1 | `facebookresearch/sam2`：`sam2/modeling/sam/mask_decoder.py` | `MaskDecoder.predict_masks`、候选选择 |
| S2 | `facebookresearch/sam2`：`sam2/modeling/sam/prompt_encoder.py` | `PromptEncoder` |
| S3 | `facebookresearch/sam2`：`sam2/modeling/sam/transformer.py` | `TwoWayAttentionBlock` |
| S4 | `facebookresearch/sam2`：`sam2/modeling/sam2_base.py` | `_build_sam_heads`、`_forward_sam_heads`、记忆读写 |
| S5 | `facebookresearch/sam2`：`training/loss_fns.py`；`sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` | `MultiStepMultiMasksAndIous` 和配置的 `loss` |
| S6 | `facebookresearch/sam2`：`training/model/sam2.py` | `prepare_prompt_inputs`、纠错点采样 |
| S7 | `facebookresearch/sam3`：`sam3/model_builder.py` | 图像 backbone、decoder、segmentation head 构建 |
| S8 | `facebookresearch/sam3`：`sam3/model/maskformer_segmentation.py` | `MaskPredictor`、`PixelDecoder`、`UniversalSegmentationHead` |
| S9 | `facebookresearch/sam3`：`sam3/model/sam3_image.py` | `_update_scores_and_boxes`、`_run_segmentation_heads` |
| S10 | `facebookresearch/sam3`：`sam3/model/sam3_image_processor.py` | 推理概率组合与过滤 |
| S11 | `facebookresearch/sam3`：`sam3/train/loss/loss_fns.py` | `IABCEMdetr`、`Boxes`、`Masks`、`SemanticSegCriterion` |
| S12 | `facebookresearch/sam3`：`sam3/train/matcher.py` | `HungarianMatcher` 和匹配策略 |
| S13 | `facebookresearch/sam3`：`sam3/model/decoder.py` | 检测 decoder、reference box、presence 路径 |

原理阅读范围：SAM 2 论文 *Segment Anything in Images and Videos*、SAM 3 论文 *Segment Anything with Concepts*；论文入口见两份本地概览。本文不以工程建议替代论文结论，也不将公开代码支持的所有选项都视为默认训练启用项。
