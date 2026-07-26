<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2025-01-01 -->

# Qwen3.5-9B 模型结构与 WRC-VLA 参数分区

## 1. 文档目的与事实来源

本文介绍 Qwen3.5-9B 多模态模型结构，并把方案 C 中的“语言主干上层”、“语言主干底层”、“视觉塔前段”、“视觉塔后段”、“跨模态连接层”、“归一化或投影层”等抽象概念映射到真实模块路径。

本文参考一下离线模型：

- 离线 checkpoint：`/home/workspace/data/Qwen/Qwen3.5-9B`；
- checkpoint 配置：`config.json`、`preprocessor_config.json`、`model.safetensors.index.json`；
- 离线模型简介：`README.md`；
- 项目虚拟环境中的 Transformers 5.11.0 Qwen3.5 实现；
- 四个 safetensors 分片中 775 个张量的名称、shape 和 dtype。

> 免责声明：本文所称“前段/后段”、“底层/上层”是项目为受控冻结和消融实验定义的工程分区，不是 Qwen3.5 原生配置字段。任何分区都必须通过实验验证，不能把层号边界当成模型官方语义。

## 2. 总览

Qwen3.5-9B 是一个“27 层视觉 Transformer + 视觉 patch merger + 32 层混合语言主干 + 独立 LM head”的早融合多模态因果语言模型：

```text
图像/视频
  -> 3D Patch Embedding
  -> 27 层 Vision Transformer
  -> 2×2 Spatial Patch Merger：1152 -> 4096
  -> 替换序列中的 image/video placeholder embedding
  -> 32 层混合语言主干
       24 层 Gated DeltaNet 线性注意力
       8 层 Gated Full Attention
  -> Final RMSNorm
  -> 独立 LM Head
  -> 248,320 维 logits
```

模型没有独立的 cross-attention connector。视觉 merger 输出的 4096 维向量直接进入语言 token 序列，跨模态融合发生在后续全部语言层中。

## 3. 离线模型总体规格

| 项目 | 离线 checkpoint 实测值 |
|---|---:|
| 顶层架构类 | `Qwen3_5ForConditionalGeneration` |
| checkpoint 总参数 | 9,653,104,368 |
| 当前 Transformers 主推理路径参数，不含 MTP | 9,409,813,744 |
| checkpoint 权重体积 | 19,306,216,416 bytes |
| 主要权重 dtype | BF16 |
| 词表大小 | 248,320 |
| 原生上下文长度 | 262,144 tokens |
| 语言隐藏维度 | 4,096 |
| 语言层数 | 32 |
| 视觉隐藏维度 | 1,152 |
| 视觉层数 | 27 |
| 视觉输出维度 | 4,096 |
| 输入 embedding 与 LM head 绑定 | 否，`tie_word_embeddings=false` |
| MTP 辅助参数 | 243,290,624 |

输入 embedding 和 LM head 各有 1,017,118,720 个参数，分别约占 checkpoint 的 10.54%。因为两者不绑定，WRC-VLA 扩词表时必须分别初始化、训练和保存新增输入行与新增输出行。

## 4. 顶层模块树

```text
Qwen3_5ForConditionalGeneration
├── model: Qwen3_5Model
│   ├── visual: Qwen3_5VisionModel
│   │   ├── patch_embed.proj
│   │   ├── pos_embed
│   │   ├── rotary_pos_emb
│   │   ├── blocks[0..26]
│   │   │   ├── norm1
│   │   │   ├── attn.qkv
│   │   │   ├── attn.proj
│   │   │   ├── norm2
│   │   │   ├── mlp.linear_fc1
│   │   │   └── mlp.linear_fc2
│   │   └── merger
│   │       ├── norm
│   │       ├── linear_fc1
│   │       └── linear_fc2
│   └── language_model: Qwen3_5TextModel
│       ├── embed_tokens
│       ├── layers[0..31]
│       │   ├── input_layernorm
│       │   ├── linear_attn 或 self_attn
│       │   ├── post_attention_layernorm
│       │   └── mlp
│       ├── norm
│       └── rotary_emb
└── lm_head
```

checkpoint 还包含 `mtp.*` 权重，但 Transformers 5.11.0 的 `Qwen3_5ForConditionalGeneration` 当前不实例化该分支，并把 `^mtp.*` 配置为可忽略的 unexpected keys。MTP 不应默认加入 WRC-VLA 的冻结、LoRA 或全参数训练范围，除非训练与推理框架显式支持该分支。

## 5. 视觉塔

### 5.1 Patch Embedding

真实模块路径为：

```text
model.visual.patch_embed.proj
```

它是一个 Conv3d，kernel 和 stride 均为：

```text
(temporal_patch_size, patch_size, patch_size) = (2, 16, 16)
```

每个原始 patch 包含 `3 × 2 × 16 × 16 = 1,536` 个输入值，并被投影到 1,152 维。静态图像由 processor 适配 temporal patch；视频则按时间维切分。

视觉位置表示同时使用：

- `model.visual.pos_embed`：2,304 个可学习位置、每个 1,152 维，对应基础 `48 × 48` 网格，并按输入网格插值；
- `model.visual.rotary_pos_emb`：在视觉注意力中提供二维/时空 rotary position embedding。

### 5.2 27 个视觉 Transformer block

真实模块路径为：

```text
model.visual.blocks.0
...
model.visual.blocks.26
```

每个 block 都是标准 pre-norm 残差结构：

```text
x = x + VisionAttention(LayerNorm(x))
x = x + VisionMLP(LayerNorm(x))
```

视觉注意力使用：

- hidden size 1,152；
- 16 个 attention head；
- 每头 72 维；
- 联合 `qkv` 投影；
- 非因果注意力；
- 可按每张图或视频的 `cu_seqlens` 处理可变长度视觉序列。

视觉 MLP 使用：

- `linear_fc1: 1152 -> 4304`；
- `GELU(tanh approximation)`；
- `linear_fc2: 4304 -> 1152`。

每个视觉 block 有 15,239,504 个参数，27 个 block 合计 411,466,608。

### 5.3 Spatial Patch Merger

真实模块路径为：

```text
model.visual.merger
```

`spatial_merge_size=2` 表示每 `2 × 2` 个空间 patch 合并为一个语言侧视觉 token。merger 的计算为：

```text
4 × 1152 = 4608
-> LayerNorm
-> Linear 4608 -> 4608
-> GELU
-> Linear 4608 -> 4096
```

merger 有 40,119,040 个参数。它同时完成两件事：

1. 把视觉 token 数缩小为原来的四分之一；
2. 把视觉隐藏维度从 1,152 对齐到语言隐藏维度 4,096。

对于当前项目中的 448×448 图像输入，空间 patch(16x16) 网格约为 `28×28`；经过 `2×2` merger 后得到约 `14×14=196` 个语言侧视觉 token。一张图约 196 token，两张图约 392 token，与当前数据配置中的估算一致。

## 6. “跨模态连接层”究竟是什么

Qwen3.5-9B 没有类似 Flamingo/Q-Former 的独立 cross-attention connector，也没有一组只负责图文交互的 adapter。它的连接过程是：

1. `model.visual` 生成 merger 后的 4,096 维视觉向量；
2. processor 在文本序列中放置 `image_token_id=248056` 或 `video_token_id=248057`；
3. 模型用 `masked_scatter` 把这些 placeholder 的普通 token embedding 替换为视觉向量；
4. 图像、视频和文本 embedding 组成同一序列；
5. 统一序列进入全部 32 个语言层进行早融合。

因此在方案 C 中：

- 狭义“跨模态连接层”应映射为 `model.visual.merger`；
- placeholder mask 和 `masked_scatter` 是无参数的数据路由，不是可训练层；
- M-RoPE 位置构造主要是无参数逻辑；
- 真正的跨模态推理分布在 `model.language_model.layers[0..31]`，不能假设只解冻 merger 就等于解冻全部跨模态能力。

## 7. 语言主干

### 7.1 32 层混合布局

语言主干不是 32 层同构 full-attention Transformer，而是 8 次重复以下四层周期：

```text
3 × (Gated DeltaNet Linear Attention -> Dense FFN)
1 × (Gated Full Attention -> Dense FFN)
```

对应层号：

| 类型 | 层号 | 数量 |
|---|---|---:|
| Gated DeltaNet 线性注意力 | 0–2、4–6、8–10、12–14、16–18、20–22、24–26、28–30 | 24 |
| Gated Full Attention | 3、7、11、15、19、23、27、31 | 8 |

这条事实对 LoRA 非常重要：如果只匹配 `q_proj/k_proj/v_proj/o_proj`，只能覆盖 8 个 full-attention 层，会漏掉 24 个 Gated DeltaNet 层。

### 7.2 Gated DeltaNet 线性注意力层

代表性路径：

```text
model.language_model.layers.0.linear_attn
```

主要参数包括：

```text
in_proj_qkv
in_proj_z
in_proj_a
in_proj_b
conv1d
A_log
dt_bias
norm
out_proj
```

配置为：

- Q/K：16 个 head，每头 128 维；
- V：32 个 head，每头 128 维；
- depth-wise causal Conv1d kernel size：4；
- recurrent state 和 gated delta rule 用于线性复杂度的长序列建模；
- `A_log`、`dt_bias` 和部分状态计算使用 FP32 以保持数值稳定性。

每个线性注意力语言层约 218,407,104 个参数。

### 7.3 Gated Full Attention 层

代表性路径：

```text
model.language_model.layers.3.self_attn
```

配置为：

- 16 个 Q head；
- 4 个 KV head，即 grouped-query attention；
- head dimension 256；
- Q 旋转维度 64，对应 `partial_rotary_factor=0.25`；
- `q_proj` 同时产生 query 与输出 gate；
- attention 输出乘 sigmoid gate 后再经过 `o_proj`；
- Q/K 各自使用 head-wise RMSNorm。

主要模块路径为：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
self_attn.q_norm
self_attn.k_norm
```

每个 full-attention 语言层约 209,723,904 个参数。

### 7.4 Dense FFN

所有 32 个语言层都包含稠密 SwiGLU FFN：

```text
gate_proj: 4096 -> 12288
up_proj:   4096 -> 12288
SiLU(gate_proj(x)) * up_proj(x)
down_proj: 12288 -> 4096
```

离线 checkpoint 中没有 expert、router 或 MoE 权重。因此模型卡 Highlights 中关于 sparse MoE 的描述应视为 Qwen3.5 家族级概述，而不是这个 9B checkpoint 的真实 FFN 结构。

### 7.5 语言位置编码与最终归一化

语言模型使用 interleaved M-RoPE，把文本、时间、高度和宽度位置组织到统一 rotary embedding 中。配置包括：

```text
rope_theta = 10,000,000
partial_rotary_factor = 0.25
mrope_section = [11, 11, 10]
mrope_interleaved = true
```

32 层之后使用：

```text
model.language_model.norm
```

它是 4,096 维 RMSNorm。输出再进入独立的 `lm_head`。

## 8. Embedding、LM Head 与扩词表

真实模块路径为：

```text
model.language_model.embed_tokens
lm_head
```

两者 shape 均为：

```text
[248320, 4096]
```

且 checkpoint 明确设置：

```text
tie_word_embeddings = false
```

因此扩充轨迹词表时：

- 新增输入 token 行决定轨迹 token 作为 history/prompt 输入时的表示；
- 新增 LM head 行决定轨迹 token 作为 assistant 输出时的分类边界；
- 只训练新增 input embedding 而不训练新增 LM head 行是不完整的；
- 使用 PEFT Trainable Tokens 时需要确认是否同时覆盖未绑定的两个模块；
- 通用 VLM 模式应 mask 新增轨迹 logits，避免扩词表改变原词表的 softmax 归一化和普通回答。

## 9. 参数量分解

| 区域 | 参数量 | checkpoint 占比 |
|---|---:|---:|
| 语言 32 层 | 6,919,561,728 | 71.68% |
| 输入 embedding | 1,017,118,720 | 10.54% |
| LM head | 1,017,118,720 | 10.54% |
| 视觉 27 blocks | 411,466,608 | 4.26% |
| 视觉 merger | 40,119,040 | 0.42% |
| 视觉 patch embedding | 1,770,624 | 0.02% |
| 视觉 position embedding | 2,654,208 | 0.03% |
| 语言 final norm | 4,096 | 小于 0.01% |
| MTP 辅助分支 | 243,290,624 | 2.52% |
| 总计 | 9,653,104,368 | 100% |

“9B”是模型系列的标称规模。离线 checkpoint 实际包含约 9.65B 参数，其中约 243M 属于当前 Transformers 主推理路径未实例化的 MTP 辅助分支。

## 10. 方案 C 的工程分区

### 10.1 推荐的初始层号边界

为了保持每个四层混合周期完整，语言主干按以下方式分区：

| 工程区域 | 层号 | 参数量 | 含义 |
|---|---|---:|---|
| 语言主干底层 | 0–7 | 1,729,890,432 | 两个完整混合周期，优先保护基础词法、位置和早期跨模态表示 |
| 语言主干中层 | 8–23 | 3,459,780,864 | 四个完整混合周期，承担大部分共享推理 |
| 语言主干上层 | 24–31 | 1,729,890,432 | 两个完整混合周期，最先放置驾驶 LoRA 或进入受控解冻候选 |

视觉塔按 9 个 block 一组分区：

| 工程区域 | 层号 | 参数量 | 含义 |
|---|---|---:|---|
| 视觉塔前段 | blocks 0–8 | 137,155,536 | 低层 patch、纹理、边缘和基础局部结构，优先冻结 |
| 视觉塔中段 | blocks 9–17 | 137,155,536 | 中层视觉组合表示 |
| 视觉塔后段 | blocks 18–26 | 137,155,536 | 接近 merger 的高级视觉表示，优先放置视觉 LoRA 或受控解冻 |

这个 8/16/8 和 9/9/9 分区是为了实验可解释性而定义，不表示第 8、24 或 18 层存在模型原生语义断点。

### 10.2 六类抽象区域到真实模块的映射

| 方案 C 术语 | 真实模块 |
|---|---|
| 语言主干底层 | `model.language_model.layers.0` 至 `.7` |
| 语言主干上层 | `model.language_model.layers.24` 至 `.31` |
| 视觉塔前段 | `model.visual.patch_embed`、`model.visual.pos_embed`、`model.visual.blocks.0` 至 `.8` |
| 视觉塔后段 | `model.visual.blocks.18` 至 `.26` |
| 跨模态连接层 | 狭义为 `model.visual.merger`；广义融合还包括全部语言层 |
| 归一化层 | 视觉 `norm1/norm2`、merger `norm`、语言 `input_layernorm/post_attention_layernorm`、full-attention `q_norm/k_norm`、linear-attention `norm`、最终 `language_model.norm` |
| 投影层 | 视觉 `qkv/proj` 与 `linear_fc1/2`、merger `linear_fc1/2`、语言 attention/DeltaNet 投影、语言 MLP `gate/up/down_proj`、`lm_head` |

## 11. LoRA 与受控解冻建议

### 11.1 第一优先：冻结原参数，只加驾驶 LoRA

首个方案 B/C 基线应冻结所有原始参数，在语言主干上层 `layers.24–31` 放置 LoRA。目标模块必须同时覆盖两种 token mixer：

```text
线性注意力层：
  linear_attn.in_proj_qkv
  linear_attn.in_proj_z
  linear_attn.in_proj_a
  linear_attn.in_proj_b
  linear_attn.out_proj

全注意力层：
  self_attn.q_proj
  self_attn.k_proj
  self_attn.v_proj
  self_attn.o_proj

所有上层：
  mlp.gate_proj
  mlp.up_proj
  mlp.down_proj
```

`target_modules="all-linear"` 不适合作为未经检查的默认配置，因为它可能同时命中视觉塔、merger、LM head 或其他不希望适配的线性层。应使用显式模块路径或经过打印验证的正则表达式。

### 11.2 第二优先：视觉塔后段 Adapter/LoRA

如果图像打乱、遮挡或时间帧消融表明语言侧 LoRA 无法充分利用视觉信息，可在 `model.visual.blocks.18–26` 的以下模块加入 LoRA：

```text
attn.qkv
attn.proj
mlp.linear_fc1
mlp.linear_fc2
```

视觉 LayerNorm 默认保持冻结，先隔离“新增低秩容量”的收益。

### 11.3 第三优先：受控解冻 merger

`model.visual.merger` 是最小的有参数视觉到语言接口，只有约 40.1M 参数，适合作为第一个原始共享模块解冻候选。但它服务所有图像和视频任务，更新后会直接改变全部视觉 token 的 4096 维分布，因此必须配合 VLM replay、教师蒸馏和 3%/5% 保真门槛。

### 11.4 第四优先：受控解冻语言主干上层

只有 LoRA、视觉后段 Adapter 和 merger 实验仍显示容量瓶颈时，才逐个完整四层周期解冻语言上层，例如先解冻 28–31，再扩展到 24–31。不要只解冻 full-attention 层 27/31 而忽略相邻 DeltaNet 层，否则会破坏周期级消融的可解释性。

### 11.5 归一化层的处理

Norm 参数量很小，但它们控制共享激活尺度，改变后可能影响所有通用任务。推荐顺序为：

1. LoRA 阶段全部冻结 norm；
2. merger 或原始 block 受控解冻时，先保持 norm 冻结作为对照；
3. 只有验证显示尺度适配是瓶颈时，再单独开放目标区域的 norm；
4. 不把“参数少”误认为“遗忘风险低”。

Gated DeltaNet 的 `A_log`、`dt_bias`、`conv1d` 和 gated norm 也不属于普通 LoRA 线性投影。默认保持冻结，若需调整必须作为独立消融。

## 12. 推荐的渐进实验顺序

| 阶段 | 可训练区域 | 研究问题 |
|---|---|---|
| E0 | 仅新增轨迹 input/output token 行 | 扩词表和输出分类是否正确 |
| E1 | E0 + 语言上层 24–31 LoRA | 冻结视觉表示是否足以支持轨迹学习 |
| E2 | E1 + 视觉后段 18–26 LoRA | 驾驶任务是否需要视觉特征适配 |
| E3 | E2 + 原始 merger | 最小共享跨模态接口更新是否带来收益 |
| E4 | E3 + 原始语言层 28–31 | LoRA 是否存在明确容量瓶颈 |
| E5 | E4 + 原始语言层 24–27 | 扩大共享更新是否仍位于 VLM/VLA Pareto 前沿 |

每一步都应从同一个已知基线分叉，并同时报告轨迹指标、VLM 退化、可训练参数量、GPU hours 和推理延迟。若某一步没有显著轨迹收益，就不应为了“更深微调”继续扩大解冻范围。

## 13. 与当前 WRC-VLA 实现的关系

当前 `WRCModel` 通过 `AutoModelForImageTextToText` 加载 `Qwen3_5ForConditionalGeneration`，正确保留视觉塔。使用 `AutoModelForCausalLM` 会映射到 text-only 变体并丢失视觉模块，因此不适用于图像到轨迹训练。

当前模型扩词表时：

- 调用 `resize_token_embeddings`；
- 初始化新增输入 embedding；
- 在 input/output 未绑定时分别初始化新增 LM head；
- 把扩大后的 vocab size 写回嵌套 backbone config。

后续实现冻结或 LoRA 时应基于本文列出的真实路径生成参数清单，并在训练开始前打印：

- 可训练参数名称；
- 各区域可训练参数量；
- 新旧 token 行的梯度；
- 原始冻结参数训练前后的 hash；
- LoRA 是否覆盖 24 个 linear-attention 层和 8 个 full-attention 层中的预期子集。

## 14. 关键结论

1. Qwen3.5-9B 的语言主干是 24 个 Gated DeltaNet 层与 8 个 Gated Full Attention 层组成的混合结构，不是普通 32 层全注意力 Transformer。
2. 视觉塔包含 27 个 1,152 维 Transformer block，merger 把 `2×2` patch 合并并投影到语言侧 4,096 维。
3. 模型没有独立 cross-attention connector；狭义连接层是 `model.visual.merger`，广义跨模态融合发生在全部语言层。
4. 输入 embedding 与 LM head 不绑定，扩词表必须分别处理新增输入行和输出行。
5. “上下层/前后段”是 WRC-VLA 的实验分区，推荐按完整混合周期和等量视觉 block 划分。
6. LoRA 配置必须覆盖 DeltaNet 和 full-attention 两套模块名，不能只匹配 `q_proj/v_proj`。
7. Norm、DeltaNet 状态参数和 merger 虽然参数量较小，仍可能显著改变通用 VLM 行为，必须受 3%/5% 保真门槛约束。

## 15. 本地参考

- `/home/workspace/data/Qwen/Qwen3.5-9B/config.json`
- `/home/workspace/data/Qwen/Qwen3.5-9B/preprocessor_config.json`
- `/home/workspace/data/Qwen/Qwen3.5-9B/model.safetensors.index.json`
- `/home/workspace/data/Qwen/Qwen3.5-9B/README.md`
- `.venv/lib/python3.13/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `.venv/lib/python3.13/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py`

