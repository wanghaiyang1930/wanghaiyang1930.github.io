# Transformer 技术要点

Transformer 是一种以注意力机制为核心的序列建模架构。它不依赖循环结构，而是让序列中不同位置的 token 通过注意力直接交换信息。

本文从整体结构开始，介绍 Self-Attention、Multi-Head Attention、位置编码、Feed-Forward Network、残差与归一化，并补充 Encoder、Decoder、训练、推理和 KV Cache 等实现细节。

---

## 1. Transformer 解决什么问题？

传统 RNN 按时间顺序处理序列：当前时刻依赖前一时刻的隐藏状态。这样具有天然的顺序性，但长距离依赖需要经过多次状态传递，并且训练难以充分并行。

Transformer 使用注意力让每个位置直接访问其他位置：

```text
输入 token
    ↓ embedding + positional information
Transformer blocks
    ↓
输出表示或下一个 token 的概率
```

Transformer 的主要特点：

- **全局依赖**：一个位置可以直接关注任意其他位置；
- **高度并行**：训练时整段序列可以同时计算；
- **结构模块化**：注意力、前馈网络、残差和归一化反复堆叠；
- **上下文长度受限**：标准全注意力的时间和显存开销通常随序列长度平方增长。

---

## 2. 输入表示与整体张量形状

设 batch 大小为 `B`，序列长度为 `L`，模型隐藏维度为 `D`：

```text
X ∈ R^(B × L × D)
```

文本模型通常先把 token id 映射为 embedding：

```text
token_ids ∈ R^(B × L)
token_embedding(token_ids) → X ∈ R^(B × L × D)
```

然后注入位置信息：

```text
X_with_position = X + positional_information
```

位置信息也可以通过相对位置偏置、RoPE 等方式作用在注意力内部，而不一定直接加到输入。详细位置编码可参见同目录的 `positional_encoding.md`。

Transformer block 通常不改变 `B`、`L` 和 `D`，只更新每个位置的表示。

---

## 3. Self-Attention 的基本计算

### 3.1 Query、Key、Value

自注意力从同一个输入 `X` 产生三种表示：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

其中：

- **Query**：当前位置想寻找什么信息；
- **Key**：每个位置具有什么可匹配的特征；
- **Value**：匹配后真正被聚合的内容。

如果 `X ∈ R^(L × D)`，单头注意力通常有：

```text
Q, K ∈ R^(L × d_k)
V ∈ R^(L × d_v)
```

### 3.2 缩放点积注意力

标准公式为：

```text
Attention(Q, K, V)
  = softmax(QKᵀ / √d_k + mask) V
```

分解为四步：

1. `QKᵀ`：计算每个 query 和所有 key 的匹配分数；
2. 除以 `√d_k`：控制点积的数值尺度；
3. 加 mask：屏蔽 padding、未来位置或不允许访问的位置；
4. softmax 后与 `V` 加权求和。

注意力权重矩阵的形状为：

```text
QKᵀ ∈ R^(L × L)
```

第 `i` 行表示位置 `i` 对所有 key 位置的关注分布。

### 3.3 为什么要除以 `√d_k`？

假设 query 和 key 的每个分量均值接近 0、方差接近 1，则点积包含约 `d_k` 项，方差会随 `d_k` 增大。若直接把较大的分数送入 softmax，分布容易变得过于尖锐，梯度变小。

除以 `√d_k` 可以让分数尺度更稳定。它不是归一化到概率，而是 softmax 前的数值缩放。

---

## 4. Mask 的两种主要用途

### 4.1 Padding mask

一个 batch 中不同样本长度可能不同，需要用 padding 补齐。padding 位置不应对有效 token 产生注意力贡献。

可以把不可见位置的分数设为一个很小的值：

```text
masked_score = -∞
```

softmax 后这些位置的权重接近 0。实际代码通常使用足够小的浮点数，并需注意混合精度下的数值范围。

### 4.2 Causal mask

自回归语言模型预测第 `t` 个 token 时只能访问当前位置和历史位置，不能偷看未来：

```text
允许：k ≤ t
屏蔽：k > t
```

典型的下三角可见性矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

因果 mask 与位置编码不是同一件事：mask 控制“能不能看”，位置机制表示“在哪里或相隔多远”。

---

## 5. Multi-Head Attention

### 5.1 为什么要多头？

单个注意力头只有一组相似度空间。多头注意力将隐藏维度拆成多个子空间，让不同头学习不同关系，例如局部邻近、指代、句法或长距离依赖。

设头数为 `H`，每个头的维度为 `d_head`：

```text
D = H * d_head
```

输入投影后重排为：

```text
[B, L, D] → [B, H, L, d_head]
```

每个头独立计算注意力：

```text
head_h = Attention(Q_h, K_h, V_h)
```

最后拼接所有头，再做输出投影：

```text
MultiHead(Q, K, V)
  = Concat(head_1, ..., head_H) W_O
```

### 5.2 参数量

若 Q、K、V 都来自同一个隐藏维度 `D`，标准实现通常用一个合并投影产生 `3D` 个输出：

```text
W_QKV ∈ R^(D × 3D)
W_O   ∈ R^(D × D)
```

忽略 bias 时，参数量约为：

```text
3D² + D² = 4D²
```

多头改变的是参数的组织和计算路径；在总隐藏维度 `D` 固定时，头数本身通常不会把投影参数量乘上头数。

### 5.3 Multi-Query 与 Grouped-Query Attention

标准 MHA 为每个头分别保存 Q、K、V。推理时 KV Cache 可能很大，因此可以减少 Key/Value 头数：

- **MHA**：Q、K、V 头数相同；
- **MQA**：多个 Q 头共享一组 K/V；
- **GQA**：多个 Q 头分组共享 K/V，每组对应一个 K/V 头。

这样可以降低 KV Cache 的内存和带宽开销，但改变了模型结构，不能直接把已训练好的 MHA 配置当作 GQA 使用而期待结果不变。

---

## 6. Encoder 与 Decoder

### 6.1 Encoder block

典型 Encoder block：

```text
输入 X
  ↓ Self-Attention
残差连接 + LayerNorm
  ↓ Feed-Forward Network
残差连接 + LayerNorm
  ↓
输出
```

Encoder self-attention 通常允许看到序列中的所有位置，因此是双向上下文。

### 6.2 Decoder block

经典 Encoder-Decoder Transformer 的 Decoder block：

```text
输入 Y
  ↓ Causal Self-Attention
残差连接 + LayerNorm
  ↓ Cross-Attention：Q 来自 Decoder，K/V 来自 Encoder
残差连接 + LayerNorm
  ↓ Feed-Forward Network
残差连接 + LayerNorm
  ↓
输出
```

Cross-Attention 让 Decoder 在生成目标序列时访问 Encoder 的输出：

```text
Q = Y W_Q
K = EncoderOutput W_K
V = EncoderOutput W_V
```

### 6.3 常见模型类型

- **Encoder-only**：适合理解、分类、抽取和双向表示学习，例如 BERT 类模型；
- **Decoder-only**：使用因果 mask，适合自回归生成，例如 GPT 类模型；
- **Encoder-Decoder**：输入由 Encoder 编码，Decoder 自回归生成输出，适合翻译、摘要和条件生成。

---

## 7. Feed-Forward Network（FFN）

注意力负责在位置之间交换信息；FFN 负责对每个位置独立进行非线性变换。

经典 FFN：

```text
FFN(x) = activation(x W_1 + b_1) W_2 + b_2
```

对整个序列而言，FFN 对每个位置使用相同参数，但位置之间不直接通信：

```text
FFN(X[i]) 独立处理第 i 个位置
```

典型维度为：

```text
D → D_ff → D
```

其中 `D_ff` 通常大于 `D`，因此 FFN 往往占 Transformer block 参数量的主要部分。

### 7.1 参数量

忽略 bias，两个线性层的参数量为：

```text
D * D_ff + D_ff * D = 2D * D_ff
```

如果 `D_ff = 4D`，约为 `8D²`，再加上注意力投影的约 `4D²`。

### 7.2 门控 FFN

现代模型常使用门控结构，例如 SwiGLU 风格：

```text
FFN(x) = (activation(x W_gate) ⊙ x W_up) W_down
```

其中 `⊙` 是逐元素乘法。门控分支允许模型动态控制特征通道，但会增加投影矩阵数量；为了保持总参数量，实际模型通常会相应调整中间维度。

---

## 8. 残差连接与归一化

### 8.1 残差连接

子层输出通常与输入相加：

```text
output = x + sublayer(x)
```

残差路径为信息和梯度提供了更短的传播通道，使深层堆叠更容易训练。

### 8.2 Post-LN 与 Pre-LN

经典 Post-LN 结构：

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

常见 Pre-LN 结构：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

两者都使用残差和归一化，但训练稳定性、梯度路径和最终输出处理有所不同。实现或加载权重时必须匹配原模型结构，不能只根据层名判断。

### 8.3 LayerNorm 的作用

LayerNorm 通常在最后一个隐藏维度上进行归一化。对单个 token 的隐藏向量 `x`：

```text
mean = average(x)
variance = average((x - mean)^2)
normalized = (x - mean) / sqrt(variance + epsilon)
output = gamma * normalized + beta
```

`epsilon` 用于避免除零和改善数值稳定性。LayerNorm 与 BatchNorm 不同，它不依赖 batch 维度上的统计量。

---

## 9. 计算复杂度与内存

设序列长度为 `L`，隐藏维度为 `D`：

### 9.1 注意力的二次项

`QKᵀ` 产生 `L × L` 的注意力分数矩阵，因此标准全注意力的主要复杂度近似为：

```text
O(L²D)
```

注意力矩阵的存储也与 `L²` 成正比；在 batch 和多头维度下，实际中间张量可能是：

```text
[B, H, L, L]
```

### 9.2 线性投影和 FFN

QKV 投影、输出投影和 FFN 的主要计算量分别近似为：

```text
attention projections: O(LD²)
FFN:                   O(LD D_ff)
```

当 `L` 很长时，`L²D` 的注意力项可能成为瓶颈；当 `D` 或 `D_ff` 较大而序列较短时，线性层可能占主要计算量。

### 9.3 FlashAttention 类实现

FlashAttention 类算法通过分块计算 Q、K、V，避免显式存储完整的 `L × L` 注意力矩阵，同时保持精确或近似相同的注意力结果。

它主要减少显存读写和中间激活占用，不改变标准注意力的数学依赖关系，也不自动把理论时间复杂度从二次降为线性。

---

## 10. 训练流程

### 10.1 自回归语言模型

给定 token 序列：

```text
x_0, x_1, ..., x_(L-1)
```

模型在位置 `t` 预测下一个 token `x_(t+1)`。训练时通常采用 teacher forcing：所有目标 token 一次性送入，通过 causal mask 保证位置 `t` 看不到未来标签。

输出 logits 形状为：

```text
[B, L, vocabulary_size]
```

交叉熵只在目标位置计算，通常将输入右移或将标签左移对齐。

### 10.2 困惑度

语言模型平均负对数似然为：

```text
loss = -1/N * Σ_t log p(x_t | x_<t)
```

困惑度定义为：

```text
perplexity = exp(loss)
```

padding 或不参与损失的位置应使用 ignore mask，否则指标会被无效 token 污染。

### 10.3 Encoder-Decoder 训练

Decoder 输入通常是右移后的目标序列，使用 causal mask；Cross-Attention 使用 Encoder 输出，并结合源序列 padding mask。

训练时目标序列可以并行计算，推理时仍通常需要逐 token 生成，因此训练和推理的计算模式并不完全相同。

---

## 11. 推理与 KV Cache

### 11.1 为什么生成需要逐步进行？

自回归模型在第 `t` 步生成 token 后，才能知道第 `t+1` 步的输入。因而生成存在顺序依赖：

```text
生成 token_1
  ↓
生成 token_2
  ↓
生成 token_3
```

训练阶段可以并行计算全部位置，但推理阶段输出 token 之间通常不能完全并行。

### 11.2 KV Cache 的原理

第一个生成步骤会计算历史 token 的 Key 和 Value。后续步骤中，历史 K/V 不会改变，因此可以缓存：

```text
past_keys   = concat(old_keys, new_key)
past_values = concat(old_values, new_value)
```

当前 token 只需要计算新的 Q、K、V，然后用当前 Q 查询完整的缓存 K/V。

缓存通常按层、按头保存，形状类似：

```text
keys   ∈ R^(B × H_kv × cached_length × d_head)
values ∈ R^(B × H_kv × cached_length × d_head)
```

KV Cache 将每步重复计算历史投影的浪费去掉，但会随着上下文长度线性增长内存。使用 GQA 或 MQA 可以减少 K/V 头数。

### 11.3 位置处理必须连续

如果 prompt 已经使用位置 `0` 到 `99`，下一个 token 应使用位置 `100`，不能因为当前输入只有一个 token 就重新使用位置 `0`。

滑动窗口中，物理缓存下标不一定等于逻辑位置。淘汰旧 token 后是否重置位置，必须和模型训练及位置编码约定一致。

---

## 12. 输出层与权重绑定

Decoder-only 语言模型通常将最终隐藏状态映射到词表 logits：

```text
logits = hidden W_vocab + bias
```

如果词表大小为 `V`，参数矩阵通常为 `D × V`，可能是整个模型中很大的矩阵。

一种常见做法是权重绑定（weight tying）：让输入 embedding 矩阵和输出投影矩阵共享权重。这样可以减少参数，并使输入和输出词表示处于同一参数空间；但必须匹配具体模型的转置和 bias 约定。

logits 不是概率。只有经过 softmax 后才得到词表概率；实际生成时常在 logits 上直接应用 temperature、top-k 或 top-p，再进行采样。

---

## 13. 生成策略

### 13.1 Greedy decoding

每一步选择概率最大的 token：

```text
next_token = argmax(logits)
```

速度快、结果确定，但可能陷入重复或缺乏多样性。

### 13.2 Temperature

```text
probabilities = softmax(logits / temperature)
```

temperature 小于 1 会使分布更尖锐，大于 1 会使分布更平坦。它主要改变采样随机性，不会改变模型本身的 logits。

### 13.3 Top-k 与 Top-p

- **Top-k**：只保留概率最高的 `k` 个 token；
- **Top-p / nucleus sampling**：按概率从高到低累加，只保留累计概率达到 `p` 的最小集合。

这些过滤通常在 softmax 前对 logits 执行，保留集合外的 logits 设为负无穷，然后再归一化采样。

### 13.4 重复控制与停止条件

实际生成还需要处理重复惩罚、禁止 token、EOS 停止、最大长度和 batch 中不同样本提前结束等逻辑。这些属于解码策略，不是 Transformer block 本身的数学结构。

---

## 14. PyTorch 教学实现

下面实现一个不带缓存的单层 Pre-LN Decoder block，输入和输出形状均为 `[batch, length, d_model]`：

```python
import torch
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm_before_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_before_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, hidden_states, padding_mask=None):
        normalized = self.norm_before_attention(hidden_states)
        length = normalized.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=normalized.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.ffn(
            self.norm_before_ffn(hidden_states)
        )
        return hidden_states


block = DecoderBlock(d_model=64, num_heads=4, d_ff=256)
hidden_states = torch.randn(2, 8, 64)
output = block(hidden_states)
assert output.shape == hidden_states.shape
```

`key_padding_mask` 的形状通常为 `[batch, length]`，其中 `True` 表示对应 key 位置需要被忽略。不同 PyTorch API 对 mask 的形状、布尔含义和浮点 mask 约定可能不同，使用时应以所调用 API 的文档和测试为准。

---

## 15. 工程实现中的常见问题

### 15.1 `d_model` 必须能被头数整除

标准多头注意力通常要求：

```text
d_model % num_heads == 0
d_head = d_model / num_heads
```

否则无法均匀拆分头维度，除非使用支持非均匀头维度的特殊实现。

### 15.2 Mask 的设备和 dtype

mask 应与计算所在设备兼容。布尔 mask、加性浮点 mask 和不同精度下的负无穷处理方式不同；混用时可能产生类型错误或数值异常。

### 15.3 Padding mask 与 position_ids

padding mask 决定哪些位置不能被关注；position ids 决定有效 token 使用哪个逻辑位置。给 padding 赋位置编号不会自动屏蔽它们，左 padding、KV Cache 和批处理对齐时尤其容易出错。

### 15.4 训练与推理结构必须匹配

训练使用的 causal mask、位置编码、LayerNorm 顺序、dropout、权重布局和 token shift 必须与推理代码一致。只要其中一项不一致，模型就可能出现明显质量下降。

### 15.5 Dropout 的位置

Transformer 中可能在注意力权重、残差分支、FFN 或 embedding 后使用 dropout。训练模式与评估/推理模式必须正确切换，否则输出会包含不应有的随机性。

### 15.6 混合精度与 softmax

QK 分数可能具有较大动态范围。实践中通常使用稳定的 softmax 实现，并合理选择 float16、bfloat16 或 float32 的计算路径。不能简单地先手写 `exp(scores)` 再归一化，否则容易溢出。

---

## 16. 方法与模块的关系

| 模块 | 主要作用 | 是否在位置之间通信 |
| --- | --- | --- |
| Self-Attention | 根据内容聚合上下文 | 是 |
| Cross-Attention | 从另一序列读取信息 | 是 |
| FFN | 对每个位置做非线性变换 | 否，参数共享但位置独立 |
| Residual | 保留输入并改善梯度传播 | 不直接交换信息 |
| LayerNorm | 稳定每个位置的特征尺度 | 通常在隐藏维度上归一化 |
| Positional Encoding | 注入顺序或相对距离 | 取决于具体实现 |

理解 Transformer 时，可以把一个 block 记成：

```text
注意力：跨位置交换信息
FFN：每个位置独立重组特征
残差：提供稳定的信息通路
归一化：控制中间表示尺度
位置机制：补充顺序和距离
```

---

## 17. 总结

Transformer 的核心不是单独某个公式，而是多个结构部件的配合：

1. **Self-Attention** 用 Q、K、V 计算内容相关的加权聚合。
2. **Multi-Head Attention** 在多个子空间中并行建模不同关系。
3. **位置机制** 为原本缺少顺序信息的注意力补充绝对位置或相对距离。
4. **FFN** 对每个 token 独立执行高维非线性变换。
5. **残差和归一化** 支撑深层网络的稳定训练。
6. **Causal mask** 约束自回归模型只能访问历史信息。
7. **KV Cache** 避免生成时重复计算历史 Key 和 Value。

分析一个 Transformer 实现时，建议按以下顺序核对：张量布局、QKV 投影、头维度、位置机制、mask 语义、LayerNorm 顺序、FFN 中间维度、训练目标和推理缓存。只看模块名称而不核对这些细节，容易把结构相似但行为不同的模型误认为完全相同。
