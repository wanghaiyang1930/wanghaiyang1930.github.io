# 位置编码（Positional Encoding）技术详解

位置编码解决的核心问题是：让模型不仅知道 token 的内容，还能利用它们的顺序、距离和空间位置。

本文沿着“为什么需要位置 → 在哪里注入 → 公式如何工作 → 工程上如何实现”的路线，介绍绝对位置编码、相对位置表示、RoPE 和 ALiBi。论文来源在文末列出，文中的小例子与代码用于解释原理，不对应特定模型的完整实现。

---

## 1. 为什么注意力需要位置信息？

### 1.1 没有位置机制的自注意力具有置换等变性

忽略 dropout，考虑没有位置编码、没有位置相关 mask 的自注意力：

```text
Q = X W_Q
K = X W_K
V = X W_V

Attention(X) = softmax(QKᵀ / √d_head) V
```

如果用置换矩阵 P 重新排列输入行，则：

```text
Q' = PQ
K' = PK
V' = PV

Q'K'ᵀ = P(QKᵀ)Pᵀ
Attention(PX) = P Attention(X)
```

这意味着输入重新排列，输出只是相应重新排列，而注意力本身没有获得“第几个位置”或“相隔几步”的显式信息。

这里是“置换等变”，不是“置换后输出完全不变”。上述推导基于标准注意力公式 [1]。

### 1.2 因果 mask 不等于位置编码

因果 mask 规定 token 只能访问自身和历史位置；它已经引入方向性，因此不能把前面的无 mask 结论直接套到因果模型上。

但 mask 主要回答“能不能看”，位置机制则显式表达“在哪里、相隔多远”。二者不是同一个功能。

---

## 2. 位置可以注入到哪里？

可以按位置机制的作用位置分类：

| 方法 | 注入位置 | 典型表达 |
| --- | --- | --- |
| 正弦绝对位置编码 | 输入隐藏表示 | hidden = embedding + position |
| 可学习绝对位置编码 | 输入隐藏表示 | hidden = embedding + table[position] |
| 相对位置表示 | 注意力的 Key/Value 相关计算 | query 与相对位置向量交互 |
| 相对位置偏置 | softmax 之前的分数 | score += bias[relative_position] |
| RoPE | 投影后的 Query、Key | query/key 按位置旋转 |
| ALiBi | softmax 之前的分数 | score -= slope × distance |

这些方法分别见 [1]—[5]。需要特别区分：给输入向量加编码、旋转 Q/K、给分数加偏置，是三种不同的计算方式。

---

## 3. 正弦绝对位置编码

### 3.1 公式与频率

原始 Transformer 使用不同频率的正弦和余弦函数 [1]。设位置为 position，隐藏维度为 d_model，第 pair_index 个维度对的角频率为：

```text
frequency[pair_index] = 10000^(-2 * pair_index / d_model)

PE[position, 2 * pair_index] = sin(position * frequency[pair_index])
PE[position, 2 * pair_index + 1] = cos(position * frequency[pair_index])
```

相邻两个通道构成一个 sin/cos 对；不同维度对使用不同频率。

例如 d_model = 4：

```text
PE(position) = [sin(position), cos(position),
                sin(position / 100), cos(position / 100)]

PE(0) = [0, 1, 0, 1]
```

高频分量随位置变化更快，低频分量变化更慢。它们共同形成多尺度位置表示，而不是简单地把位置编号复制到每个通道。

### 3.2 为什么要同时使用 sin 和 cos？

利用和角公式，对于某个频率 ω 和位移 Δ：

```text
sin((position + Δ)ω)
    = sin(position·ω)cos(Δω) + cos(position·ω)sin(Δω)

cos((position + Δ)ω)
    = cos(position·ω)cos(Δω) - sin(position·ω)sin(Δω)
```

因此，对固定 Δ，可以通过一个与 position 无关的线性变换，把当前位置的 sin/cos 对转换到偏移后的位置。这是原论文选择该设计的动机之一 [1]。

进一步直接计算一个维度对的内积：

```text
sin(position·ω)sin(other_position·ω)
+ cos(position·ω)cos(other_position·ω)
= cos((position - other_position)ω)
```

但不能据此断言“加了正弦编码的注意力只依赖相对距离”。实际注意力还包含内容向量和学习到的投影矩阵，展开后有内容与位置的交叉项。

### 3.3 为什么通常是相加，不是拼接？

相加保持隐藏维度不变：

```text
[batch, length, d_model] + [length, d_model]
                      → [batch, length, d_model]
```

拼接也可以设计，但会改变后续投影层的输入维度与参数量，不能直接替换已有模型中的相加。相加也不意味着存在唯一的逆操作可以把内容和位置重新分离。

正弦公式可以计算训练长度之外的位置，但“能计算编码”不代表模型一定能正确处理更长上下文，见第 8 节。

---

## 4. 可学习绝对位置编码

建立一个参数表：

```text
position_table ∈ R^(max_positions × d_model)
hidden[position] = token_embedding[position] + position_table[position]
```

参数量为：

```text
max_positions * d_model
```

例如 max_positions = 2048、d_model = 768 时，参数量为 1,572,864。

这种方式让训练决定位置表示，而不是预先指定函数；原始 Transformer 也比较过可学习位置编码与正弦编码 [1]。

工程限制是：参数表索引有边界，未经训练的位置向量也没有可靠的学习基础。扩展表、插值或重新训练都属于额外设计，不能仅把配置中的最大长度改大就认为能力已经扩展。

---

## 5. 相对位置：直接表达“相隔多远”

### 5.1 Shaw 相对位置表示

令 query 位置为 position_q，key 位置为 position_k：

```text
relative_position = position_k - position_q
```

Shaw 等人的方法把相对位置向量加入注意力计算 [2]：

```text
score[q, k] = query[q]ᵀ (key[k] + relative_key[q, k]) / √d_head

output[q] = Σ_k weight[q, k] * (value[k] + relative_value[q, k])
```

其中 relative_key 和 relative_value 由相对距离查表得到；这不是只加一个标量偏置，而是让 query 和相对位置向量发生交互。

为控制参数规模，可以把距离截断到 [-max_distance, max_distance]。超出边界的距离共享边界向量，代价是无法再区分这些更远的具体距离。

### 5.2 T5 风格的分桶相对位置偏置

更轻量的方式是为每个注意力头学习标量偏置 [3]：

```text
score[head, q, k] = content_score[head, q, k]
                 + bias[head, bucket(position_k - position_q)]
```

近距离划分得细，远距离通常采用更粗的对数分桶，超过阈值后共享桶。双向注意力还需要区分方向；因果版本结合可见性约束处理历史距离。

若有 num_heads 个头和 num_buckets 个桶，偏置表参数量为：

```text
num_heads * num_buckets
```

分桶减少了参数数量，但不意味着显式生成完整的 length × length 偏置矩阵没有显存成本。

---

## 6. RoPE：旋转位置编码

### 6.1 核心操作

RoPE 将单个注意力头中的通道分成二维对，按照位置旋转 Query 和 Key [4]。

对一个通道对 [first, second]，角度 angle = position × frequency：

```text
rotated_first  = first * cos(angle) - second * sin(angle)
rotated_second = first * sin(angle) + second * cos(angle)
```

对应二维旋转矩阵：

```text
R(angle) = [ cos(angle)  -sin(angle) ]
           [ sin(angle)   cos(angle) ]
```

不同通道对采用不同频率。基础形式为：

```text
frequency[pair_index] = base^(-2 * pair_index / rotary_dim)
```

rotary_dim 是实际参与旋转的维度，必须为偶数；当只旋转部分通道时，其余通道保持不变。base 和 rotary_dim 都必须与具体模型的训练配置一致。

### 6.2 为什么能引入相对位置？

把每个位置对应的分块旋转矩阵记为 R_position。使用列向量记号：

```text
rotated_query = R_position_q · query
rotated_key   = R_position_k · key
```

旋转后的点积为：

```text
rotated_queryᵀ rotated_key
    = queryᵀ R_position_qᵀ R_position_k key
    = queryᵀ R_(position_k - position_q) key
```

因为旋转矩阵的转置等于反向旋转，同一频率下两个旋转可以合并。这说明：虽然分别使用了绝对位置进行旋转，但它们在点积中的位置作用表现为相对位移 [4]。

这里并不是说整个分数只由距离决定；query、key 仍然携带内容以及上游网络已经混入的信息。

### 6.3 一个数值例子

取某个二维对的频率为 1，且 query = key = [1, 0]：

```text
旋转后的 query = [cos(position_q), sin(position_q)]
旋转后的 key   = [cos(position_k), sin(position_k)]

点积 = cos(position_k - position_q)
```

所以位置对 (2, 5) 和 (12, 15) 在这个例子中的点积相同。这直接来自上面的旋转恒等式。

另一方面，cos 会振荡，因此不能声称“RoPE 的每一个注意力分数都随距离单调下降”。

### 6.4 在注意力中的位置

典型计算顺序为：

```text
hidden
  ↓ 线性投影与拆分注意力头
query, key, value
  ↓ 对 query、key 应用 RoPE
rotated_query, rotated_key, value
  ↓ 点积、缩放、mask、softmax
attention_weights
  ↓ 对 value 加权求和
output
```

这里介绍的常见用法不旋转 Value。二维旋转保持范数，因此不会仅因位置变大而放大向量长度；标准注意力仍使用原本的缩放因子。

### 6.5 维度配对不能随意替换

两种常见组织方式是：

```text
相邻配对：(0, 1), (2, 3), ...
半区配对：(0, half), (1, half + 1), ...
```

在同时进行一致的通道重排和权重变换时，它们可以表达等价运算。但对于已经训练好的权重，不能只更换旋转函数而不调整约定。

下面代码采用相邻配对，目的是展示公式，不是任意模型 checkpoint 的直接替换实现。

---

## 7. ALiBi：直接给远距离注意力加惩罚

ALiBi 不给输入添加位置向量，而是在注意力分数上添加与距离线性相关的偏置 [5]。

对因果注意力中的可见位置 position_k ≤ position_q：

```text
score[head, q, k] = query[q]ᵀ key[k] / √d_head
                 - slope[head] * (position_q - position_k)
```

各头使用预先确定的正斜率。斜率大时，更强地偏好近距离；斜率小时，距离惩罚更弱。未来位置仍需要用因果 mask 屏蔽。

对于同一 query，如果两个 key 的内容分数完全相同，那么距离额外增加 distance_delta 后，两者 softmax 权重之比会包含：

```text
exp(-slope * distance_delta)
```

这是从 softmax 定义得到的结果，说明其局部偏好；并不意味着远距离内容永远不能获得更高权重。

原始 ALiBi 的斜率不是可学习的位置参数，也没有最大长度位置表。不过，论文中的长度外推效果并不构成无限上下文质量保证 [5]。

---

## 8. 长上下文：能算，不等于能泛化

位置机制至少涉及三种不同限制：

1. 实现限制：位置表、缓存或索引是否支持目标长度。
2. 数值与分布限制：更大位置、不同相位和距离分布是否偏离训练条件。
3. 能力与资源限制：模型是否能使用远距离信息，计算和显存是否足够。

RoPE 虽然可以计算更大位置的旋转，但原模型未必适应这些相位组合。Position Interpolation 使用压缩位置的方式，把更长序列映射回较接近原训练范围的坐标，并结合微调 [6]。

其基本思想可写为：

```text
extension_factor = target_length / original_length
scaled_position = position / extension_factor
angle = scaled_position * frequency
```

这同时缩小了相邻 token 的相位间隔，因此并不是免费的能力扩展。其他按频率区别缩放的方法也不能简单等同于“把 base 改大”。

位置编码本身也不会消除标准全注意力的二次计算开销；高效注意力实现可以改变中间存储方式，但这是另一个维度的优化问题。

---

## 9. PyTorch 示例：相邻配对的 RoPE

以下实现约定：

- features 形状为 [batch, heads, length, head_dim]。
- position_ids 形状为 [batch, length]，由调用方提供整数位置。
- 全部 head_dim 参与旋转，且 head_dim 为正偶数。
- 角度与旋转在 float32 下计算，最后转换回输入 dtype。
- 它是教学实现，不包含融合算子、缓存管理和长上下文缩放。

```python
import torch


def apply_rope(features, position_ids, base=10000.0):
    if features.ndim != 4:
        raise ValueError("features must have shape [batch, heads, length, head_dim]")
    if not features.is_floating_point():
        raise ValueError("features must be floating point")

    batch_size, _, sequence_length, head_dim = features.shape
    if head_dim == 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be positive and even")
    if position_ids.shape != (batch_size, sequence_length):
        raise ValueError("position_ids must have shape [batch, length]")
    if base <= 0:
        raise ValueError("base must be positive")

    pair_indices = torch.arange(
        0, head_dim, 2, device=features.device, dtype=torch.float32
    )
    inverse_frequencies = base ** (-pair_indices / head_dim)
    positions = position_ids.to(device=features.device, dtype=torch.float32)
    angles = positions.unsqueeze(-1) * inverse_frequencies
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)

    paired = features.float().reshape(
        batch_size, features.shape[1], sequence_length, head_dim // 2, 2
    )
    first = paired[..., 0]
    second = paired[..., 1]
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    )
    return rotated.flatten(-2).to(dtype=features.dtype)
```

这里 cosine 和 sine 的形状为 [batch, 1, length, head_dim / 2]，沿 head 维广播。相同的位置频率可用于不同头，不意味着各头的 query/key 内容相同。

### 可以检查的数学性质

以下检查接在上述定义之后运行，用来验证零位置恒等、范数保持、共同平移不变性和增量位置一致性：

```python
torch.manual_seed(0)
query = torch.randn(2, 3, 5, 8)
key = torch.randn(2, 3, 5, 8)
positions = torch.arange(5).expand(2, -1)

rotated_query = apply_rope(query, positions)
rotated_key = apply_rope(key, positions)

torch.testing.assert_close(
    apply_rope(query, torch.zeros_like(positions)), query
)
torch.testing.assert_close(
    rotated_query.norm(dim=-1), query.norm(dim=-1)
)

scores = rotated_query @ rotated_key.transpose(-1, -2)
shifted_scores = (
    apply_rope(query, positions + 7)
    @ apply_rope(key, positions + 7).transpose(-1, -2)
)
torch.testing.assert_close(scores, shifted_scores, atol=1e-5, rtol=1e-5)

torch.testing.assert_close(
    rotated_key[:, :, -1:, :],
    apply_rope(key[:, :, -1:, :], positions[:, -1:]),
)
```

这些检查只验证旋转模块，不代表完整模型、KV cache 或特定预训练权重已经验证。float32 也不是无限精度，对极端大位置仍应评估误差。

---

## 10. 工程实现中容易忽略的细节

以下是从位置公式和张量布局出发的实现检查清单，而不是某个框架的默认行为说明。

### 10.1 KV cache 中的位置必须连续一致

如果 prompt 使用位置 0 到 99，下一个 token 应使用位置 100，而不是因为本次输入只有一个 token 就重新使用 0。

如果缓存保存的是已经旋转的 Key，后续不能再次对同一缓存执行 RoPE。也可以设计保存未旋转 Key 的缓存，但必须在使用时按一致的位置约定处理，不能混用。

对于滑动窗口，物理缓存索引不一定等于逻辑位置。不能仅因缓存淘汰了旧 token，就把新 token 的位置重置为当前缓存长度。

### 10.2 Padding 与 position_ids 是两回事

需要区分：

- position_ids：有效 token 使用哪个逻辑坐标。
- attention mask：哪些 token 可以被关注。

左侧 padding 时，一种常见构造思路是对有效位置标记做累计求和再减一，让真实 token 从 0 开始；padding 位置使用合法占位索引，同时通过 mask 屏蔽。

不能把这个思路无条件套到所有模型，仍要匹配其训练约定。设置位置编号也不会自动屏蔽 padding。

### 10.3 拼接多条样本时，重置位置不等于隔离注意力

把多条独立样本打包进一个序列时，如果希望它们互不影响，需要相应的分块注意力 mask。只在样本边界将位置归零，不会阻止它们互相访问。

### 10.4 二维和多维位置不是简单的序列长度

对于图像 patch，可考虑分别编码 row 和 column，例如：

```text
PE(row, column) = concat(PE_row(row), PE_column(column))
```

也可以分别学习行列向量后相加，或给不同通道组施加不同坐标轴的旋转。上式只是一个构造示例，不是所有视觉模型的统一标准。

对二维 RoPE，如果 row 和 column 都使用成对旋转，那么分配给每个轴的维度都应为偶数。若两个轴等分全部旋转维度，总旋转维度就需要能被 4 整除。

将二维网格按行展平得到的一维索引差，不总能表示真实空间距离；视频、三维点和时间序列还需要考虑坐标轴与单位的一致性。

---

## 11. 方法对比与核心结论

| 方法 | 位置参数 | 核心设计 | 需要关注的限制 |
| --- | --- | --- | --- |
| 正弦绝对编码 | 无 | 多频率 sin/cos 与输入相加 | 可计算长位置不代表能泛化 |
| 可学习绝对编码 | 位置向量表 | 学习每个位置的表示 | 表大小、未训练位置 |
| Shaw 相对表示 | 相对向量表 | 内容与相对向量交互 | 距离截断、计算组织 |
| T5 相对偏置 | 分桶标量表 | 按距离桶调整分数 | 远距离分辨率、偏置存储 |
| 基础 RoPE | 无新增位置参数 | 旋转 Q/K，引入相对相位 | 配对布局、频率、缓存位置 |
| 原始 ALiBi | 无新增位置参数 | 每头固定线性距离偏置 | 局部偏好、外推质量边界 |

最重要的三个区分是：

1. 绝对位置描述“我在哪里”，相对位置描述“你离我多远”。
2. 加输入、旋转 Q/K、加分数偏置，不是同一种计算。
3. 编码可以延伸到更大索引，不代表模型自然具备更长上下文能力。

## 12. 参考论文

- [1] Vaswani et al. *Attention Is All You Need*. 2017. arXiv:1706.03762，尤其是第 3.5 节。
- [2] Shaw et al. *Self-Attention with Relative Position Representations*. 2018. arXiv:1803.02155。
- [3] Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv:1910.10683，2019 年预印本，2020 年发表于 JMLR。
- [4] Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021 年预印本。arXiv:2104.09864。
- [5] Press et al. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. 2021 年预印本，ICLR 2022。arXiv:2108.12409。
- [6] Chen et al. *Extending Context Window of Large Language Models via Positional Interpolation*. 2023. arXiv:2306.15595。
