# Multi-Head Attention（多头注意力）推导

Multi-Head Attention（MHA）是 Transformer 的核心模块。它解决的问题可以概括为：

> 对序列中的每个位置，动态地从所有位置提取与当前任务最相关的信息。

例如，在句子 `小明把书放在桌子上，因为它很重` 中，模型处理“它”时，应该更关注“书”，而不是距离更近的“桌子”。注意力机制允许模型根据内容动态决定关注谁。

本文从最基础的点积相似度开始，逐步推导出 Scaled Dot-Product Attention 和 Multi-Head Attention。

---

## 1. 前提

假设输入序列长度为 `n`，模型隐藏维度为 `d_model`：

```text
X ∈ R^(n × d_model)
```

- `n`：序列长度，例如句子中的 token 数量
- `d_model`：每个 token 的特征维度
- `X[i]`：第 `i` 个 token 的向量

注意力的输出通常仍然是一个 `n × d_model` 的矩阵，也就是每个位置得到一个融合了上下文信息的新表示。

---

## 2. 注意力到底要计算什么

对序列中的某个位置 `i`，我们希望得到一个新的向量 `o_i`。这个向量应该满足：

1. 找到与当前位置相关的其他位置；
2. 相关位置贡献更多信息；
3. 不相关位置贡献更少信息；
4. 把这些信息加权汇总起来。

因此，注意力可以抽象成：

```text
输出 = 对所有 Value 的加权平均
权重 = Query 与 Key 的匹配程度
```

这就引出了三个角色：

- **Query（查询）**：当前位置想寻找什么信息
- **Key（键）**：每个位置可以提供什么信息，用于被匹配
- **Value（值）**：真正被取出并汇总的内容

在自注意力（Self-Attention）中，`Q`、`K`、`V` 都来自同一个输入 `X`，但经过了不同的线性变换。

---

## 3. 从相似度开始：Query 和 Key 的点积

先不考虑线性变换。假设第 `i` 个位置的查询向量是 `q_i`，第 `j` 个位置的键向量是 `k_j`。

用点积衡量二者的匹配程度：

```text
s_ij = q_i · k_j
```

展开后：

```text
s_ij = ∑ₗ q_i,l k_j,l
```

如果 `q_i` 与 `k_j` 的方向相近，点积较大，说明第 `i` 个位置应该更多关注第 `j` 个位置；如果方向不一致，点积可能较小甚至为负数。

把所有位置两两计算，可以得到一个注意力分数矩阵：

```text
S = QKᵀ
```

其中：

```text
Q ∈ R^(n × d_k)
K ∈ R^(n × d_k)
S ∈ R^(n × n)
```

矩阵中第 `i` 行表示第 `i` 个 Query 对所有 Key 的匹配分数：

```text
S[i, j] = q_i · k_j
```

---

## 4. 为什么需要 Softmax

点积得到的分数可以是任意实数，不能直接作为“权重”。我们希望每一行权重满足：

```text
权重 ≥ 0
所有权重之和 = 1
```

因此对每一行应用 Softmax：

```text
A[i, j] = exp(S[i, j]) / ∑ₗ exp(S[i, l])
```

矩阵形式为：

```text
A = softmax(S)
```

Softmax 的作用有两个：

1. 把分数转换成概率分布形式的权重；
2. 通过指数函数放大分数差异，让高匹配分数获得更大的权重。

例如：

```text
分数：    [2.0, 1.0, 0.0]
Softmax： [0.665, 0.245, 0.090]
```

第一个位置的匹配分数最高，所以它在最终汇总中占比最大。

---

## 5. 用权重加权 Value

现在假设每个位置都有一个 Value 向量 `v_j`。第 `i` 个位置的输出就是所有 Value 的加权和：

```text
o_i = ∑ⱼ A[i, j] v_j
```

写成矩阵形式：

```text
O = AV
```

其中：

```text
A ∈ R^(n × n)
V ∈ R^(n × d_v)
O ∈ R^(n × d_v)
```

把前面的步骤连起来，就得到最朴素的点积注意力：

```text
Attention(Q, K, V) = softmax(QKᵀ)V
```

对第 `i` 个位置来说，完整过程是：

```text
q_i 与所有 k_j 做点积
        ↓
得到分数 [s_i1, s_i2, ..., s_in]
        ↓
对分数做 Softmax
        ↓
得到权重 [a_i1, a_i2, ..., a_in]
        ↓
对所有 v_j 做加权求和
        ↓
得到输出 o_i
```

---

## 6. 为什么要除以 `√d_k`

Transformer 使用的不是：

```text
softmax(QKᵀ)V
```

而是：

```text
softmax(QKᵀ / √d_k)V
```

这一步称为 **Scaled Dot-Product Attention**（缩放点积注意力）。

### 6.1 点积维度变大时会发生什么

假设 `q` 和 `k` 的每个分量都满足：

```text
E[q_l] = E[k_l] = 0
Var(q_l) = Var(k_l) = 1
```

点积为：

```text
q · k = ∑ₗ q_l k_l
```

如果各维近似独立，则：

```text
E[q · k] ≈ 0
Var(q · k) ≈ d_k
标准差(q · k) ≈ √d_k
```

也就是说，`d_k` 越大，点积的数值波动越大。

例如，`d_k = 64` 时，点积的标准差大约是 `8`。分数一旦变得很大或很小，Softmax 就容易接近 one-hot：某一个位置权重接近 `1`，其他位置接近 `0`。

这样会带来两个问题：

- Softmax 进入饱和区，输出对输入不敏感；
- 梯度变小，训练不稳定。

除以 `√d_k` 后：

```text
Var((q · k) / √d_k) ≈ 1
```

分数的尺度回到了相对稳定的范围。因此最终公式是：

```text
ScaledDotProductAttention(Q, K, V)
    = softmax(QKᵀ / √d_k)V
```

---

## 7. Q、K、V 是怎么得到的

在 Self-Attention 中，输入是同一个矩阵 `X`，分别经过三组可学习的线性变换：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

假设：

```text
X   ∈ R^(n × d_model)
W_Q ∈ R^(d_model × d_k)
W_K ∈ R^(d_model × d_k)
W_V ∈ R^(d_model × d_v)
```

那么：

```text
Q ∈ R^(n × d_k)
K ∈ R^(n × d_k)
V ∈ R^(n × d_v)
```

为什么要使用三组不同的参数？因为三种角色的功能不同：

- `W_Q` 学习如何表达“我想找什么”；
- `W_K` 学习如何表达“我能被什么查询匹配”；
- `W_V` 学习如何表达“真正要传递的内容”。

把线性变换代入缩放点积注意力：

```text
Attention(X)
 = softmax((XW_Q)(XW_K)ᵀ / √d_k)(XW_V)
```

由于：

```text
(XW_K)ᵀ = W_KᵀXᵀ
```

也可以写成：

```text
Attention(X)
 = softmax(XW_QW_KᵀXᵀ / √d_k)XW_V
```

这个形式说明，注意力分数本质上由输入 token 两两之间的双线性关系决定。

---

## 八、从单头注意力推导多头注意力

单个注意力模块只有一套 `W_Q`、`W_K`、`W_V`，可以称为一个 **head**：

```text
head(X) = Attention(XW_Q, XW_K, XW_V)
```

但一套注意力可能只能学习一种关系。例如：

- 一个 head 关注主谓关系；
- 一个 head 关注指代关系；
- 一个 head 关注相邻词；
- 一个 head 关注远距离依赖。

因此 Transformer 使用 `h` 个独立的注意力头。第 `i` 个头拥有自己独立的参数：

```text
Q_i = XW_i^Q
K_i = XW_i^K
V_i = XW_i^V
```

每个头分别计算缩放点积注意力：

```text
head_i
 = softmax(Q_iK_iᵀ / √d_k)V_i
```

然后把所有头的输出在最后一个维度上拼接：

```text
MultiHead(Q, K, V)
 = Concat(head_1, head_2, ..., head_h)W_O
```

对于 Self-Attention，完整公式为：

```text
MultiHead(X)
 = Concat(
       softmax(XW_1^Q(XW_1^K)ᵀ / √d_k)XW_1^V,
       ...,
       softmax(XW_h^Q(XW_h^K)ᵀ / √d_k)XW_h^V
   )W_O
```

其中 `W_O` 是输出投影矩阵，用来把多个 head 拼接后的表示重新混合。

---

## 九、为什么每个 head 的维度通常是 `d_model / h`

设：

```text
d_model = 512
h       = 8
```

通常令：

```text
d_k = d_v = d_model / h = 64
```

每个 head 的输出维度是 `64`，拼接后：

```text
8 × 64 = 512 = d_model
```

因此多头注意力的输入输出维度可以保持不变：

```text
X                  : (n, 512)
每个 head 的输出     : (n, 64)
拼接后的输出         : (n, 512)
经过 W_O 后的输出     : (n, 512)
```

这也是多头设计的一个重要优点：不是简单地把计算量扩大 `h` 倍，而是把总特征维度拆分到不同子空间中并行计算。

---

## 十、一个具体的维度推导

假设：

```text
batch_size = 2
n          = 10
d_model    = 512
h          = 8
d_k        = d_v = 64
```

输入：

```text
X: (2, 10, 512)
```

线性变换后，可以按 head 拆分为：

```text
Q, K, V: (2, 8, 10, 64)
```

计算 `QKᵀ` 时，最后两个维度做矩阵乘法：

```text
(2, 8, 10, 64) × (2, 8, 64, 10)
    = (2, 8, 10, 10)
```

这里的 `10 × 10` 表示：每个 head 中，每个 token 对所有 token 的注意力分数。

乘以 `V`：

```text
(2, 8, 10, 10) × (2, 8, 10, 64)
    = (2, 8, 10, 64)
```

拼接 8 个 head：

```text
(2, 8, 10, 64) → (2, 10, 512)
```

最后经过 `W_O`，输出仍然是：

```text
(2, 10, 512)
```

---

## 十一、Mask 是怎么加入的

在实际 Transformer 中，Softmax 前通常还会加入 mask：

```text
A = softmax((QKᵀ / √d_k) + M)
```

### 11.1 Padding Mask

一个 batch 中的句子长度可能不同，短句后面需要补 `<PAD>`。模型不应该关注这些补齐位置，因此把对应分数设为负无穷：

```text
M[i, j] = 0       可关注
M[i, j] = -∞      不可关注
```

由于：

```text
exp(-∞) = 0
```

经过 Softmax 后，被屏蔽位置的权重就是 `0`。

### 11.2 Causal Mask

在自回归语言模型中，第 `i` 个 token 不能看到未来的第 `j` 个 token（`j > i`）。因此使用下三角 mask：

```text
M[i, j] = 0       j ≤ i
M[i, j] = -∞      j > i
```

这样第 `i` 个位置只能利用当前位置及之前的信息，避免训练时“偷看答案”。

---

## 十二、PyTorch 形式的伪代码

下面的代码展示了 Multi-Head Attention 的核心计算过程，省略了 dropout、残差连接和 LayerNorm：

```python
import math
import torch
import torch.nn.functional as F


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # x: (batch_size, seq_len, d_model)
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = x @ w_q  # (batch_size, seq_len, d_model)
    k = x @ w_k
    v = x @ w_v

    # 拆成多个 head: (batch_size, num_heads, seq_len, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # 注意力分数: (batch_size, num_heads, seq_len, seq_len)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v

    # 合并 head: (batch_size, seq_len, d_model)
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, seq_len, d_model)
    return output @ w_o
```

实际工程中通常使用 `torch.nn.MultiheadAttention` 或 Transformer 模块，而不是手写上述代码；但理解这段计算有助于排查 shape、mask 和数值稳定性问题。

---

## 十三、Self-Attention、Cross-Attention 的区别

### 13.1 Self-Attention

```text
Q = XW_Q
K = XW_K
V = XW_V
```

`Q`、`K`、`V` 来自同一个序列，作用是让序列内部的 token 互相交流。

### 13.2 Cross-Attention

```text
Q = X_decoder W_Q
K = X_encoder W_K
V = X_encoder W_V
```

Query 来自 decoder，Key 和 Value 来自 encoder。它的作用是让 decoder 在生成当前 token 时，从 encoder 的输出中寻找相关信息。

因此二者的核心公式完全相同，区别只在于 `Q`、`K`、`V` 的来源不同。

---

## 十四、MHA 的整体推导总结

从输入 `X` 出发，Multi-Head Attention 的推导链条可以浓缩为：

```text
输入 X
  ↓
分别线性映射得到 Q、K、V
  Q = XW_Q, K = XW_K, V = XW_V
  ↓
计算 Query-Key 相似度
  S = QKᵀ
  ↓
缩放，避免点积数值过大
  S = S / √d_k
  ↓
加入 mask（可选）
  S = S + M
  ↓
Softmax 得到注意力权重
  A = softmax(S)
  ↓
对 Value 加权求和
  O = AV
  ↓
多个 head 独立计算
  head_i = softmax(Q_iK_iᵀ / √d_k)V_i
  ↓
拼接多个 head，并做输出投影
  MHA = Concat(head_1, ..., head_h)W_O
```

最终公式：

```text
MultiHead(Q, K, V)
 = Concat(
       softmax(Q_1K_1ᵀ / √d_k)V_1,
       ...,
       softmax(Q_hK_hᵀ / √d_k)V_h
   )W_O
```

最核心的一句话是：

> 每个 head 在自己的表示子空间中，根据 Query 和 Key 的匹配程度，对 Value 做加权汇总；多个 head 并行后再融合，从而同时建模不同类型的依赖关系。

---

## 十五、常见误区

1. **注意力权重不是 `Q` 和 `K` 的逐元素乘法**：标准实现使用矩阵乘法 `QKᵀ`，得到所有 Query-Key 对之间的两两分数。
2. **Softmax 的维度通常是最后一维**：每个 Query 对所有 Key 的权重之和应为 `1`，即 `softmax(..., dim=-1)`。
3. **缩放因子是 `√d_k`，不是 `√d_model`**：它对应的是参与点积的 Query/Key 维度。
4. **多头不是复制同一个 head**：每个 head 有独立的投影参数，学习不同的表示子空间。
5. **注意力输出不是只保留最相关的一个 Value**：Softmax 后通常是对所有 Value 的加权组合，只是相关位置权重更大。
6. **MHA 本身没有位置信息**：如果没有位置编码或 RoPE 等机制，注意力对 token 的排列顺序并不敏感。
