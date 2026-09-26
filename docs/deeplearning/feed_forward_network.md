# Feed-Forward Network（FFN）教程

Feed-Forward Network（前馈神经网络，简称 **FFN**）是 Transformer Block 中和 Self-Attention 同样重要的组成部分。

Attention 主要负责：

> 让不同位置的 token 彼此交换信息。

FFN 主要负责：

> 对每个位置已经聚合好的信息进行独立的非线性变换，提取和重组更高层的特征。

Transformer 中常见的 FFN 也叫 **Position-wise Feed-Forward Network**，因为它对序列中的每个位置使用相同的网络，但每个位置之间不直接交换信息。

本文从最基础的两层全连接网络开始，推导 Transformer FFN 的公式，并解释它和 Attention 的分工、激活函数、维度设计、参数量以及现代模型中的门控 FFN。

---

## 1. FFN 在 Transformer 中的位置

一个典型的 Transformer Block 可以抽象为：

```text
输入 X
  ↓
Multi-Head Attention
  ↓
残差连接 + LayerNorm
  ↓
Feed-Forward Network
  ↓
残差连接 + LayerNorm
  ↓
输出
```

Transformer Block 通常包含两个子层：

1. **Attention 子层**：在不同 token 之间传递信息；
2. **FFN 子层**：对每个 token 的表示做非线性特征变换。

如果一个序列表示为：

```text
X ∈ R^(n × d_model)
```

其中 `n` 是序列长度，`d_model` 是隐藏维度，那么 FFN 会对 `X` 的每一行独立计算，并且每一行使用完全相同的参数。

因此可以把它理解为：

```text
FFN(X[i])：处理第 i 个 token
FFN(X[j])：处理第 j 个 token
```

但 `FFN(X[i])` 和 `FFN(X[j])` 之间没有直接的矩阵乘法，token 之间的信息交流由 Attention 完成。

---

## 2. 从最基础的全连接层开始

一个线性层的基本形式是：

```text
y = Wx + b
```

其中 `x` 是输入向量，`W` 是权重矩阵，`b` 是偏置向量，`y` 是输出向量。

如果连续使用两个线性层：

```text
h = W₁x + b₁
y = W₂h + b₂
```

代入后：

```text
y = W₂(W₁x + b₁) + b₂
  = W₂W₁x + W₂b₁ + b₂
```

这仍然等价于一个线性变换。因此，如果两个线性层之间没有激活函数，堆叠再多线性层也不会增加模型的**非线性表达能力**。

这就是 FFN 必须加入非线性激活函数的原因。

---

## 3. 两层 FFN 的核心公式

在两个线性层之间加入激活函数 `σ`：

```text
h = σ(W₁x + b₁)
y = W₂h + b₂
```

合并写成：

```text
FFN(x) = W₂ σ(W₁x + b₁) + b₂
```

这就是 Transformer 中最经典的 Position-wise FFN 公式。

其中：

- 第一层 `W₁` 把维度从 `d_model` 扩展到 `d_ff`；
- 激活函数引入非线性；
- 第二层 `W₂` 把维度从 `d_ff` 压回 `d_model`。

对应的维度为：

```text
x  ∈ R^(d_model)
W₁ ∈ R^(d_ff × d_model)
b₁ ∈ R^(d_ff)
h  ∈ R^(d_ff)
W₂ ∈ R^(d_model × d_ff)
b₂ ∈ R^(d_model)
y  ∈ R^(d_model)
```

---

## 4. 为什么 FFN 要先升维再降维

Transformer 中通常设置：

```text
d_ff ≈ 4 × d_model
```

例如：

```text
d_model = 512
d_ff    = 2048
```

处理过程是：

```text
512 维输入
   ↓ W₁
2048 维中间表示
   ↓ 激活函数
2048 维非线性表示
   ↓ W₂
512 维输出
```

先升维的直觉是：

1. 在更高维空间中，模型可以表示更多特征组合；
2. 激活函数可以在更大的特征空间中筛选和重组信息；
3. 再降维后，把有用的信息压缩回 Transformer 的主干维度。

可以把它类比为：

```text
输入特征 → 展开到更大的工作空间 → 非线性加工 → 压缩回主空间
```

`d_ff` 不是序列长度，也不是 Attention 的 head 数量，而是 FFN 内部的中间特征维度。

---

## 5. 为什么需要激活函数

### 5.1 没有激活函数时

没有激活函数时：

```text
FFN(x) = W₂(W₁x + b₁) + b₂
```

它本质上仍然是线性函数，无法学习复杂的非线性关系。

### 5.2 加入激活函数后

加入激活函数后：

```text
FFN(x) = W₂ σ(W₁x + b₁) + b₂
```

以 ReLU 为例：

```text
ReLU(z) = max(0, z)
```

如果某个中间特征为负数，ReLU 会把它置为零；如果为正数，则保留它。这相当于学习一种简单的特征开关。

有了非线性，FFN 才能学习类似下面的关系：

- 某些特征同时出现时才激活；
- 某个特征超过阈值后才传递；
- 不同输入区域使用不同的线性变换。

从几何角度看，激活函数让网络可以在不同区域使用不同的线性映射，因此两层 FFN 可以逼近复杂函数。

---

## 6. 常见激活函数

### 6.1 ReLU

```text
ReLU(x) = max(0, x)
```

优点是计算简单、正数区域梯度稳定；缺点是负数区域梯度为零，可能出现“死亡 ReLU”。早期 Transformer 论文中经常使用 ReLU。

### 6.2 GELU

现代 Transformer 中常用 GELU（Gaussian Error Linear Unit）：

```text
GELU(x) = x Φ(x)
```

其中 `Φ(x)` 是标准正态分布的累积分布函数。常见近似形式为：

```text
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

和 ReLU 的硬截断不同，GELU 是平滑的，会根据输入大小对特征进行软门控。

### 6.3 SiLU / Swish

```text
SiLU(x) = x · sigmoid(x)
```

SiLU 也是平滑的门控激活函数，在一些现代架构中会和 GLU 结构结合使用。

---

## 7. 对单个 Token 的完整计算

假设：

```text
d_model = 4
d_ff    = 8
```

某个 token 的输入是：

```text
x = [x₁, x₂, x₃, x₄]ᵀ
```

第一层线性变换：

```text
z = W₁x + b₁
```

此时 `z ∈ R^8`。然后经过激活函数：

```text
h = GELU(z)
```

最后通过第二层：

```text
y = W₂h + b₂
```

得到 `y ∈ R^4`。完整计算链条为：

```text
x (4)
  ↓ W₁
z (8)
  ↓ GELU
h (8)
  ↓ W₂
y (4)
```

这里 `h` 可以理解为该 token 在更宽特征空间中的中间表示。

---

## 8. 序列输入时为什么叫 Position-wise FFN

对于序列输入：

```text
X ∈ R^(n × d_model)
```

使用右乘记号时，FFN 可以写成：

```text
FFN(X) = σ(XW₁ + b₁)W₂ + b₂
```

其中：

```text
X  ∈ R^(n × d_model)
W₁ ∈ R^(d_model × d_ff)
W₂ ∈ R^(d_ff × d_model)
```

因此：

```text
XW₁       : (n, d_ff)
σ(...)    : (n, d_ff)
σ(...)W₂  : (n, d_model)
```

关键点是：`X` 的每一行都经过同一组 `W₁、b₁、W₂、b₂`，但行与行之间没有相互混合。

---

## 9. FFN 和 Attention 的分工

| 模块 | 主要作用 | 是否混合不同位置 | 典型计算 |
|------|----------|------------------|----------|
| Attention | 建立 token 之间的关系 | 是 | `softmax(QKᵀ)V` |
| FFN | 变换单个 token 的特征 | 否 | `W₂σ(W₁x+b₁)+b₂` |

Attention 负责回答：

> 当前 token 应该从哪些其他 token 获取信息？

FFN 负责回答：

> 在收集到上下文信息之后，当前 token 的表示应该如何被重新编码？

可以把 Transformer Block 简化理解为：

```text
Attention：信息交流
FFN：信息加工
```

---

## 10. FFN 的参数量推导

忽略偏置时，经典 FFN 的参数量为：

第一层：

```text
d_model × d_ff
```

第二层：

```text
d_ff × d_model
```

总参数量：

```text
2 × d_model × d_ff
```

如果 `d_ff = 4d_model`，那么：

```text
参数量 ≈ 8d_model²
```

例如 `d_model = 512、d_ff = 2048` 时，两个权重矩阵共有：

```text
512 × 2048 + 2048 × 512 = 2,097,152
```

再加上两个偏置向量 `2048 + 512 = 2560`。这说明 FFN 往往占据 Transformer 较大的参数和计算量。

---

## 十一、FFN 的计算量推导

对于一个长度为 `n` 的序列，第一层和第二层矩阵乘法的主要计算量分别约为：

```text
n × d_model × d_ff
n × d_ff × d_model
```

因此 FFN 总计算量约为：

```text
2n × d_model × d_ff
```

若 `d_ff = 4d_model`：

```text
计算量 ≈ 8n × d_model²
```

相比之下，Attention 的 `QKᵀ` 包含与 `n²` 相关的计算。因此：

- 序列较短时，FFN 的计算和参数可能更显著；
- 序列很长时，Attention 中和 `n²` 相关的部分会变得昂贵。

---

## 十二、反向传播的基本推导

设：

```text
z = W₁x + b₁
h = σ(z)
y = W₂h + b₂
```

假设损失函数为 `L`，从输出层开始反向传播。

### 12.1 第二层的梯度

因为 `y = W₂h + b₂`，所以：

```text
∂L/∂W₂ = (∂L/∂y)hᵀ
∂L/∂b₂ = ∂L/∂y
∂L/∂h  = W₂ᵀ(∂L/∂y)
```

### 12.2 经过激活函数

因为 `h = σ(z)`，根据链式法则：

```text
∂L/∂z = (∂L/∂h) ⊙ σ'(z)
```

其中 `⊙` 表示逐元素乘法。

### 12.3 第一层的梯度

因为 `z = W₁x + b₁`，所以：

```text
∂L/∂W₁ = (∂L/∂z)xᵀ
∂L/∂b₁ = ∂L/∂z
∂L/∂x  = W₁ᵀ(∂L/∂z)
```

这说明激活函数的导数直接影响梯度能否有效传回第一层。

---

## 十三、现代 Transformer 中的门控 FFN

经典 FFN 是：

```text
FFN(x) = W₂ σ(W₁x + b₁) + b₂
```

现代模型中常见门控线性单元（Gated Linear Unit，GLU）。它使用一条分支产生特征，另一条分支产生门控信号：

```text
GLU(x) = (W_a x) ⊙ σ(W_b x)
```

再经过输出投影：

```text
FFN_GLU(x) = W_o[(W_a x) ⊙ σ(W_b x)]
```

其中 `W_a x` 是候选特征，`σ(W_b x)` 是门控权重，`⊙` 是逐元素相乘。

### 13.1 SwiGLU

SwiGLU 使用 SiLU 作为门控激活：

```text
SwiGLU(x) = [SiLU(xW₁) ⊙ (xW₃)]W₂
```

和普通 FFN 相比，SwiGLU 通常需要三组投影矩阵：`W₁、W₂、W₃`，因此比较模型配置时不能只看名义上的 `d_ff`，还要看实际参数量。

### 13.2 GeGLU

GeGLU 使用 GELU 作为门控激活：

```text
GeGLU(x) = [GELU(xW₁) ⊙ (xW₃)]W₂
```

SwiGLU 和 GeGLU 的共同思想是：

> 一条路径生成特征，另一条路径决定这些特征应该通过多少。

---

## 十四、Pre-LN 和 Post-LN 中的 FFN

FFN 通常和 LayerNorm、残差连接一起使用。常见有两种结构。

### 14.1 Post-LN

```text
H = LayerNorm(X + Attention(X))
Y = LayerNorm(H + FFN(H))
```

### 14.2 Pre-LN

现代 Transformer 中常见：

```text
H = X + Attention(LayerNorm(X))
Y = H + FFN(LayerNorm(H))
```

无论采用哪一种结构，FFN 的核心计算仍然是：

```text
W₂ σ(W₁x + b₁) + b₂
```

区别主要在于 LayerNorm、残差和子层之间的排列顺序。

---

## 15. PyTorch 实现

### 15.1 最基础的 FFN

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return self.net(x)
```

`nn.Linear` 默认作用于输入张量的最后一个维度，因此输入 `(batch_size, seq_len, d_model)` 会自动变成 `(batch_size, seq_len, d_ff)`，再变回 `(batch_size, seq_len, d_model)`。

### 15.2 手动写出矩阵计算

```python
class ManualFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        hidden = self.w1(x)
        hidden = nn.functional.gelu(hidden)
        return self.w2(hidden)
```

这段代码对应：

```text
hidden = GELU(W₁x + b₁)
output = W₂hidden + b₂
```

### 15.3 SwiGLU 风格实现

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)
```

对应公式：

```text
output = W₂(SiLU(W₁x) ⊙ W₃x)
```

---

## 十六、一个完整的 Transformer 子层示例

以 Pre-LN 结构为例：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attention_mask=None):
        normalized = self.norm1(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
        )
        x = x + attention_output
        x = x + self.ffn(self.norm2(x))
        return x
```

这个结构体现了 Transformer Block 的核心分工：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

---

## 十七、输入输出维度总结

假设：

```text
batch_size = 2
seq_len    = 10
d_model    = 512
d_ff       = 2048
```

输入：

```text
x: (2, 10, 512)
```

经过第一层和激活函数：

```text
hidden: (2, 10, 2048)
```

经过第二层：

```text
output: (2, 10, 512)
```

FFN 不改变 batch size 和序列长度，只在最后一个特征维度上做：

```text
512 → 2048 → 512
```

---

## 十八、FFN 的整体推导总结

从一个 token 的表示 `x` 出发：

```text
输入 x ∈ R^(d_model)
  ↓
第一层线性映射
  z = W₁x + b₁
  ↓
升维到 d_ff
  z ∈ R^(d_ff)
  ↓
非线性激活
  h = σ(z)
  ↓
第二层线性映射
  y = W₂h + b₂
  ↓
降维回 d_model
  y ∈ R^(d_model)
```

最终公式：

```text
FFN(x) = W₂ σ(W₁x + b₁) + b₂
```

最核心的一句话是：

> Attention 负责让 token 之间交换信息，FFN 负责在每个位置上独立地进行非线性特征加工。

---

## 19. 常见误区

1. **FFN 不是只包含一个线性层**：Transformer FFN 通常至少包含两层线性层和一个非线性激活函数。
2. **FFN 不负责 Token 之间的信息交流**：它对每个位置独立计算，跨位置的信息交流由 Attention 完成。
3. **`d_ff` 不是序列长度**：它是 FFN 的中间特征维度，常见设置约为 `4 × d_model`。
4. **两个线性层之间必须有非线性**：否则多个线性层仍然可以合并成一个线性层。
5. **FFN 的输出通常要回到 `d_model`**：这样才能和残差分支逐元素相加。
6. **SwiGLU 的参数量不能直接按普通 FFN 计算**：门控结构通常包含三组投影矩阵，需要单独核算。
7. **FFN 本身没有位置信息建模能力**：它只处理当前位置的特征，位置信息通常由位置编码、RoPE 或 Attention 中的位置信息提供。
