# 卷积（Convolution）技术详解

卷积是计算机视觉和深度学习中提取局部模式的基础算子。它通过一个可学习的小窗口，在输入上滑动并进行加权求和，从而识别边缘、纹理、形状和更高层的空间结构。

本文以二维卷积为主线，解释卷积的计算方式、输出尺寸、参数量和计算量，以及 `stride`、`padding`、`dilation`、`groups` 等参数如何改变行为。

---

## 1. 卷积解决什么问题？

一张图像或特征图可以表示为：

```text
输入 X ∈ R^(C_in × H_in × W_in)
输出 Y ∈ R^(C_out × H_out × W_out)
```

其中 `C` 表示通道数，`H` 和 `W` 表示空间尺寸。卷积通过一个小的局部窗口扫描输入，把邻域中的值加权汇总为输出特征。

卷积通常具有三个结构特性：

1. **局部连接**：一个输出位置只观察输入的一小块区域；
2. **参数共享**：同一个卷积核在不同空间位置复用；
3. **平移等变性**：在边界处理一致时，输入局部模式平移，输出响应也会相应平移。

参数共享使参数量不随图像宽高线性增加；局部连接则利用了图像中邻近位置通常更相关的先验。

---

## 2. 一个二维卷积如何计算？

### 2.1 单通道输入、单个卷积核

假设输入局部区域和卷积核均为 `3 × 3`：

```text
输入局部区域       卷积核
a b c              w00 w01 w02
d e f              w10 w11 w12
g h i              w20 w21 w22
```

输出位置的计算为：

```text
y = a*w00 + b*w01 + c*w02
  + d*w10 + e*w11 + f*w12
  + g*w20 + h*w21 + i*w22
  + bias
```

卷积核随后向右、向下滑动，在每个位置重复同一组权重。

严格的数学卷积会先翻转卷积核；但深度学习框架通常实现的是**互相关（cross-correlation）**，即不翻转卷积核的滑动加权求和。因为卷积核是通过训练学习的，工程上通常仍称这个算子为卷积。

### 2.2 多通道输入

如果输入有 `C_in` 个通道，一个普通卷积核也有 `C_in` 个通道的权重：

```text
kernel ∈ R^(C_in × K_h × K_w)
```

对一个输出位置，需要在每个输入通道上做局部加权，再把通道结果相加：

```text
y = Σ_c Σ_u Σ_v
        X[c, h + u, w + v] * W[c, u, v]
    + bias
```

### 2.3 多个输出通道

如果有 `C_out` 个卷积核，就会得到 `C_out` 个输出通道：

```text
W ∈ R^(C_out × C_in × K_h × K_w)
b ∈ R^(C_out)
Y ∈ R^(C_out × H_out × W_out)
```

每个输出通道通常学习检测不同的局部模式。卷积层本身并不知道某个通道一定表示“边缘”或“颜色”，这些语义由数据和后续网络共同形成。

---

## 3. 输出尺寸公式

对高度方向，普通二维卷积的输出尺寸为：

```text
H_out = floor(
    (H_in + 2 * padding - dilation * (kernel_size - 1) - 1)
    / stride
    + 1
)
```

宽度方向同理。更一般地，若高度方向参数分别写成 `K_h`、`P_h`、`S_h`、`D_h`：

```text
H_out = floor(
    (H_in + 2*P_h - D_h*(K_h - 1) - 1) / S_h + 1
)
```

有效卷积核尺寸为：

```text
effective_kernel = dilation * (kernel_size - 1) + 1
```

例如：

```text
H_in = W_in = 32
kernel_size = 3
padding = 1
stride = 1
dilation = 1
```

得到：

```text
H_out = W_out = floor((32 + 2 - 2 - 1) / 1 + 1) = 32
```

### 3.1 `same` 与 `valid`

- `valid`：通常表示不补零，输出尺寸会缩小；在许多 API 中对应 `padding=0`。
- `same`：目标是让 `stride=1` 时输出空间尺寸与输入相同。

对于奇数卷积核和 `stride=1`，常见选择是：

```text
padding = dilation * (kernel_size - 1) / 2
```

例如 `kernel_size=3, dilation=1` 时使用 `padding=1`；`kernel_size=3, dilation=2` 时使用 `padding=2`。

但 `same` 不意味着任意 stride 都保持原尺寸。偶数卷积核或某些 stride 可能需要上下、左右不对称 padding。

---

## 4. 卷积的核心参数

### 4.1 `kernel_size`

`kernel_size=3` 表示每次观察 `3 × 3` 的局部区域。更大的卷积核可以覆盖更大邻域，但也增加参数量和计算量。

两个连续的 `3 × 3` 卷积，在不改变分辨率时具有近似 `5 × 5` 的理论感受野：

```text
3 + (3 - 1) = 5
```

同时，两个 `3 × 3` 卷积之间可以插入两次非线性激活，参数量通常也少于一个 `5 × 5` 卷积，因此现代网络大量使用 `3 × 3` 卷积。

### 4.2 `stride`

`stride=1` 时卷积窗口逐像素移动；`stride=2` 时每次移动两个像素，通常使输出高度和宽度约减半。

stride 同时影响输出尺寸、后续层看到的有效采样间隔、计算量和特征图存储量。卷积中的 stride 下采样带有可学习参数，而池化通常使用固定聚合规则。

### 4.3 `padding`

常见的零填充会在输入边界外补 0。padding 可以控制输出尺寸、减少边界信息过快缩小，并让奇数卷积核在 `stride=1` 时保持尺寸。

边界补零会让靠近边缘的位置看到更多人工的 0。也可以使用反射、复制或循环等边界模式，具体取决于任务和框架 API。

### 4.4 `dilation`

膨胀卷积在卷积核元素之间插入间隔，从而扩大感受野而不直接增加参数数量：

```text
kernel_size = 3, dilation = 1：看 x0, x1, x2
kernel_size = 3, dilation = 2：看 x0, x2, x4
```

有效卷积核尺寸为：

```text
effective_kernel = dilation * (kernel_size - 1) + 1
```

因此 `3 × 3`、`dilation=2` 的有效覆盖范围是 `5 × 5`，但单通道、单卷积核仍只有 9 个权重。连续使用过大的膨胀率可能产生栅格效应，导致采样位置过于稀疏。

### 4.5 `groups`

`groups` 把输入和输出通道划分成若干组，每组只在组内做卷积。通常要求：

```text
in_channels  能被 groups 整除
out_channels 能被 groups 整除
```

`groups=1` 是普通卷积；分组数增加后，通道连接更稀疏、参数量更少，但跨组信息不会自动交流。

---

## 5. 参数量与计算量

### 5.1 普通卷积参数量

忽略 bias 时，分组卷积的权重参数量为：

```text
parameters = C_out * (C_in / groups) * K_h * K_w
```

如果使用 bias，再加上：

```text
parameters += C_out
```

普通卷积 `groups=1` 时：

```text
parameters = C_out * C_in * K_h * K_w + C_out
```

例如：

```text
C_in = 64
C_out = 128
kernel = 3 × 3
```

不含 bias 的参数量为：

```text
128 * 64 * 3 * 3 = 73,728
```

### 5.2 乘加次数

每个输出位置、每个输出通道大约需要：

```text
(C_in / groups) * K_h * K_w
```

次乘法和相近数量的加法。因此 MACs（乘加操作对）可估算为：

```text
MACs = H_out * W_out
     * C_out
     * (C_in / groups)
     * K_h * K_w
```

若用 FLOPs 统计，很多工具把一次乘法和一次加法算作 2 FLOPs，也有工具把一次乘加算作 1 MAC；比较结果时必须确认统计口径。

### 5.3 参数量不等于计算量

同一个卷积核参数量固定，但输入分辨率增加后，卷积要在更多位置执行，计算量和激活内存都会增加。反过来，一个参数很多但输出分辨率很低的层，计算量不一定最大。

部署优化还需要考虑内存访问、张量布局、硬件 kernel 和算子融合，不能只看参数量。

---

## 6. 感受野（Receptive Field）

某个输出位置能够影响到的输入区域称为它的感受野。需要区分：

- **理论感受野**：按网络结构计算出的最大输入覆盖范围；
- **有效感受野**：训练后真正贡献较大的区域，通常小于理论感受野且并非均匀。

### 6.1 递推公式

令：

- `r_l`：第 `l` 层的理论感受野大小；
- `j_l`：第 `l` 层相邻特征位置在原图上的间隔，也叫 jump；
- `k_l`：第 `l` 层卷积核大小；
- `s_l`：第 `l` 层 stride；
- `d_l`：第 `l` 层 dilation。

初始化：

```text
r_0 = 1
j_0 = 1
```

每经过一层：

```text
effective_kernel_l = d_l * (k_l - 1) + 1
r_l = r_(l-1) + (effective_kernel_l - 1) * j_(l-1)
j_l = j_(l-1) * s_l
```

padding 通常影响边界对齐和输出坐标，不改变内部位置的理论感受野大小。

### 6.2 例子

连续两个 `3 × 3`、`stride=1` 的卷积：

```text
第一层：r = 1 + (3 - 1) * 1 = 3
第二层：r = 3 + (3 - 1) * 1 = 5
```

如果第二层改成 `stride=2`：

```text
第一层：r = 3, j = 1
第二层：r = 3 + 2 * 1 = 5, j = 2
```

第二层输出位置之间对应原输入中相隔 2 个像素的区域。

---

## 7. 常见卷积结构

### 7.1 `1 × 1` 卷积：逐位置的通道混合

`1 × 1` 卷积不查看相邻空间位置，但会在每个 `(h, w)` 位置上对通道做线性变换：

```text
Y[:, h, w] = W · X[:, h, w] + b
```

因此它常用于调整通道数、压缩或扩展特征、进行跨通道信息融合，以及在 bottleneck 结构中降低后续 `3 × 3` 卷积的计算量。

它不扩大空间感受野，但可以改变每个位置的表示维度。

### 7.2 分组卷积

例如：

```text
C_in = 8, C_out = 12, groups = 4
```

每组处理 2 个输入通道并产生 3 个输出通道。参数量是普通卷积的 `1 / groups`，但不同组之间没有直接通道交互。某些网络会用通道混洗或普通 `1 × 1` 卷积重新混合通道。

### 7.3 深度卷积（Depthwise Convolution）

当：

```text
groups = C_in
```

并通常令 `C_out = C_in` 时，每个输入通道单独使用一个空间卷积核，不与其他输入通道混合。

参数量从普通卷积的：

```text
C_in * C_out * K_h * K_w
```

降为：

```text
C_in * K_h * K_w
```

深度卷积自身只做空间特征提取，不做跨通道融合。

### 7.4 深度可分离卷积

常见的深度可分离卷积由两步组成：

```text
输入
  ↓ depthwise：每个通道独立做空间卷积
中间特征
  ↓ pointwise：1 × 1 卷积混合通道
输出
```

当 `C_in=C_out=C`、核大小为 `K × K` 时，普通卷积参数量约为：

```text
C²K²
```

深度可分离卷积参数量约为：

```text
CK² + C²
```

当 `C` 和 `K` 较大时，后者明显更小；但实际速度取决于硬件、实现和内存访问，不一定严格等于理论 MACs 的比例。

### 7.5 空洞、可变形和动态卷积

- **空洞卷积**：固定地扩大采样间隔，扩大感受野。
- **可变形卷积**：学习采样位置偏移，使采样网格适应目标形状。
- **动态卷积**：根据输入生成或选择不同的卷积权重或组合。

后两者比普通卷积更灵活，但通常需要额外的偏移预测、权重生成或特殊算子支持，计算和实现复杂度也更高。

---

## 8. 卷积与下采样、上采样

### 8.1 Strided convolution

通过设置 `stride > 1`，卷积可以同时完成特征提取和空间下采样：

```text
高分辨率输入
    ↓ stride=2 的卷积
低分辨率、可学习特征
```

这种方式与“先普通卷积、再池化”不同：下采样的特征变换可以由同一个可学习算子共同决定。

### 8.2 转置卷积不是普通卷积的逆

转置卷积是卷积线性算子的转置，不是严格的逆运算。它常用于学习上采样，但输出尺寸和重叠覆盖需要特别检查，某些配置还可能产生棋盘格伪影。

关于上采样、转置卷积、插值加卷积和 Pixel Shuffle，可参见同目录的 `upsample.md`。

---

## 9. 一维、二维和三维卷积

卷积的基本思想不依赖维度：

- **1D 卷积**：对时间序列、音频或文本特征沿长度方向滑动；
- **2D 卷积**：对图像或二维特征图沿高度和宽度滑动；
- **3D 卷积**：对视频、体数据沿时间/深度、高度、宽度滑动。

三维卷积输入可表示为：

```text
[batch, channels, depth, height, width]
```

其输出尺寸公式在每个空间维度独立应用。维度增加会显著提高卷积核覆盖的体积和计算量，因此视频模型常使用分解卷积、稀疏采样或沿时间和空间分别处理。

### 9.1 因果一维卷积

时间序列中，如果当前位置不能看到未来，卷积需要使用因果 padding。对核大小为 `K`、dilation 为 `D` 的一维卷积，左侧通常需要：

```text
left_padding = D * (K - 1)
```

实现时经常先在左侧补齐，再使用 `padding=0` 的卷积，从而避免右侧未来信息进入当前输出。对齐方式必须和序列索引约定一致。

---

## 10. 边界、对齐与数值细节

### 10.1 卷积核中心的坐标

当使用奇数核、对称 padding 和 `stride=1` 时，输出位置通常可以自然对应输入中的同一中心位置。

偶数卷积核没有唯一的中心。即使输出尺寸相同，不同 padding 方案也可能产生半个像素的对齐差异。在检测、分割和特征融合中，这类差异可能累积为明显的边界偏移。

### 10.2 padding 不只是尺寸参数

常见 padding 模式包括：

- `zero padding`：边界外为 0；
- `reflect padding`：反射边界附近的值；
- `replicate padding`：复制边缘值；
- `circular padding`：按周期环绕。

即使输出尺寸相同，这些模式也会产生不同结果，尤其是浅层卷积和小图像输入。

### 10.3 bias 与归一化层

如果卷积后紧接着 BatchNorm、LayerNorm 或其他能吸收平移项的归一化层，卷积 bias 往往可以省略，以减少少量参数并避免重复偏置。

但是否关闭 bias 要根据实际结构和归一化位置判断，不能因为“用了归一化”就对所有卷积无条件关闭。

### 10.4 权重初始化与激活函数

卷积的输出方差受输入通道数和核面积影响。常见初始化会根据 `fan_in` 或 `fan_out` 缩放权重，例如 ReLU 网络常使用 He/Kaiming 初始化。

初始化的目标是让信号和梯度在深层传播时保持合理尺度；它不会改变卷积的尺寸公式，也不能代替归一化或残差结构。

---

## 11. PyTorch 实现示例

### 11.1 普通二维卷积

```python
import torch
from torch import nn

features = torch.randn(2, 3, 32, 32)
layer = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
)

output = layer(features)
assert output.shape == (2, 16, 32, 32)
```

此处输入是 `[batch, channels, height, width]`，`padding=1` 使 `3 × 3`、`stride=1` 的卷积保持空间尺寸。

### 11.2 使用 stride 下采样

```python
downsample = nn.Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
)

downsampled = downsample(output)
assert downsampled.shape == (2, 32, 16, 16)
```

### 11.3 分组和深度可分离卷积

```python
grouped = nn.Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    padding=1,
    groups=4,
)

depthwise = nn.Conv2d(
    in_channels=32,
    out_channels=32,
    kernel_size=3,
    padding=1,
    groups=32,
)

pointwise = nn.Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=1,
)

depthwise_separable = nn.Sequential(depthwise, pointwise)
```

### 11.4 手工核对参数量

```python
def count_conv_parameters(
    in_channels, out_channels, kernel_size, groups=1, bias=True
):
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError("channels must be divisible by groups")
    kernel_height, kernel_width = kernel_size
    weight_count = (
        out_channels
        * (in_channels // groups)
        * kernel_height
        * kernel_width
    )
    return weight_count + (out_channels if bias else 0)


assert count_conv_parameters(64, 128, (3, 3)) == 73_856
```

最后一个断言包含 `128` 个 bias 参数，因此结果是 `73,728 + 128 = 73,856`。

---

## 12. 常见误区

### 12.1 “卷积一定会缩小图像”

不一定。输出尺寸由 `kernel_size`、`padding`、`stride` 和 `dilation` 共同决定。`3 × 3`、`padding=1`、`stride=1` 可以保持尺寸；`stride=2` 才通常下采样。

### 12.2 “卷积核越大越好”

更大的核扩大局部覆盖范围，但也增加参数、计算和过拟合风险。多个小核、空洞卷积或多尺度结构可能以更低代价获得更大感受野。

### 12.3 “卷积核是按图像通道独立处理的”

普通卷积的一个输出通道会同时混合所有输入通道；只有深度卷积或特定分组卷积才限制通道之间的连接。

### 12.4 “参数量少，运行一定更快”

深度可分离卷积和分组卷积参数量较少，但实际速度还受硬件并行度、内存访问和算子实现影响。理论 MACs 只能作为估算，不是端到端延迟的充分条件。

### 12.5 “卷积具有平移不变性”

更准确地说，理想离散卷积具有平移等变性：输入平移，输出响应也平移。分类网络通过池化、步幅、全局聚合等操作可能获得一定平移不敏感性，但这不是卷积层单独保证的严格不变性。

### 12.6 “相同输出尺寸就一定空间对齐”

偶数卷积核、不对称 padding、不同上采样方式或多次取整都可能造成对齐差异。特征融合时应核对实际坐标约定，而不只检查张量的 `H`、`W` 是否相等。

---

## 13. 设计卷积模块时的检查清单

使用或实现卷积层时，可以按以下顺序检查：

1. 输入张量布局是否正确，例如 PyTorch `Conv2d` 通常使用 `[N, C, H, W]`。
2. `in_channels`、`out_channels` 和 `groups` 是否满足整除条件。
3. 根据公式手工计算输出高度和宽度，确认是否符合预期。
4. 检查 `stride`、`dilation` 和 padding 是否一起构成了期望的感受野与对齐方式。
5. 估算参数量、MACs 和激活内存，而不是只看卷积核大小。
6. 若使用分组或深度卷积，确认后续是否需要跨通道混合。
7. 若卷积接归一化层，确认是否需要保留 bias。
8. 在边界敏感任务中验证 padding 模式和偶数核的坐标偏移。

## 14. 总结

卷积可以概括为：**用共享的局部权重，在空间位置上滑动并进行通道聚合**。

- `kernel_size` 决定局部窗口；
- `stride` 决定滑动间隔和常见的下采样比例；
- `padding` 决定边界处理和输出尺寸；
- `dilation` 在不增加核参数的情况下扩大理论感受野；
- `groups` 控制通道连接的稀疏程度；
- `1 × 1` 卷积主要负责逐位置的通道变换；
- 深度可分离卷积用“空间卷积 + 通道混合”降低计算成本。

真正理解卷积，不应只记住一个公式，还要同时关注输出尺寸、坐标对齐、参数连接方式、感受野和硬件计算代价。
