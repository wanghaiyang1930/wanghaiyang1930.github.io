<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GenerationMixin Instruction

## GenerationMixin 是什么？

`GenerationMixin` 是 Hugging Face Transformers 库提供的一个 **mixin 类**，它给模型添加"根据输入生成文本"的能力。

**类比：** 如果模型是一台会思考的大脑，`GenerationMixin` 就是给它装上一张"嘴巴"，让它能一个字一个字地把想法说出来。

具体来说，它负责的是**自回归生成**（autoregressive generation）：模型每次预测下一个 token，然后把这个 token 加到输入末尾，再预测下一个，如此循环，直到生成完整的句子。

```
输入：今天天气
   ↓ 预测下一个词
今天天气 → 很
   ↓ 把"很"加进去，继续预测
今天天气很 → 好
   ↓ 继续
今天天气很好 → <结束>
```

这个循环的逻辑就藏在 `GenerationMixin` 里。

## 什么是 Mixin？

在讲 `GenerationMixin` 之前，先理解"mixin"这个概念。

**Mixin 是一种只提供功能、不单独使用的类。** 它像"插件"一样，被混入（mix in）到其他类里，为它们添加特定能力。

```python
# Mixin 本身不能独立工作，它需要被"混入"到真正的模型类里
class GenerationMixin:
    def generate(self, ...):
        # 依赖 self.forward()，但自己不实现 forward
        ...

# 真正的模型类，同时继承 PreTrainedModel 和 GenerationMixin
class CustomModel(PreTrainedModel, GenerationMixin):
    def forward(self, ...):  # 模型主体由这里提供
        ...
    # generate() 从 GenerationMixin 自动获得
```

## 核心方法：generate()

`GenerationMixin` 最重要的方法就是 `generate()`。它是你唯一需要直接调用的方法。

```python
output = model.generate(
    input_ids,              # 输入的 token
    max_new_tokens=100,     # 最多生成 100 个新 token
    do_sample=True,         # 是否随机采样
    temperature=0.7,        # 温度（控制随机性）
    top_p=0.9,              # 核采样
)
```

### generate() 内部做了什么？

`generate()` 把一堆繁琐的工作都打包好了：

```
1. 准备配置        → 合并 generation_config 和你传的参数
2. 准备输入        → 处理 attention_mask、position_ids 等
3. 选择生成策略    → 根据参数决定用哪种解码方式
4. 循环生成        → 一个 token 一个 token 地生成
5. 停止判断        → 遇到结束符或达到长度上限就停
6. 返回结果        → 输出完整的 token 序列
```

如果没有 `GenerationMixin`，这些逻辑你都得自己写，至少几百行代码。

## 生成策略（解码方式）

`generate()` 会根据你传的参数，自动选择不同的**解码策略**。这是理解 `generate()` 的关键。

> **注意：** 早期版本的 transformers 有公开的 `greedy_search()`、`sample()`、`beam_search()` 等方法。新版本已经把它们改成内部私有方法，统一由 `generate()` 根据 `GenerationConfig` 自动调度。你不需要直接调用它们，只需要通过 `generate()` 的参数来控制。

### 1. 贪婪解码（Greedy Decoding）

每一步都选概率最高的 token。最简单、最快，但结果比较死板。

```python
output = model.generate(
    input_ids,
    do_sample=False,   # 不采样
    num_beams=1,       # 不用束搜索
)
```

**特点：** 确定性输出，每次运行结果相同。适合需要稳定输出的场景。

### 2. 束搜索（Beam Search）

同时保留多个候选序列（"束"），最后选整体概率最高的。质量更高，但更慢。

```python
output = model.generate(
    input_ids,
    do_sample=False,
    num_beams=5,       # 保留 5 个候选
)
```

**类比：** 贪婪解码像每一步只走当前最优的路；束搜索像同时探索 5 条路，最后选整体最好的那条。

### 3. 采样（Sampling）

按概率分布随机选 token，输出更有创意、更多样。

```python
output = model.generate(
    input_ids,
    do_sample=True,        # 开启采样
    temperature=0.7,       # 温度：越高越随机，越低越保守
    top_k=50,              # 只从概率最高的 50 个 token 里选
    top_p=0.9,             # 核采样：只从累积概率 90% 的 token 里选
)
```

**参数解释：**
- `temperature`：控制随机性。1.0 是原始分布，>1 更随机，<1 更确定
- `top_k`：只考虑概率最高的 K 个候选
- `top_p`：只考虑累积概率达到 p 的候选（动态数量）

### 策略对比

| 策略 | 参数 | 特点 | 适用场景 |
|------|------|------|----------|
| 贪婪解码 | `do_sample=False, num_beams=1` | 快、确定 | 需要稳定输出 |
| 束搜索 | `do_sample=False, num_beams>1` | 质量高、慢 | 翻译、摘要 |
| 采样 | `do_sample=True` | 多样、有创意 | 对话、创作 |

## GenerationConfig：生成的默认配置

`GenerationMixin` 配合 `GenerationConfig` 使用。`GenerationConfig` 存储生成的默认参数。

```python
# 查看模型的生成配置
print(model.generation_config)
# GenerationConfig {
#   "max_length": 20,
#   "do_sample": false,
#   "temperature": 1.0,
#   ...
# }

# 修改默认配置
model.generation_config.max_new_tokens = 200
model.generation_config.temperature = 0.8

# 之后调用 generate() 会自动用这些默认值
output = model.generate(input_ids)
```

**参数优先级：** 调用 `generate()` 时传的参数 > `generation_config` 里的默认值。

```python
# generation_config.temperature = 0.8
output = model.generate(input_ids, temperature=0.5)  # 实际用 0.5（传参优先）
```

## 总结

| 概念 | 说明 |
|------|------|
| `GenerationMixin` | 给模型添加文本生成能力的 mixin 类 |
| `generate()` | 唯一需要直接调用的生成方法 |
| 解码策略 | 通过 `generate()` 的参数控制（贪婪/束搜索/采样） |
| `GenerationConfig` | 存储生成的默认参数 |
| 在项目中 | `DefaultModel` 继承它以暴露标准生成接口，实际委托给 backbone |

**一句话总结：** `GenerationMixin` 把"自回归生成文本"的复杂逻辑打包成一个即插即用的 `generate()` 方法，你只需要调用它并传入参数，就能用各种策略生成文本。
