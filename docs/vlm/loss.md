<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VLM Loss Instruction

## Loss 是什么？

`loss`（损失）是一个数字，用来衡量**模型的预测和正确答案差多少**。

**类比：** 把训练想成学生做题。学生写出答案后，老师对照标准答案打分——错得越离谱，扣分越多。这个"扣的分"就是 loss。训练的全部目标，就是让模型不断调整参数，把这个分数降到最低。

```
模型预测：这张图里是「一只猫」
正确答案：这张图里是「一只狗」
         ↓ 对比
loss = 比较大（预测错了）

不断训练后……
模型预测：这张图里是「一只狗」
正确答案：这张图里是「一只狗」
         ↓ 对比
loss = 很小（预测对了）
```

## 核心思想：VLM 的 loss 本质是"预测下一个词"

这是理解 VLM loss 最关键的一点：**VLM 和纯语言模型算 loss 的方式完全一样，都是"预测下一个 token"**。

图像在这里只是"额外的输入"，它经过 [视觉编码器 + mmproj](mmproj.md) 变成一串"视觉 token"，和文本 token 拼在一起送进语言模型。但**算 loss 时，图像 token 本身不参与打分**——模型要预测的永远是"下一个文本 token"。

```TEXT
  图像 ──► [ViT + mmproj] ──► 视觉 token ┐
                                          ├──► [语言模型 LLM] ──► 每个位置的 logits
  文本 ──► [词嵌入]        ──► 文本 token ┘                            │
                                                                       ▼
                                                        和"正确的下一个 token"对比
                                                                       │
                                                                       ▼
                                                                     loss
```

## 三步走：从模型输出到一个 loss 数字

### 第 1 步：模型吐出 logits

语言模型每处理一个位置，都会输出一个长度等于**词表大小**的向量，叫 `logits`。它表示"下一个 token 是词表里每个词的可能性有多大"。

```python
outputs = model(input_ids=input_ids, pixel_values=pixel_values)
print(outputs.logits.shape)
# [batch_size, seq_len, vocab_size]
# 比如 [1, 100, 152064]：1 条样本、100 个位置、每个位置对 15 万个词打分
```

- `seq_len`：不是全局固定，它是当前一个 Batch 中序列的长度。
- `logits`：训练时是一次性并行算出所有位置的分布，这正是 Transformer 和老式 RNN 最大的区别。
    - 如果同时计算所有位置的 `logits`，位置 1 在预测时，能不能偷看到位置 2、3 的答案？ 答案是不能，如果能偷看，那"预测下一个词"就作弊了。靠的是 因果掩码（causal mask）。它强制每个位置在做注意力时，只能看到自己和左边的 token，右边的一律屏蔽。
- `generation`：训练时"答案全知道"，可以并行铺开一次算完；推理时"答案还没生成"，只能一步一步往外蹦，预测一个、再预测下一。

### 第 2 步：错位对齐（shift）

因果语言模型的规则是"用前面的 token 预测下一个 token"。所以要把 logits 和 labels **错开一位**再对比：第 `i` 个位置的预测，对应第 `i+1` 个位置的真实 token。

```
位置:        0      1      2      3      4
tokens:    [图] [ 一 ] [ 只 ] [ 猫 ] [<eos>]

用位置0-i的内容，去预测位置 i+1 的输出概率，模型在每个位置看到的内容（因果掩码）:
位置 0 看到: [图]                → 预测位置 1 是什么 (一)
位置 1 看到: [图, 一]            → 预测位置 2 是什么 (只)
位置 2 看到: [图, 一, 只]        → 预测位置 3 是什么 (猫)
位置 3 看到: [图, 一, 只, 猫]    → 预测位置 4 是什么 (<eos>)

对齐方式（shift）:
logits 去掉最后一个:  [位置0, 位置1, 位置2, 位置3]
labels 去掉第一个:    [  一,    只,    猫,  <eos>]
                     ↑一一对应后再算 loss↑
```

- `labels` 中位置 i 是一次确定的词，而 `logits` 是一个此表长度的概率分布，他俩都不是一个维度，怎么做 `loss` ？
  - 这是是交叉熵（）做的事情，对于每个位置：1）把 logits 转成概率分布（softmax）；2）找到 labels 指定的那个词的概率；3）算 -log(概率)；4）然后把所有位置的 loss 平均。

```python
# Hugging Face 模型内部大致这样做
shift_logits = logits[:, :-1, :].contiguous()   # 去掉最后一个位置
shift_labels = labels[:, 1:].contiguous()        # 去掉第一个位置
```

### 第 3 步：交叉熵（CrossEntropy）

把错位后的 logits 和 labels 送进**交叉熵损失函数**。它衡量"模型给正确答案的那个词打的分够不够高"——正确词的概率越高，loss 越小。

```python
import torch.nn as nn

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(
    shift_logits.view(-1, vocab_size),  # 拍平成 [位置总数, 词表大小]
    shift_labels.view(-1),              # 拍平成 [位置总数]
)
```

## 最关键的机制：-100 决定"哪些 token 要打分"

上面代码里的 `ignore_index=-100` 是理解 VLM loss 的**核心**。

规则很简单：**labels 里值为 `-100` 的位置，完全不参与 loss 计算。** 交叉熵会直接跳过它们，既不打分，也不产生梯度。

这就给了我们一个"开关"，可以精确控制**只对我们希望模型学会生成的部分算 loss**。在 VLM 里，一条训练样本通常长这样：

```
[图像 token] [用户提问] [模型的回答]
   ↑忽略        ↑忽略        ↑只对这段算 loss
   -100        -100      真实 token id
```

为什么这样设计？

| 部分 | labels 设为 | 原因 |
|------|------------|------|
| 图像 token | `-100` | 图像是"输入条件"，模型的任务不是"生成图像 token"，而是看懂它 |
| 用户提问（prompt） | `-100` | 问题是给定的，不需要模型去"预测问题" |
| 模型回答（answer） | 真实 token id | 这才是我们要教模型学会生成的内容 |

**一句话：** 图像 token 和提问 token 只是"喂给模型看的上下文"，只有答案部分才是"要考的题"。

### 举个具体例子

```python
# 假设一条样本：看图回答 "这是什么？" → "一只猫"
input_ids = [img, img, img,  这, 是, 什, 么, ?,   一, 只, 猫, <eos>]
labels    = [-100,-100,-100, -100,-100,-100,-100,-100, 一, 只, 猫, <eos>]
#            └── 图像，忽略 ──┘└──── 提问，忽略 ────┘└── 回答，算 loss ──┘
```

模型在图像 + 提问位置照样做前向计算（提供上下文），但只有"一只猫 <eos>"这几个位置的预测误差，才会被累加进最终的 loss。

> **提示：** 图像 token 具体用哪个占位 id、答案从第几个 token 开始，由数据处理阶段的 collator 决定（参见 [Trainer](trainer.md) 里的 `data_collator`）。它的核心工作之一，就是正确地把该忽略的位置填成 `-100`。

## 在本项目中：loss 从哪来

得益于 [PreTrainedModel](pre_trained_model.md) 的封装，你几乎不用手写上面的 shift 和交叉熵。只要在调用模型时传入 `labels`，模型的 `forward()` 会自动帮你算好 loss 并放进返回值里。

```python
# DefaultModel 把计算委托给内部的 backbone（如 Qwen3.5-VL）
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    attention_mask=attention_mask,
    labels=labels,          # 传了 labels，就会自动算 loss
)

loss = outputs.loss         # 直接拿到一个标量
loss.backward()             # 反向传播
```

而 [Trainer](trainer.md) 又在此之上再封装一层——它在训练循环里自动调用 `outputs.loss`、做反向传播、更新参数，你连 `loss.backward()` 都不用写。

```
数据(含labels) ──► model.forward() ──► outputs.loss ──► Trainer 自动反向传播、更新参数
```

## 总结

| 概念 | 说明 |
|------|------|
| loss 本质 | 衡量"预测的下一个 token"和"真实 token"差多少 |
| VLM 特点 | 图像经 mmproj 变成视觉 token 作为上下文，本身不算 loss |
| shift 错位 | 用位置 `i` 的输出预测位置 `i+1` 的 token |
| 交叉熵 | 正确 token 概率越高，loss 越小 |
| `-100` | labels 为 -100 的位置被忽略——用来屏蔽图像和提问 |
| 只对答案算 loss | 模型只学"该生成的回答"，不学"生成图像或提问" |
| 项目中 | 传入 `labels`，`forward()` 自动算 loss；`Trainer` 自动反向传播 |

**一句话总结：** VLM 的 loss 就是"预测下一个 token"的交叉熵——图像和提问只作为上下文输入并用 `-100` 屏蔽掉，模型只对**答案部分**的预测误差打分，训练就是把这个分数不断降到最低。
