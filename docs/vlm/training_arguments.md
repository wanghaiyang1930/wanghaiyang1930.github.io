<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TrainingArguments Instruction

## TrainingArguments 是什么？

`TrainingArguments` 是一个存放**所有训练超参数**的配置类。它是训练的"控制面板"，你想调的每一个训练相关的旋钮都在这里。

**类比：** 如果 `Trainer` 是一台洗衣机，`TrainingArguments` 就是上面的旋钮和按钮——水温、转速、洗涤时长、是否烘干。你设好参数，按下开始，机器就照着做。

## 为什么需要它？

训练一个模型涉及几十个参数：学习率、batch 大小、训练轮数、多久保存一次、用不用混合精度……如果这些参数散落在代码各处，既难管理又难复现。

`TrainingArguments` 把它们**集中到一个对象**里：
- 一处配置，全局生效
- 方便保存和复现实验
- `Trainer` 直接读取它，自动应用

## 常用参数分类

```python
from transformers import TrainingArguments

args = TrainingArguments(
    # === 基础 ===
    output_dir="./outputs",              # 输出目录（必填）
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=8,       # 每个 GPU 的 batch 大小
    per_device_eval_batch_size=8,        # 评估时的 batch 大小

    # === 优化器 ===
    learning_rate=2e-5,                  # 学习率
    weight_decay=0.01,                   # 权重衰减
    warmup_steps=500,                    # 学习率预热步数
    lr_scheduler_type="cosine",          # 学习率调度策略

    # === 梯度 ===
    gradient_accumulation_steps=4,       # 梯度累积（模拟大 batch）
    max_grad_norm=1.0,                   # 梯度裁剪阈值

    # === 混合精度 ===
    bf16=True,                           # 用 bfloat16 加速（推荐 A100/H100）
    # fp16=True,                         # 或用 float16（较老的 GPU）

    # === 日志 ===
    logging_steps=100,                   # 每 100 步记一次日志

    # === 保存 ===
    save_strategy="steps",               # 按步数保存
    save_steps=1000,                     # 每 1000 步存一次
    save_total_limit=2,                  # 最多保留 2 个检查点

    # === 评估 ===
    eval_strategy="steps",               # 按步数评估
    eval_steps=500,                      # 每 500 步评估一次

    # === 数据加载 ===
    dataloader_num_workers=4,            # 数据加载进程数
    dataloader_pin_memory=True,          # 锁页内存，加速传输

    # === 其他 ===
    seed=42,                             # 随机种子（复现用）
)
```

> **提示：** `output_dir` 是唯一必填参数，其余都有合理默认值。

## 梯度累积：小显存跑大 batch

这是个很实用的技巧。假设你想要 batch size = 32，但显存只够 8。

```python
args = TrainingArguments(
    per_device_train_batch_size=8,   # 实际每次 8 条
    gradient_accumulation_steps=4,   # 累积 4 次再更新
    # 等效 batch size = 8 × 4 = 32
)
```

原理：连续算 4 个小 batch 的梯度，累加起来，再一次性更新参数。效果接近直接用 batch=32，但显存只需 batch=8 的量。

## 混合精度：加速训练

用更低精度的浮点数（16 位）代替 32 位，速度更快、显存更省。

| 选项 | 适用硬件 | 特点 |
|------|----------|------|
| `bf16=True` | A100 / H100 等新卡 | 数值范围大，更稳定，推荐 |
| `fp16=True` | 较老的 GPU（如 V100） | 兼容性好，可能有数值溢出 |

## 分布式训练相关字段

多 GPU 训练时，`TrainingArguments` 会自动填充这些字段（通常你不用手动设）：

| 字段 | 含义 |
|------|------|
| `world_size` | 总进程数（GPU 总数） |
| `process_index` | 当前进程的编号（第几个 GPU） |
| `local_rank` | 当前节点内的进程编号 |

启动分布式训练用 `torchrun`，框架自动设置这些值：

```bash
torchrun --nproc_per_node=8 train.py
```

## 完整使用示例

```python
from transformers import TrainingArguments
from src.models import DefaultTrainer

args = TrainingArguments(
    output_dir="./wrc_outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=100,
    save_steps=1000,
    dataloader_num_workers=4,
    seed=42,
)

trainer = DefaultTrainer(
    model=model,
    args=args,                    # 传入配置
    train_dataset=train_dataset,
    data_collator=collator,
)

trainer.train()
```

## 总结

| 分类 | 关键参数 |
|------|----------|
| 基础 | `output_dir`、`num_train_epochs`、`per_device_train_batch_size` |
| 优化器 | `learning_rate`、`weight_decay`、`warmup_steps` |
| 显存优化 | `gradient_accumulation_steps`、`bf16`/`fp16` |
| 保存/评估 | `save_steps`、`eval_steps`、`save_total_limit` |
| 分布式 | `world_size`、`process_index`（自动填充） |
| 数据加载 | `dataloader_num_workers`、`dataloader_pin_memory` |

**一句话总结：** `TrainingArguments` 是训练的中央控制面板，把几十个训练超参数集中管理；你的 `DefaultTrainer` 从中读取参数来构建数据加载器和控制训练流程。
