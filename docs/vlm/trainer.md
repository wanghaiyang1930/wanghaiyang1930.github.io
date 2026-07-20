<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trainer Instruction

## Trainer 是什么？

`Trainer` 是 Hugging Face Transformers 库提供的训练管理器，它把模型训练的所有繁琐工作都打包好了。

**类比：** 如果训练一个模型是做一顿饭，`Trainer` 就是自动化厨房：
- 自动控制火候（学习率调度）
- 自动翻炒（梯度更新）
- 自动记录（日志和检查点）
- 自动上菜（评估和保存）

你只需要提供食材（数据）和菜谱（模型），剩下的它都帮你搞定。

## 为什么需要 Trainer？

### 问题：手写训练循环太麻烦

如果不用 Trainer，你需要手写这些代码：

```python
# 手写训练循环 - 至少 50+ 行代码
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 2. 反向传播
        loss.backward()
        
        # 3. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 4. 优化器更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 5. 日志记录
        if step % log_interval == 0:
            print(f"Loss: {loss.item()}")
        
        # 6. 保存检查点
        if step % save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint-{step}.pt")
        
        # 7. 评估
        if step % eval_interval == 0:
            evaluate(model, eval_dataloader)
        
        # ... 还有更多细节处理
```

而且还要处理：
- 分布式训练（多 GPU）
- 混合精度训练（FP16/BF16）
- 梯度累积
- 早停策略
- 学习率调度
- ...

### 解决方案：用 Trainer 一行搞定

```python
from transformers import Trainer, TrainingArguments

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始训练 - 就这么简单！
trainer.train()
```

Trainer 自动帮你做了上面所有的事情！

## Trainer 的核心组件

```python
trainer = Trainer(
    model=model,                    # 你的模型
    args=training_args,             # 训练配置
    train_dataset=train_dataset,    # 训练数据
    eval_dataset=eval_dataset,      # 验证数据（可选）
    data_collator=data_collator,    # 数据整合器（可选）
    compute_metrics=compute_metrics # 评估指标函数（可选）
)
```

### 1. TrainingArguments - 训练配置

这是训练的"控制面板"，所有超参数都在这里设置：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # === 基础设置 ===
    output_dir="./results",              # 保存路径
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=16,      # 每个 GPU 的 batch size
    
    # === 优化器设置 ===
    learning_rate=2e-5,                  # 学习率
    weight_decay=0.01,                   # 权重衰减
    warmup_steps=500,                    # 预热步数
    
    # === 日志和保存 ===
    logging_steps=100,                   # 每 100 步记录一次日志
    save_steps=1000,                     # 每 1000 步保存检查点
    save_total_limit=2,                  # 只保留最近 2 个检查点
    
    # === 评估 ===
    evaluation_strategy="steps",         # 按步数评估
    eval_steps=500,                      # 每 500 步评估一次
    
    # === 分布式训练 ===
    # 自动检测，无需手动配置
    
    # === 混合精度 ===
    fp16=True,                           # 使用 FP16 加速训练
)
```

### 2. 数据集

Trainer 接受标准的 PyTorch `Dataset`：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回一个样本（字典格式）
        return {
            "input_ids": self.data[idx]["input_ids"],
            "labels": self.data[idx]["labels"],
        }
```

### 3. Data Collator - 数据整合器

把多个样本整合成一个 batch：

```python
from transformers import DataCollatorWithPadding

# 自动把不同长度的序列填充到相同长度
collator = DataCollatorWithPadding(tokenizer)
```

## 完整使用示例

### 示例 1：基础训练

```python
from transformers import TrainingArguments
from src.models import DefaultTrainer, WRCVLAModel
from src.data import WeightedDatabase, DefaultCollator

# 1. 准备模型
model = WRCVLAModel.from_pretrained("path/to/model")

# 2. 准备数据
train_dataset = WeightedDatabase(data_path="train.json")
collator = DefaultCollator(processor, max_length=2048)

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=100,
    save_steps=1000,
)

# 4. 创建训练器
trainer = DefaultTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model("./final_model")
```

### 示例 2：带验证的训练

```python
# 准备验证数据
eval_dataset = WeightedDatabase(data_path="val.json")

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    
    # 配置验证
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=16,  # 验证时可以用更大的 batch
    
    # 保存最佳模型
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,  # loss 越小越好
)

trainer = DefaultTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 添加验证数据
    data_collator=collator,
)

trainer.train()
```

### 示例 3：分布式训练（多 GPU）

```python
# 使用 torchrun 启动（命令行）:
# torchrun --nproc_per_node=4 train.py

# train.py 中的代码和单 GPU 一模一样！
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # 每个 GPU 8 个样本
    # 总 batch size = 8 * 4 = 32
)

trainer = DefaultTrainer(...)
trainer.train()  # Trainer 自动处理分布式逻辑
```

### 示例 4：LoRA 微调

```python
from peft import LoraConfig, get_peft_model

# 1. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# 2. 应用 LoRA 到模型
model = WRCVLAModel.from_pretrained("Qwen/Qwen2-VL-7B")
model = get_peft_model(model, lora_config)

# 3. 查看可训练参数
trainer = DefaultTrainer(model=model, ...)
summary = trainer.get_parameter_summary()
print(f"LoRA 可训练参数: {summary['trainable_parameters']:,}")
print(f"占比: {summary['trainable_ratio']:.2%}")
# 输出: LoRA 可训练参数: 8,388,608 (0.12%)

# 4. 训练（只训练 LoRA 参数）
trainer.train()
```

### 示例 5：从检查点恢复训练

```python
# 训练中断后，从最后的检查点继续
trainer = DefaultTrainer(...)

# 自动检测 output_dir 中的最新检查点
trainer.train(resume_from_checkpoint=True)

# 或者指定特定检查点
trainer.train(resume_from_checkpoint="./outputs/checkpoint-5000")
```

## Trainer 的生命周期

```
trainer.train()
    │
    ├─> 初始化
    │   ├─ 加载模型到 GPU
    │   ├─ 创建优化器和学习率调度器
    │   └─ 创建 DataLoader
    │
    ├─> 训练循环 (每个 epoch)
    │   │
    │   └─> 每个 batch:
    │       ├─ 1. 加载数据
    │       ├─ 2. 前向传播 (compute_loss)
    │       ├─ 3. 反向传播
    │       ├─ 4. 梯度裁剪
    │       ├─ 5. 优化器更新
    │       ├─ 6. 学习率调度
    │       ├─ 7. 日志记录 (logging_steps)
    │       ├─ 8. 保存检查点 (save_steps)
    │       └─ 9. 验证评估 (eval_steps)
    │
    └─> 训练结束
        ├─ 保存最终模型
        └─ 返回训练统计信息
```

## 常用技巧

### 技巧 1：梯度累积（模拟大 batch）

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,      # 实际 batch size
    gradient_accumulation_steps=8,      # 累积 8 步
    # 等效 batch size = 4 * 8 = 32
)
```

**用途：** 显存不够时，用时间换空间。

### 技巧 2：混合精度训练

```python
training_args = TrainingArguments(
    fp16=True,  # 使用 FP16，速度快 2-3 倍，显存省一半
    # 或
    bf16=True,  # 使用 BF16（需要 Ampere 架构 GPU，如 A100）
)
```

### 技巧 3：学习率调度

```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # 余弦退火
    warmup_ratio=0.1,            # 前 10% 步数预热
)
```

### 技巧 4：早停

```python
from transformers import EarlyStoppingCallback

trainer = DefaultTrainer(
    ...,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # 连续 3 次没提升就停止
        )
    ]
)
```

### 技巧 5：自定义日志

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 自定义日志处理
        if logs:
            print(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A')}")

trainer = DefaultTrainer(..., callbacks=[CustomCallback()])
```

## DefaultTrainer vs Trainer 对比

| 特性 | Trainer (原版) | DefaultTrainer (WRC-VLA) |
|------|---------------|-------------------------|
| DataLoader 创建 | 使用默认逻辑 | 使用 `build_dataloader()` |
| 分布式支持 | ✅ | ✅ 更精细的控制 |
| 数据加载优化 | 基础 | ✅ 优化 workers/prefetch |
| 参数统计 | ❌ | ✅ `get_parameter_summary()` |
| 自定义采样器 | 需要手动实现 | ✅ 自动处理 |

## 总结

### Trainer 的价值

1. **开箱即用**：几行代码完成复杂训练流程
2. **最佳实践**：内置混合精度、分布式训练等优化
3. **易扩展**：通过继承可以灵活定制
4. **社区支持**：与 Hugging Face 生态完美集成

### 什么时候用 Trainer？

✅ **推荐使用：**
- 标准的监督学习任务
- 需要快速迭代实验
- 使用 Hugging Face 模型
- 需要分布式训练

❌ **不适合：**
- 强化学习（RL）
- 生成对抗网络（GAN）
- 非常特殊的训练逻辑

### 一句话总结

**Trainer 是 Hugging Face 提供的"全自动训练管家"，让你只需关注模型和数据，其他琐事它全包了。**

`DefaultTrainer` 在此基础上针对 WRC-VLA 项目优化了数据加载和参数统计，让训练更高效、更易监控。
