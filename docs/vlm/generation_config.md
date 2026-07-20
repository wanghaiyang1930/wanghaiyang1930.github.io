<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# GenerationConfig Instruction

## GenerationConfig 是什么？

`GenerationConfig` 是一个专门存放**文本生成参数**的配置类。它把"怎么生成"的所有设置集中管理起来。

**类比：** 如果 `generate()` 是一台打印机，`GenerationConfig` 就是打印设置面板——纸张大小、份数、单双面、黑白还是彩色。你不用每次都手动调，设一次默认值，之后就自动用。

## 为什么要单独有一个生成配置？

早期 Transformers 把生成参数（如 `temperature`、`max_length`）直接塞进模型配置 `PretrainedConfig` 里，导致"模型架构参数"和"生成行为参数"混在一起，很乱。

后来 Hugging Face 把它们拆开：

| 配置类 | 负责什么 | 举例 |
|--------|----------|------|
| `PretrainedConfig` | 模型长什么样（架构） | 层数、隐藏维度、词表大小 |
| `GenerationConfig` | 模型怎么说话（生成行为） | 温度、采样、最大长度 |

拆开的好处：同一个模型可以配多套生成设置（对话用一套、翻译用一套），互不干扰。

## 常用参数

```python
from transformers import GenerationConfig

config = GenerationConfig(
    # === 长度控制 ===
    max_new_tokens=256,      # 最多生成多少个新 token（推荐用这个）
    min_new_tokens=10,       # 至少生成多少个

    # === 是否采样 ===
    do_sample=True,          # True=随机采样，False=确定性解码

    # === 采样参数（do_sample=True 时生效）===
    temperature=0.7,         # 温度：越高越随机，越低越保守
    top_k=50,                # 只从概率最高的 50 个 token 里选
    top_p=0.9,               # 核采样：只从累积概率 90% 的 token 里选

    # === 束搜索 ===
    num_beams=1,             # >1 时启用束搜索

    # === 多候选输出 ===
    num_return_sequences=1,  # 返回几条候选序列（束搜索时不能超过 num_beams）
    num_beam_groups=1,       # 分组束搜索的组数（>1 时需配 diversity_penalty）

    # === 重复控制 ===
    repetition_penalty=1.1,  # 惩罚重复，>1 减少复读
    no_repeat_ngram_size=3,  # 禁止重复的 n-gram

    # === 特殊 token ===
    pad_token_id=0,
    eos_token_id=2,          # 遇到它就停止生成
)
```

> **提示：** 优先用 `max_new_tokens` 而不是 `max_length`。前者是"生成多少新内容"，后者是"输入+输出总长度"，后者容易因为输入变长而被截断。

## 参数优先级

调用 `generate()` 时传的参数会**覆盖** `generation_config` 里的默认值。

```python
# 模型默认配置：temperature = 0.8
model.generation_config.temperature = 0.8

# 调用时传 0.5，实际用 0.5
output = model.generate(input_ids, temperature=0.5)

# 不传，用默认的 0.8
output = model.generate(input_ids)
```

记忆口诀：**传参 > 默认配置**。

## 加载、保存与查看

```python
from transformers import GenerationConfig

# 从模型目录加载（读取 generation_config.json）
config = GenerationConfig.from_pretrained("Qwen/Qwen3.5-VL-7B")

# 查看
print(config)

# 修改
config.temperature = 0.9
config.max_new_tokens = 512

# 保存（生成 generation_config.json）
config.save_pretrained("./my_model")
```

保存后的 `generation_config.json` 长这样：

```json
{
  "max_new_tokens": 512,
  "do_sample": true,
  "temperature": 0.9,
  "top_p": 0.9,
  "eos_token_id": 2
}
```

## 总结

| 概念 | 说明 |
|------|------|
| `GenerationConfig` | 集中管理文本生成参数的配置类 |
| 与 `PretrainedConfig` 区别 | 前者管"怎么生成"，后者管"模型架构" |
| 参数优先级 | `generate()` 传参 > `generation_config` 默认值 |
| 持久化 | 保存为 `generation_config.json` |
| 项目中 | `DefaultGenerationConfig` 重写 `from_model_config` 返回干净默认值 |

**一句话总结：** `GenerationConfig` 把生成行为的参数从模型架构配置里独立出来，让你能灵活管理"模型怎么说话"，而不影响"模型长什么样"。
