<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PreTrainedModel Instruction

## PreTrainedModel 是什么？

`PreTrainedModel` 是 Hugging Face Transformers 里**所有模型的基类**。你用过的 `BertModel`、`Qwen3_5ForConditionalGeneration`，往上追溯都继承自它。

**类比：** 它像一份"模型标准底盘"。汽车厂商造不同车型，但都基于同一个底盘（提供刹车、方向盘、轮子）。`PreTrainedModel` 提供了每个模型都需要的通用能力，各模型只需在上面加自己的独特结构。

## 它提供了哪些通用能力？

继承 `PreTrainedModel` 后，模型自动获得：

| 能力 | 方法 | 作用 |
|------|------|------|
| 加载 | `from_pretrained()` | 从本地或 Hub 加载模型权重和配置 |
| 保存 | `save_pretrained()` | 保存权重和配置到目录 |
| 配置管理 | `self.config` | 自动关联 `PretrainedConfig` |
| 权重初始化 | `init_weights()` | 按规则初始化参数 |
| 权重绑定 | `tie_weights()` | 共享输入输出 embedding 的权重 |
| embedding 操作 | `get/set_input_embeddings()` | 读写词嵌入层 |
| 显存优化 | `gradient_checkpointing_enable()` | 用时间换显存 |
| 上传 | `push_to_hub()` | 上传到 Hugging Face Hub |

这些都不用你自己写，继承就有。

## 自定义模型的必要约定

要写一个自己的 `PreTrainedModel` 子类，通常需要设置几个类属性：

```python
from transformers import PreTrainedModel

class MyModel(PreTrainedModel):
    config_class = MyConfig          # 指定配套的配置类
    base_model_prefix = "model"      # 主干模块的名字（用于权重加载）

    def __init__(self, config):
        super().__init__(config)     # 必须调用，传入 config
        # ... 定义你的网络层

    def forward(self, input_ids, **kwargs):
        # ... 前向计算
        ...
```

**关键约定：**
- `config_class`：告诉框架用哪个配置类（比如 `AutoModel` 靠它找配置）
- `base_model_prefix`：标识"主干"模块名，加载权重时用来对齐参数名
- `__init__` 必须接收 `config` 并调用 `super().__init__(config)`
- 必须实现 `forward()`

## 权重绑定（tie_weights）

这是 `PreTrainedModel` 一个容易忽略但重要的功能。

很多语言模型里，**输入的词嵌入层**和**输出的预测层**共享同一套权重（节省参数、提升效果）。这叫"权重绑定"。

```python
# 概念示意
model.embed_tokens.weight  # 输入：token → 向量
model.lm_head.weight       # 输出：向量 → token
# 这两个可以是同一份权重
```

`PreTrainedModel.tie_weights()` 自动处理这个绑定。是否绑定由配置里的 `tie_word_embeddings` 控制。

## 使用示例

```python
from src.models import DefaultModel

# 加载（from_pretrained 来自 PreTrainedModel）
model = DefaultModel.from_qwen_pretrained("Qwen/Qwen3.5-VL-7B")

# 查看配置（self.config 来自 PreTrainedModel）
print(model.config.action_config)

# 保存（save_pretrained 来自 PreTrainedModel）
model.save_pretrained("./my_wrc_model")

# 开启梯度检查点省显存
model.gradient_checkpointing_enable()
```

## 总结

| 概念 | 说明 |
|------|------|
| `PreTrainedModel` | 所有 Transformers 模型的基类，提供通用能力 |
| 核心方法 | `from_pretrained` / `save_pretrained` / `tie_weights` 等 |
| 必要约定 | `config_class`、`base_model_prefix`、实现 `forward()` |
| 项目中 | `DefaultModel` 继承它，把方法委托给内部 `backbone` |

**一句话总结：** `PreTrainedModel` 提供了模型加载、保存、权重管理等一整套通用底盘，你的 `DefaultModel` 站在这个底盘上，把实际计算委托给内部包装的 Qwen3.5 backbone。
