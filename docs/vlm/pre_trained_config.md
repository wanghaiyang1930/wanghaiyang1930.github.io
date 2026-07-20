<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PretrainedConfig Instruction

### PretrainedConfig 是什么？

简单来说，`PretrainedConfig` 就是一个**配置文件的 Python 类**。

想象你在用 Word 写文档，你会设置字体大小、页边距、行间距等。`PretrainedConfig` 就是做同样的事情，只不过它是给 AI 模型设置参数的。

比如：
- 模型有多少层？
- 每层有多少个神经元？
- 使用什么样的激活函数？

这些信息都存在 `PretrainedConfig` 里。

### 为什么需要它？

**问题：** 如果没有配置文件会怎样？

假设你训练了一个模型，过了一周想再加载它。如果没有配置文件，你得记住：
- "哦，这个模型用的是 12 层"
- "隐藏层大小是 768"
- "注意力头数是 12 个"

如果记错了，模型就加载不出来了。

**解决方案：** 用 `PretrainedConfig` 自动保存这些信息

当你保存模型时，配置也会自动保存成一个 `config.json` 文件。下次加载时，系统自动读取配置，不用你记任何东西。

### 最简单的例子

#### 例子 1：下载一个模型的配置

```python
from transformers import AutoConfig

# 从 Hugging Face 下载 Qwen2 模型的配置
config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")

# 看看配置里有什么
print(config)
```

**输出：**
```
Qwen2Config {
  "hidden_size": 3584,           # 每层有 3584 个神经元
  "num_hidden_layers": 28,       # 一共 28 层
  "num_attention_heads": 28,     # 28 个注意力头
  "vocab_size": 152064,          # 词汇表大小
  ...
}
```

这就像打开一个模型的"说明书"，告诉你这个模型是怎么构造的。

#### 例子 2：修改配置创建自己的模型

```python
from transformers import AutoConfig, AutoModel

# 先下载 Qwen2 的配置
config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")

# 修改配置，创建一个小模型
config.hidden_size = 512        # 把神经元数量改小
config.num_hidden_layers = 6    # 把层数改少

# 用这个新配置创建一个模型（随机初始化的，不是预训练的）
model = AutoModel.from_config(config)
print(f"模型参数量：{model.num_parameters() / 1e6:.1f}M")
```

**作用：** 你不用从零开始设计模型架构，只需要下载一个成熟的配置，然后调整几个参数就行。

#### 例子 3：保存和加载配置

```python
from transformers import AutoConfig

# 下载配置
config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")

# 保存到本地
config.save_pretrained("./my_model")

# 这会在 ./my_model 文件夹里生成一个 config.json 文件
```

**生成的 `config.json` 内容：**
```json
{
  "hidden_size": 3584,
  "num_hidden_layers": 28,
  "vocab_size": 152064,
  ...
}
```

下次你可以这样加载：
```python
config = AutoConfig.from_pretrained("./my_model")
```

### 创建自己的配置类

如果你要做一个自定义模型（比如一个机器人控制模型），你可以继承 `PretrainedConfig`。

#### 简单版本：机器人配置

```python
from transformers import PretrainedConfig

class RobotConfig(PretrainedConfig):
    model_type = "robot"  # 给你的模型起个名字
    
    def __init__(
        self,
        num_actions=3,           # 机器人有几个动作（比如：前进、后退、转向）
        max_speed=10.0,          # 最大速度
        sensor_dim=128,          # 传感器输入维度
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.max_speed = max_speed
        self.sensor_dim = sensor_dim

# 使用
config = RobotConfig(num_actions=5, max_speed=20.0)
config.save_pretrained("./my_robot_model")
```

保存后，`config.json` 会是这样：
```json
{
  "model_type": "robot",
  "num_actions": 5,
  "max_speed": 20.0,
  "sensor_dim": 128
}
```

### 总结

| 概念 | 类比 | 实际作用 |
|------|------|----------|
| `PretrainedConfig` | Word 文档的格式设置 | 存储模型的所有超参数 |
| `config.json` | 保存的格式文件 | 持久化配置到硬盘 |
| `from_pretrained()` | 打开文件恢复格式 | 从硬盘加载配置 |
| `save_pretrained()` | 保存格式设置 | 把配置写入硬盘 |
| 自定义配置类 | 自定义模板 | 为特定任务定义专用配置 |

**一句话总结：** `PretrainedConfig` 就是模型的"说明书"，告诉你模型是怎么构造的，怎么保存和加载它。

