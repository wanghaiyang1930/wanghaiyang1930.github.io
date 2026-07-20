<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniCPM5-1B Instruction

## 综述
面壁智能  MiniCPM5-1B 模型 https://github.com/openbmb/minicpm
https://www.youtube.com/watch?v=ox1mW2N9Z_Y&list=PL6IhU4tr7yTzA9RJGJjbaZGPUgt7vBXry&t=676s

MiniCPM5-1B：亮点与核心训练方法整理
一句话结论
MiniCPM5-1B 的关键不在于发明全新的模型架构，而在于用一套完整的数据分层、分阶段训练、专项强化学习和 On-Policy Distillation，把一个 1B 级标准 Llama 模型训练成偏重 Agent、代码、数学推理和端侧部署的高性能小模型。
整体路线可以概括为：
分层数据管理 → Base Training → Mid-Training → 双模式 SFT → 多领域专项 RL Teacher → OPD 融合 → 最终统一模型

## 一、MiniCPM5-1B 的基本情况

MiniCPM5-1B 是 MiniCPM5 系列目前发布的第一个模型，发布时间为 2026 年 5 月 19 日。
核心参数：
总参数量：10.81 亿
非 Embedding 参数：约 6.80 亿
模型层数：24 层
Attention：GQA，16 个 Q Head、2 个 KV Head
上下文长度：128K
模型架构：标准 LlamaForCausalLM
模型类型：文本 Causal Language Model
开源协议：Apache 2.0
同一模型支持 Think / No Think 两种模式
它没有自定义模型代码，也不依赖专用算子，主流推理框架可以直接加载。(Hugging Face)

## 二、MiniCPM5-1B 的主要亮点

### 1. 1B 体量，重点强化 Agent、代码和数学

官方在推理、知识、代码、指令跟随、数学、逻辑和 Agentic 等评测上给出的平均分为：
MiniCPM5-1B：42.57
同尺寸优秀开源模型最高分：35.61
官方对比对象包括 LFM2.5-1.2B-Thinking、Qwen3-0.6B/Think 和 Qwen3.5-0.8B/Think。MiniCPM5 的优势主要集中在：
工具调用
代码生成
竞赛数学
复杂推理
需要注意，这些数字来自官方汇总评测，目前更适合看作“同尺寸竞争力证明”，仍需要更多第三方真实场景测试。

### 2. 同一份权重支持“思考”和“不思考”

模型通过 Chat Template 中的 enable_thinking 参数切换：
Think：适合数学、代码和复杂推理
No Think：适合普通对话、简单问答和低时延场景
这不是分别训练和发布两个模型，而是同一 checkpoint 兼顾快速响应和深度推理。官方建议：
Think：temperature=0.9，top_p=0.95
No Think：temperature=0.7，top_p=0.95

### 3. 原生 128K 长上下文

MiniCPM5-1B 在 1B 级模型上提供 131,072 Token 上下文，适合：
本地文档助手
代码仓库理解
长文本摘要
多轮 Agent 任务
本地 RAG
但 128K 是“可输入长度”，不等于模型在整个 128K 范围内都具备同等强度的精确检索和推理能力，实际使用仍需专项测试。(Hugging Face)

### 4. 面向端侧的标准化部署

官方提供多种格式：
BF16
GGUF
MLX / 4-bit
支持的运行框架包括：
Transformers
vLLM
SGLang
llama.cpp
Ollama
LM Studio
MLX
ArcLight
由于使用标准 Llama 架构，不需要为模型开发专用 Kernel 或维护模型代码分支。

### 5. Agent 工具调用能力比较完整

MiniCPM5 使用 XML 风格输出工具调用，SGLang 已提供专用的 minicpm5 Tool Parser，可以直接转换为 OpenAI 兼容的 tool_calls。
因此，它的产品定位不是单纯的“本地聊天模型”，而是更偏：
本地 Coding Agent
端侧工具调用 Agent
桌面助手
垂直业务 Agent

## 三、核心训练流程

### 阶段一：Base Training

Base Training 分为两个过程：
Stable Training
Decay Training
主要目标是建立：
基础语言能力
广泛知识
基础推理能力
稳定的训练状态
后续 Mid-Training 和 Post-Training 的模型基础
官方没有公开 Base Training 的总 Token 数、数据混合比例、Batch Size 和具体学习率曲线。
公开的数据资源主要包括：
Ultra-FineWeb
Ultra-FineWeb-L3
UltraData-Math

### 阶段二：Mid-Training

Mid-Training 的作用不是继续无差别地堆通用数据，而是：
加强目标领域能力
调整模型的数据分布
加强数学、代码、推理等高价值能力
为后续 SFT 和 RL 建立更好的初始化
UltraData 的核心思想是：不同质量的数据，不应该从头到尾混在一起训练。
其数据分为五级：
L0：未经处理的原始数据，主要用于归档
L1：经过清洗和去重的数据
L2：经过模型筛选、信息密度较高的数据
L3：经过重写、合成或人工优化的“教材级”数据
L4：经过组织和验证的结构化知识，如知识库、数据库和知识图谱
其中大规模 L1/L2 数据适合建立通用能力，更昂贵、更高质量的 L3 数据应重点放在 Mid-Training、Annealing 等后期关键阶段。实验表明，后期逐步提高数据质量，优于从头到尾把不同质量数据混合训练。(arXiv)

### 阶段三：SFT

MiniCPM5 的 SFT 规模非常大，官方口径为：
200B Token Deep-Thinking SFT
200B Token Hybrid-Thinking SFT
合计约 400B SFT Token。
两个数据阶段分别承担不同目标：
Deep-Thinking SFT
重点训练：
长推理链
数学推理
代码推理
复杂问题分解
深度思考格式
Hybrid-Thinking SFT
重点让模型学会：
哪些问题需要思考
哪些问题不需要输出长推理
在 Think 与 No Think 之间切换
保持普通对话和指令跟随能力
这也是同一模型能够同时支持 Think 和 No Think 的重要训练基础。
阶段四：多领域专项 RL
团队没有直接用一个统一奖励模型解决所有问题，而是分别训练多个专项 RL Teacher，领域包括：
数学
代码
闭卷知识问答
写作
指令跟随
长上下文
通用对话
RL 数据和信号包括：
DAPO-Math-17K
TriviaQA
NQ-Open
LongWriter-Zero-RLData
合成的可验证 RLVR 数据
Pair-wise RLHF 偏好信号
这种方式相当于先培养多个“领域老师”，再把不同老师的能力合并回一个小模型。
阶段五：OPD 能力融合
最后一个关键阶段是 On-Policy Distillation，简称 OPD。
普通离线蒸馏通常是：
Teacher 生成答案，Student 模仿 Teacher 的答案。
OPD 则是：
Student 自己生成答案，Teacher 在 Student 实际走过的每一个 Token 状态上进行指导。
因此，三种训练方式的区别是：
SFT：离策略、密集监督
RL：在策略、稀疏结果奖励
OPD：在策略、密集 Token 级监督
OPD 同时兼顾了：
Student 自己的真实生成分布
Teacher 提供的细粒度指导
比单纯结果奖励更高的数据效率
比纯离线蒸馏更少的分布偏移 (Thinking Machines Lab)

## 四、MiniCPM5 使用 OPD 的具体技巧

### 1. 使用 Reverse KL 作为 Advantage

MiniCPM5 在 RL 框架中，用 Teacher 与 Student 分布之间的 Reverse KL 作为 Advantage，替代传统的答案正确性验证奖励。
这样每个 Token 都能获得训练信号，而不只是整道题最后得到一个 0 或 1。

### 2. 只计算 Top-K Token

完整计算整个词表上的 KL 成本很高。
MiniCPM5 的做法是：
分别取 Teacher 和 Student 的 Top-K Logits
对两个 Top-K Token 集合取并集
在并集上计算 Reverse KL
用较低成本近似完整词表的分布差异
这在监督精度和训练效率之间取得了平衡。(Hugging Face)

### 3. 复用专项 RL Teacher 的训练 Prompt

OPD 阶段直接复用各领域 RL Teacher 使用过的同域 Prompt，不需要重新构建一套蒸馏数据。
这一设计有两个价值：
降低数据整理成本
让 Teacher 和 Student 的输入分布更一致
OPD 研究显示，Teacher 和 Student 的思考模式越接近，Top-K Token 重叠越高，OPD 越容易成功；使用 Teacher 后训练阶段的数据和 Prompt Template，也有助于提升蒸馏效果。(ar5iv)

### 4. Teacher 必须真正获得“新能力”

OPD 研究发现，大模型 Teacher 分数更高，并不必然意味着它能有效教会 Student。
成功的 OPD 通常需要满足两个条件：
Teacher 与 Student 的思考模式基本兼容
Teacher 拥有 Student 尚未获得的新知识或新能力
MiniCPM5 先对各个 Teacher 进行专项 RL，再进行 OPD。从设计逻辑看，专项 RL 的作用之一就是让 Teacher 获得可迁移的新能力，而不是简单用一个“更大但训练方式相同”的模型做蒸馏。(ar5iv)

## 五、控制“小模型越想越长”

小型推理模型做 RL 时，一个常见问题是：
为了提高奖励，模型不断增加推理长度，最终出现 Length Explosion。
MiniCPM5 在 Reasoning RL 中使用了两阶段长度调度：
第一阶段允许模型充分探索推理路径
第二阶段逐步约束输出长度
官方公布的结果是，经过 RL + OPD：
数学、代码和指令跟随平均提升 16 分
输出触及最大 Token 上限的比例下降 29 个百分点
也就是说，最终模型不仅更强，而且减少了无意义的超长推理。
这里与 JustRL 的关系值得注意：
JustRL 强调简单、固定超参数的 RL 基线
它发现过度使用长度惩罚、复杂调度和强验证器，可能压缩探索空间
MiniCPM5 借鉴其简化思路，但针对实际出现的过长输出，又加入了两阶段长度控制
其思路不是“训练技巧越多越好”，而是只在明确观察到问题时增加控制策略。(ar5iv)

## 六、这套训练方法最值得借鉴的地方

MiniCPM5 真正值得关注的不是某一个单点算法，而是以下组合：

### 1. 数据不再只是“清洗后混合”
而是根据质量和用途分级：
大规模普通数据建立基础能力
高信息密度数据加强目标领域
教材级数据放在关键后期
可验证数据用于 RL

### 2. 先培养专项老师，再合并能力

不是要求一个小模型同时通过一套统一 RL 学会所有能力，而是：
数学 Teacher、代码 Teacher、写作 Teacher、QA Teacher分别强化，再通过 OPD 汇总到一个发布模型。
这比直接混合多个不同奖励信号，更容易控制各领域训练质量。

### 3. Think / No Think 从训练阶段统一设计

双模式不是推理阶段临时增加 Prompt，而是在 SFT 数据阶段同时训练：
深度推理
简短回答
普通对话
自动选择推理方式

### 4. OPD 同时解决能力融合和数据效率

专项 RL Teacher 的能力不需要通过重新生成海量离线答案来迁移，直接使用 Student 的在策略 Rollout 和 Teacher 的 Token 分布完成融合。

### 5. 架构尽量标准，复杂度放在训练体系

MiniCPM5 没有为了跑分引入难以部署的特殊模型结构，而是使用标准 Llama 架构，把主要创新放在：
数据
SFT
RL
OPD
部署生态
这对端侧模型尤其重要。

## 七、仓库提供的微调 Skills

仓库提供两个顶层 Agent Skill：
minicpm5-deploy：自动选择部署方案
minicpm5-finetune：自动选择微调框架
微调框架推荐逻辑：
第一次微调：LLaMA-Factory
单张 24GB 以内显卡：Unsloth + QLoRA
需要 DPO / KTO / ORPO：ms-swift 或 TRL
需要最强代码控制能力：TRL + PEFT
OpenMMLab 技术栈：XTuner
几个容易踩坑的点：
LLaMA-Factory 要使用模型自身 Chat Template，不能直接套 llama3
ms-swift 需要显式指定 llama 和 chatml
TRL 使用 assistant_only_loss 时，需要在训练 Chat Template 中加入 {% generation %}
训练时可以临时修改 Chat Template，但推理时必须重新加载原始 Tokenizer
如果 Chat Template 对不齐，最典型的问题就是 Loss 异常或模型输出乱码

## 八、总体评价

MiniCPM5-1B 的核心竞争力可以概括为：
用标准模型架构，通过高强度数据工程和后训练，把一个 1B 模型做成具备长上下文、混合推理、工具调用和代码能力的端侧 Agent 模型。
它最值得研究的是三点：
UltraData 分层数据体系
专项 RL Teacher + OPD 能力融合
Deep-Thinking + Hybrid-Thinking 双模式 SFT
目前公开资料仍有一些空白：
Base Training 和 Mid-Training 的总 Token 数未披露
各类数据的精确混合比例未披露
RL Teacher 的规模和训练超参数未披露
两阶段长度调度的具体长度设置未披露
尚缺少大规模第三方端侧和 Agent 实测
因此，MiniCPM5-1B 已经展示了一套很有参考价值的小模型训练范式，但其官方性能结论仍需要更多独立测试验证。