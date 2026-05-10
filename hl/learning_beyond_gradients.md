# Learning Beyond Gradients — 博客总结与启发式学习入门指南

> 整理日期：2026-05-10  
> 博客原文：[https://trinkle23897.github.io/learning-beyond-gradients/](https://trinkle23897.github.io/learning-beyond-gradients/)  [中文](https://trinkle23897.github.io/learning-beyond-gradients/#zh)
> 作者：Jiayi Weng（清华大学 / OpenAI 背景，强化学习研究者）

---

## 1. 核心观点

博客开篇提出一个现象：

> "As LLM agents get stronger, coding gets faster and better. But the phenomenon I find more interesting is different: a coding agent can keep reading failures, editing code, adding tests, and watching results..."

**核心洞察：LLM Agent 在推理时通过"读错误 → 改代码 → 跑测试 → 观察结果"的循环，实现了一种不依赖梯度下降的"学习"。**

---

## 2. 主要论点

### 2.1 推理即优化（Reasoning as Gradient）

- 传统 ML 训练靠梯度下降更新参数
- LLM Agent 在推理时通过自然语言推理来"定向更新"代码/方案，这等价于一种"梯度"
- 当模型推理能力足够强时，这种定向更新比穷举搜索（tree search）更高效

### 2.2 从 Tree Search 到 Directed Update 的跨越

- **弱模型**：推理不可靠，需要靠 tree search 穷举多个候选方案，用验证分数排序
- **强模型**：推理本身就能给出"方向"，一次推理 ≈ 一步梯度下降，效率远超随机搜索
- 存在一个 **crossover point**：模型推理能力超过某个阈值后，directed update 全面优于 tree search

### 2.3 无梯度的"学习循环"

- Agent 的 trial-and-error 循环（写代码 → 报错 → 分析 → 修改）本质上是一个优化过程
- 不需要反向传播，不需要更新权重，但确实在"学习"解决问题
- 这是一种 **inference-time learning**，发生在推理阶段而非训练阶段

---

## 3. 关联论文

这篇博客的学术基础是 **Reasoning as Gradient: Scaling MLE Agents Beyond Tree Search**（arXiv: 2603.01692）。

核心发现：
- 在 10 个不同能力的模型上做了 scaling 实验
- 弱模型靠 tree search 补偿推理不足
- 强模型用 reasoning-as-gradient 方式效率更高
- 随着 LLM 推理能力提升，穷举搜索变得越来越低效，定向更新的优势越来越大

> "As LLM reasoning capabilities improve, exhaustive enumeration becomes increasingly inefficient compared to directed updates, analogous to how accurate gradients enable efficient descent over random search."

---

## 四、如何入门"启发式学习"

基于这篇博客的视角，"启发式学习"在当下最实用的形态就是 **LLM Agent 的推理时优化**。

### 4.1 理解核心范式

| 范式 | 机制 | 典型场景 |
|------|------|----------|
| 梯度学习 | 反向传播 + SGD | 模型训练 |
| Tree Search | 穷举 + 评分排序 | 弱模型 Agent |
| Reasoning as Gradient | 推理 + 定向修改 | 强模型 Agent |

### 4.2 动手实践

#### 实践一：搭建 Coding Agent 循环

让 LLM 写代码 → 运行 → 读错误 → 修改，观察它如何"收敛"。

现成工具：
- Claude Code
- OpenAI Codex
- Cursor Agent

#### 实践二：对比 Tree Search vs Directed Update

用同一个任务，分别尝试：
- **方案 A**：生成 N 个方案，取验证分数最优的（tree search）
- **方案 B**：生成 1 个方案，迭代改进（directed update）

观察哪种方式在你的场景下更高效。

#### 实践三：阅读关键论文

| 论文 | 主题 | 链接 |
|------|------|------|
| Reasoning as Gradient | 理论框架：推理即梯度 | [arXiv:2603.01692](https://arxiv.org/abs/2603.01692) |
| A Self-Improving Coding Agent | 自我改进的 Agent 实现 | [arXiv:2504.15228](https://arxiv.org/html/2504.15228) |
| Fine-tuning LLM Agents without Fine-tuning LLMs | 不改权重的 Agent 学习 | [arXiv:2508.16153](https://arxiv.org/abs/2508.16153) |
| RL for MLE Agents | 用 RL 训练 Agent 迭代策略 | [arXiv:2509.01684](https://arxiv.org/html/2509.01684v1) |
| Where LLM Agents Fail and How They Learn From Failures | Agent 从失败中学习 | [arXiv:2509.25370](https://arxiv.org/abs/2509.25370) |

### 4.3 应用场景

1. **自动化 ML 实验** — Agent 自动调参、改模型结构、跑实验、分析结果
2. **代码 Bug 修复** — 读报错 → 定位 → 修复 → 验证的闭环
3. **Prompt 优化** — 用 TextGrad 等框架，把文本反馈当作"梯度"来优化 prompt
4. **自动化测试生成** — Agent 观察覆盖率不足的地方，迭代补充测试
5. **Kaggle 竞赛自动化** — MLE Agent 自动尝试不同特征工程和模型方案

---

## 五、关键 Takeaway

> **当模型足够强时，一次好的推理 > 一百次随机尝试。**
>
> 启发式学习的现代形态就是让强 LLM 在推理时做"有方向的搜索"，而不是盲目穷举。

---

## 六、延伸资源

- [Learning Beyond Gradients（原文博客）](https://trinkle23897.github.io/learning-beyond-gradients/)
- [Reasoning as Gradient（Microsoft Research）](https://www.microsoft.com/en-us/research/publication/reasoning-as-gradient-scaling-mle-agents-beyond-tree-search/)
- [MLE-bench: Evaluating ML Agents（OpenAI）](https://openai.com/index/mle-bench/)
- [TextGrad: Automatic Differentiation via Text](https://github.com/zou-group/textgrad)
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314)
