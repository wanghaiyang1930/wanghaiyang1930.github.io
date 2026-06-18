<!-- SPDX-FileCopyrightText: Copyright (c) 2026 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-18 -->

# mmproj

> 简要介绍 mmproj


 mmproj = multimodal projector，多模态投影器。它是 VLM 中连接视觉世界和语言世界的「翻译层」。

  模型架构里它在哪

  VLM 的前向流程可以拆成三段：

  图像 ──► [视觉编码器 ViT] ──► 视觉特征向量
                                      │
                                      ▼
                              [跨模态投影器 mmproj]   ← 这个
                                      │
                                      ▼
                              「视觉 token」（伪装成语言 token 的向量）
                                      │
                                      ▼
                              [语言模型 LLM]
                                      │
                                      ▼
                                    输出文本

视觉编码器（如 ViT/CLIP）把图像变成高维向量，但这些向量的「形状」和语言模型所期望的 token embedding 不匹配 —— 维度不一样、语义空间也不一样。
mmproj 就是干这个翻译的：通常是一两层 MLP（或更复杂的 resampler/Q-Former），把视觉编码器的输出映射到语言模型的 embedding 空间，让 LLM 觉得「这就是一些奇怪的 token」，从而无缝接续生成。

为什么 llama.cpp 把它单独成一个文件

历史原因 + 工程原因：

  1. 架构多样性：不同 VLM 的视觉部分差异极大（ViT 大小、patch size、投影器结构、是否带 resampler、图像预处理逻辑等），把这些塞进 libllama 主库会让代码膨胀且难以维护。
  2. 职责分离：libllama 只管语言部分（attention、KV cache、采样等），视觉部分独立到 libmtmd。两边可以各自迭代。
  3. 运行时灵活性：同一个语言主体可以搭配不同的 mmproj（比如不同分辨率版本），或者纯文本场景下完全不加载视觉部分。

  所以 llama.cpp 的 VLM 部署是两个文件：
  - model.gguf — 语言主体（量化）
  - mmproj.gguf — 视觉编码器 + 投影器（不量化或轻量化）

  注意 mmproj.gguf 里通常包含整个视觉编码器（ViT），不只是投影 MLP。命名沿用「投影器」是历史习惯，实际打包的是「视觉前端整体」。

  为什么 mmproj 不量化

  这是 VLM 量化最关键的工程决策，和 vLLM 侧 ignore=["re:visual.*"] 是同一回事：

  - 视觉编码器（ViT）的权重对低比特量化非常敏感 —— 一个 patch 编码错误就会让所有下游 token 一起跑偏，OCR、计数、小目标识别会显著掉点。
  - 投影器层数少、参数量小（Qwen3-VL 8B 整体里 ViT + 投影器加起来通常只占模型总参数的 10-20%），不量化对总体积影响有限。
  - 语言主体 7B 量化省下的显存远多于 ViT 不量化多占的部分，整体仍是净收益。

  所以**「量化语言主体 + 保留 mmproj 原精度」**几乎是所有 VLM 量化方案的通行做法，无论你用 llama.cpp 还是 vLLM/llm-compressor。

  一个直观类比

  把 VLM 想成一个翻译团队：
  - ViT 是「看图的眼睛」—— 把画面转成内部草稿。
  - mmproj 是「速记员」—— 把眼睛的草稿转成语言模型听得懂的格式。
  - LLM 是「写报告的作者」—— 根据速记员转过来的内容生成文本。

  llama.cpp 里 mmproj.gguf 这个文件，其实是把「眼睛 + 速记员」打包到了一起 —— 这俩跟「作者」职责完全不同，单独成文件、单独保留高精度，整个流水线才稳。