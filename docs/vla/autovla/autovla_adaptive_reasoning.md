<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-28 -->

# AutoVLA Adaptive Reasoning

> 本文主要用于解释 AutoVLA 是如何实现 VLA 模型的双思维模式的。
> AutoVLA 双思维模式不是两个模型,也不是显式分支,而是用同一个自回归 LLM、通过"训练数据 + 输出格式 + 奖励信号"三件套,让模型自己学会该不该想。

---


## 1. 统一输出格式

不管最终是不是要"思考",模型永远都按这个模板输出。区别只在 <think> 块里写什么:

慢思维 (complex scenario) — sft_dataset.py:186-196

```TEXT
<think>
This is a complex scenario requiring additional reasoning.
{完整 CoT:场景分析→关键物体→预测→意图推理→最终动作}
</think>
<answer>
The final output action is: <action_token_1><action_token_2>...
</answer>
```

快思维 (straightforward scenario) — sft_dataset.py:199-211

```TEXT
<think>
This is a straightforward scenario, and a direct decision can be made.
</think>
<answer>
The final output action is: <action_token_1><action_token_2>...
</answer>
```

关键设计:快思维不是"跳过 think",而是 think 块里只写一句"这场景很简单",然后直接给答案。这样自回归过程的结构完全统一,模型只需学会"什么时候多写、什么时候少写"。

## 2. SFT 阶段:用数据教会"什么样的场景值得想"

切换决策的先验来自训练数据本身。在 sft_dataset.py:180-211:

```PYTHON
if isinstance(gt_cot, str):       # 数据里带了 72B 标注好的 CoT
    has_cot = True                # → 喂"复杂场景"模板
else:                              # 数据里没有 CoT
    has_cot = False               # → 喂"简单场景"模板
```

那 gt_cot 哪些样本有、哪些没有?——由 Qwen2.5-VL-72B 在预处理阶段离线标注(INCLUDE_COT=1 时启动),只有 72B 判定为复杂、给得出推理的场景才会带 CoT 标签。所以**"哪些算复杂"实际上是 72B 大模型的判断,被蒸馏到 3B 主模型里**。

此外 SFT 训练 loss 还做了加权 — autovla.py:358-361:

```PYTHON
if hascot[0] == True:
    loss = loss * 40
    loss = loss + action_loss
```

带 CoT 的样本 loss × 40,强迫模型重视推理路径的还原。

## 3. RFT 阶段 (GRPO):用奖励惩罚"不必要的思考"

如果只做 SFT,模型会学到模板但不会自主调度——容易要么全说"复杂",要么全说"简单"。所以 RFT 引入了 CoT 长度惩罚,核心在 autovla.py:170-187:

```PYTHON
cot_penalties = torch.stack([
    torch.sigmoid((len(text) - center) * cot_penalty_coef)
        if "complex scenario" in text.lower()
        else 0.0
    for text in sample['completion_texts']
])
reward = reward - cot_penalty_weight * cot_penalties
```

配合 config 中的:

```YAML
cot_penalty:
  coef: 0.002      # sigmoid 斜率
  center: 400.0    # 长度中点(字符数)
  weight: 0.3      # 总权重
```

奖励函数 = PDM 驾驶分 − 0.3 × sigmoid((len−400) × 0.002) × [输出含 "complex scenario"]

它的含义非常清晰:

| 情况 | 惩罚 |
| --- | --- |
| 输出走了"快思维"分支 (不含 "complex scenario") | 0 — 完全不惩罚 |
| 走了"慢思维"且 CoT 很短 (len ≪ 400) | sigmoid 接近 0,惩罚很小 |
| 走了"慢思维"且 CoT 很长 (len ≫ 400) | sigmoid 接近 1,惩罚 ≈ 0.3 |

这就是"该慢则慢,该快则快"的数学表达:

- 慢思维确实让 PDM 分提高 ⇒ 即使有长度惩罚也值得 ⇒ 模型保留思考。
- 慢思维并没带来更好的轨迹 ⇒ 惩罚 > 收益 ⇒ GRPO 把策略推向"简单场景"分支。
- GRPO 的 group-relative advantage 还会在同一 prompt 的多次采样里,让"短的好答案"获得更高优势,自然压缩冗余推理。

## 4. 推理时如何"抉择"?——没有显式开关

推理流程在 autovla.py:496-525 (AutoVLA.predict),非常朴素:

- 把视频 + 速度 + 加速度 + 驾驶指令塞进 prompt(system prompt 里写着 "If necessary, use step-by-step reasoning... Otherwise, you may directly predict the final driving action.")。
- 调 vlm.generate(...),一路自回归。
- 第一个生成的句子就决定了模式:
    - 如果采样出 "This is a complex scenario requiring additional reasoning." → 接下来会继续生成完整 CoT → 最后才生成 <answer> 和动作 token。
    - 如果采样出 "This is a straightforward scenario, and a direct decision can be made." → 直接闭合 </think> → 进入 <answer> → 几个 token 出完轨迹。
- 从输出 token 流里挑出 >= action_start_id 的 token,送 ActionTokenizer.decode_token_ids_to_trajectory 还原为 10 个 (x,y) 路点(5 秒 × 2 Hz)。

所以"切换"是纯生成式的:模型用同一组权重、同一条采样链,在第一句话上就把模式选好,之后只是顺着该模式继续写下去。

## 5. 总结

SFT 教会模板和先验(哪些场景该想),RFT 用 PDM 奖励 − 长度惩罚教会节制(想得越久代价越大,除非真能开得更好);推理时模型在 <think> 的第一句自己说出 "complex" 或 "straightforward",自回归就顺势走上慢/快分支。整套机制没有外部分类器,也没有 if-else 路由,完全靠 LLM 自己学出来。
