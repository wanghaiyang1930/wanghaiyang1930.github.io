<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-28 -->

# AutoVLA Inputs

> 本文主要用于介绍 `AutoVLA` 的 Inputs 数据结构。

---

## Inputs

```TEXT
<|im_start|>system
You are an Advanced Driver Assistance...
<|im_end|>

<|im_start|>user
The autonomous vehicle is equipped with three cameras...
The first video presents the front view, comprising four sequential frames sampled at 2 Hz.
Video 1: <|vision_start|><|video_pad|><|video_pad|>...<|vision_end|>     ← 前视 ~256 video tokens
The second video presents the front-left view...
Video 2: <|vision_start|><|video_pad|>...<|vision_end|>                  ← 前左视
The third video presents the front-right view...
Video 3: <|vision_start|><|video_pad|>...<|vision_end|>                  ← 前右视
The current velocity is 5.234 m/s, acceleration is 0.123 m/s². 
The driving instruction is: turn right. 
Based on this information, plan the action trajectory for the next five seconds.
<|im_end|>

<|im_start|>assistant
<think>
This is a complex scenario requiring additional reasoning.
{CoT 推理}
</think>
<answer>
The final output action is: <action_123><action_456>...<action_789>
</answer>
<|im_end|>

```