<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-28 -->

# AutoVLA Message

> 本文用于介绍 AutoVLA 在 SFT VLM 模型时所使用的 Message 的格式。

---

## Message

```PYTHON
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "The autonomous vehicle is equipped..."},
                {"type": "video", "video": ["file://cam1.jpg", ...]},
                {"type": "text", "text": f"velocity={velocity}, instruction={instruction}"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"<think>{cot_text}</think><answer>The final output action is: {action_tokens}</answer>"}
            ]}
        ]
```

## System Content

## User Content

3 个摄像头视角(前/左前/右前) × 每个视角 4 帧时序图,共 12 张图,被组织成 3 个独立的 "video" 块,而不是 12 个独立 image 块。

```PYTHON
user_content = [
    {
        "type": "text", 
        "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right..."
    },
    {
        "type": "text", 
        "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
    },
    {
        "type": "video",
        "min_pixels": 28 * 28 * 128,   # = 100,352
        "max_pixels": 28 * 28 * 128,
        "video": [
            f"file://{front_camera_1}",
            f"file://{front_camera_2}",
            f"file://{front_camera_3}",
            f"file://{front_camera_4}",
        ]
    },
    
    # 前左视说明
    {
        "type": "text", 
        "text": "The second video presents the front-left view..."
    },
    # 前左视 4 帧
    {
        "type": "video", 
        "min_pixels": ..., 
        "max_pixels": ..., 
        "video": [front_left_1..4]
    },
    
    # 前右视说明
    {
        "type": "text", 
        "text": "The third video presents the front-right view..."
    },

    # 前右视 4 帧
    {
        "type": "video", 
        "min_pixels": ..., 
        "max_pixels": ..., 
        "video": [front_right_1..4]
    },
    
    # 车辆状态 + 指令
    {
        "type": "text", 
        "text": f"The current velocity of the vehicle is {velocity:.3f} m/s, "
                f"and the current acceleration is {acceleration:.3f} m/s². "
                f"The driving instruction is: {instruction}. "
                f"Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
    }
]
```

- 28*28*128: 
    - patch size = 14 × 14 像素
    - merge size = 2 × 2 patch
    - 128 个 Patch 对应 128 个 Vision Token
- 3 个摄像头 → 3 个独立 video 块,不混在一起
- 每个 video = 4 帧 @ 2Hz(覆盖过去 2 秒)
- add_vision_id=True 让 token 流里有 Video 1/2/3 编号,便于跨模态对齐

1. 为什么 `AutoVLA` 使用 video 而不是 image 进行编码？

- 触发 Qwen2.5-VL 的时序编码路径：type: video 走 3D 卷积 patch + temporal positional embedding,模型能学到"上一帧车在远处 → 这一帧车在近处 → 推断速度方向"。
- 压缩 token 数：如果用 image 模式,12 张图就是 12 × 128 ≈ 1500+ token,显存吃紧;video 模式只需约 768 token。
- 摄像头之间不共享时序：3 个视角分别打包成 3 个独立 video,避免左右视角被强行做时序对齐(实际上它们是同步采集的,但空间位置不同)。

## Assistant Content

