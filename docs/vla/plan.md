<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-28 -->

# Plan
---

## 目标

通过 VLA 实现车辆的自主漫游。

## 数据准备

- 图像数据；
- 轨迹数据（位置+姿态）；
- 车身数据（速度+加速度+驾驶指令）；
- COT；

## Todo List

1. 多 Camera、多 Timestamp 在 VLM 中如何编码？
2. 车身数据在 VLM 中如何编码？