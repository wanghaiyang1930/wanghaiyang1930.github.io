<!-- SPDX-FileCopyrightText: Copyright (c) 2026 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-06-15 -->

# Data Description

---

> 本文主要用于描述 Alpamayo 1.5 的原始数据需求与内部数据结构。

## Inner Data Structure

```JSON
{
    "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
    "camera_indices": camera_indices,  # (N_cameras,)
    "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
    "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
    "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
    "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
    "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
    "absolute_timestamps": all_timestamps,  # (N_cameras, num_frames)
    "t0_us": t0_us,
    "clip_id": clip_id,
}
```

- `t0_us`: T0 时刻时间戳，即对齐时间戳
- `relative_timestamps`: Type: float, Unit: seconds, Shape: (camera_numer, frame_number), Sort: [0.3s, 0.2s, 0.1s, 0.0s] 
- `absolute_timestamps`: Type: uint64, Unit: microseconds, Shape: (camera_number, frame_number), Sort: [t0-0.3s, t0-0.2s, t0-0.1s, t0]
- `ego_history_xyz`: Default 16 steps, Shape: (B, Group, T, P), T: [t0-1.5s, t0-1.4s, ..., t0-0.1s, t0], P: [x, y, z]
- `ego_future_xyz`: Default 48 steps, Shape: (B, Group, T, P), T: [t0+0.1s, ..., t0+6.4s]