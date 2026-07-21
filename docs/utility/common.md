
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common utilities

>本文主要用于介绍一些常用的命令行工具的简要使用说明。

---

## rsync
```BASH
rsync -avz --exclude 'cache' --dry-run /path/to/project/ /path/to/backup/
```