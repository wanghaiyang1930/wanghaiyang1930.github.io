<!-- SPDX-FileCopyrightText: Copyright (c) 2026 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Author: wanghaiyang -->
<!-- Date: 2026-07-028 -->

## Import&Export

```BASH

docker export -o ubuntu-2404-cuda-125-py310-iparking-dev-v1.3.tar 9cb993cdfbaa

docker import ubuntu-2404-cuda-125-py310-iparking-dev-v1.3.tar.tar ubuntu-2404-cuda-125-py310-iparking-dev:v1.3

```

## Run

```BASH
docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size=128gb \
    --name x-name  \
    --privileged \
    -v /home/data:/home/data \
    -v /home/workspace:/home/workspace \
    -w /home/workspace/source/code \
    ubuntu-2404-cuda-125-py310-dev:v1.4 /bin/bash
```