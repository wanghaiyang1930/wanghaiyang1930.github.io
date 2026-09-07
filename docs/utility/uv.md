# UV

---

## Install

默认会将 uv 安装到 ~/.local/bin 目录下。

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Envirment

```
uv init --python 3.10
uv add torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/
uv sync
source .venv/bin/activate
```