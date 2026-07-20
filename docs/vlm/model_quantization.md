<!-- SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Quantization

> 本文面向需要在本地完成大语言模型（LLM）和视觉语言模型（VLM）量化与部署的开发者，以「跑通」为目标，覆盖最常用的两条工具链：`llama.cpp`（GGUF 量化与轻量推理）和 `vLLM`（高吞吐推理服务）。本文不展开量化算法原理与精度调优，仅提供能直接复制运行的命令。
>
> - 默认硬件：2× NVIDIA RTX 3090（共 48 GB 显存，Ampere 架构，**不支持 FP8**）。
> - 默认示例模型：`Qwen2.5-7B-Instruct`（LLM）、`Qwen3-VL-8B-Instruct`（VLM）。
> - 默认输入：本地已下载的 Hugging Face 格式权重目录（不依赖在线拉取模型）。

---

## 方案对比

| 工具 | 量化格式 | 推理后端 | 适用场景 | 7B 量化后体积 |
|---|---|---|---|---|
| `llama.cpp` | GGUF (`Q4_K_M` 等) | CPU / CUDA / Metal | 本地快速验证、单机离线推理、低显存机器 | ~4.5 GB |
| `vLLM` | W4A16 (compressed-tensors) | CUDA（必须） | 服务化部署、批量评测、高并发、多卡张量并行 | ~5 GB（权重） |

经验法则：
- **本地快速验证一下输出** → `llama.cpp`。
- **OpenAI 兼容服务做评测或批量推理** → `vLLM`。

---

## llama.cpp

### 适用场景
- 单机离线推理与精度抽查；不需要高并发。
- 需要在低显存（≤24 GB）单卡上跑 13B-32B 量化模型。
- 想把模型分发为单文件（`.gguf`），便于在不同机器之间复用。

### 1. 环境准备

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
uv venv .llama
source .llama/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

### 2. 编译（启用 CUDA 后端）

**CUDA**
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release -j
```

**PPU**
```BASH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75" -DGGML_CUDA_FA_ALL_QUANTS=OFF
cmake --build build --config Release -j
```

- `-DGGML_CUDA=ON`：启用 CUDA 后端（不需要 GPU 加速时省略）。
- `-DCMAKE_CUDA_ARCHITECTURES="86"`：3090 是 sm_86，显式指定可减小二进制体积、加快编译。

### 3. HF → GGUF（FP16）

将本地 Hugging Face 格式权重转换为 FP16 精度的 GGUF 文件：

```bash
python convert_hf_to_gguf.py /path/to/Qwen2.5-7B-Instruct \
    --outfile qwen2.5-7b-instruct-f16.gguf \
    --outtype f16
```

转换出的 FP16 文件大小约为原模型的一半（7B 模型约 14 GB）。

**VLM（Qwen3-VL-8B-Instruct）需要分别生成两个 GGUF 文件**：语言模型主体 + 视觉编码器投影器（`mmproj`）。

```bash
# 1) 语言主体（与 LLM 同款命令）
python convert_hf_to_gguf.py /home/data/Qwen/Qwen3-VL-8B-Instruct \
    --outfile qwen3-vl-8b-instruct-f16.gguf \
    --outtype f16

# 2) 视觉编码器（添加 --mmproj 参数）
python convert_hf_to_gguf.py /home/data/Qwen/Qwen3-VL-8B-Instruct \
    --outfile qwen3-vl-8b-instruct-mmproj-f16.gguf \
    --outtype f16 \
    --mmproj
```

> Qwen3-VL 是 llama.cpp 中较新加入的多模态架构，主线已合并相关 PR 但尚未列入 [`tools/mtmd/README.md`](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd) 的官方支持矩阵。如果转换报架构不识别，先 `git pull` 升级到 master 最新提交后重新编译；持续报错则建议改走 vLLM 路径。

### 4. 量化（Q4_K_M）

`Q4_K_M` 是模型大小与精度的常用平衡点（约 4.5 bpw）：

```bash
./build/bin/llama-quantize \
    ./qwen2.5-7b-instruct-f16.gguf \
    ./qwen2.5-7b-instruct-Q4_K_M.gguf \
    Q4_K_M
```

7B 模型量化后约 4.5 GB。其他常用方案：`Q5_K_M`（更高精度，~5.5 GB）、`Q8_0`（接近无损，~7.5 GB）。

**VLM 量化策略**：与 vLLM 侧 `ignore visual.*` 同理 —— **只量化语言主体，`mmproj` 文件保留 FP16**（视觉编码器对低比特敏感）。

```bash
# 仅量化语言主体
./build/bin/llama-quantize \
    ./qwen3-vl-8b-instruct-f16.gguf \
    ./qwen3-vl-8b-instruct-Q4_K_M.gguf \
    Q4_K_M

# mmproj 保持原样不量化（如需更小可改用 q8_0，避免 Q4_K_M）
```

部署时同时引用「量化后的语言模型」+「未量化的 mmproj」两个文件。

### 5. 本地部署与推理验证

**单次推理**（确认能跑通）：

```bash
./build/bin/llama-cli \
    -m ./qwen2.5-7b-instruct-Q4_K_M.gguf \
    -p "你好，请用一句话介绍一下自己。" \
    -n 128 -ngl 99
```

`-ngl 99` 表示把全部层卸载到 GPU，单张 3090 可承载 7B-Q4_K_M。

**作为 HTTP 服务运行**（OpenAI 兼容接口，默认 8080 端口）：

```bash
./build/bin/llama-server \
    -m ./qwen2.5-7b-instruct-Q4_K_M.gguf \
    -ngl 99 --host 0.0.0.0 --port 8080
```

调用示例：

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 64
    }'
```

**VLM（Qwen3-VL-8B-Instruct）推理**：使用 `llama-mtmd-cli`（统一的多模态 CLI），同时加载语言模型和 `mmproj`：

```bash
./build/bin/llama-mtmd-cli \
      -m /home/data/Qwen/Qwen3-VL-8B-Instruct-FP16-Q4_K_M.gguf \
      --mmproj /home/data/Qwen/Qwen3-VL-8B-Instruct-MMPROJ-FP16.gguf \
      --image /home/workspace/source/work/parking.png \
      -p "请描述这张图片。" \
      -n 256 -c 8192 -ngl 99 \
      --split-mode layer --tensor-split 1,1 \
      --image-min-tokens 1024
```

**作为 HTTP 服务运行**（OpenAI 兼容多模态接口）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

./build/bin/llama-server \
    -m /home/data/Qwen/Qwen3-VL-8B-Instruct-F16.gguf \
    --mmproj /home/data/Qwen/Qwen3-VL-8B-Instruct-F16-MMPROJ-F16.gguf \
    -n 1024 -c 262144 -ngl 99 -np 32 \
    -b 8192 --ubatch-size 4096 \
    -fa off \
    --image-min-tokens 1024 \
    --chat-format chatml \
    --media-path /home/workspace/source/temp \
    --host 0.0.0.0 --port 8082
```

- -n 单次生成的最大 token 数
- -c KV cache 的总上下文长度，意味着输入提示和模型输出加起来的 Token 长度不超过 256k(262144)
- -b 在处理 Prompt（输入提示词） 时，一次性往 GPU 里塞多少个 Token 进行并行计算，该值越大，Prefill 阶段计算密度越高、速度越快，但瞬时显存（激活值）占用也越高
- --ubatch-size 可以把它理解为一个“动态任务拆分器”，它会把一个大的计算任务（-b 定义的批次）智能地拆分成多个更小的、适合你硬件处理能力的小批次。针对 96G 显存，推荐：-b 16384 或 -b 32768
- -np 并行 slot 数（最重要的吞吐参数）
- ngl 0 = 纯 CPU（很慢），99（或任意大数）= 全部层都上 GPU
- --image-min-tokens 处理图像时，模型会将图片切分成小块并转化为Token，该参数设定了这个转化过程的下限。如果设置较高，能让模型捕捉更多图像细节，显著提升对复杂图像（如包含密集文字的文档、图表）的理解准确率，但会增加计算量和处理时间
- --chat-format chatml: 不同的语言模型期望收到的对话历史格式是不同的。这个参数就是告诉 llama.cpp 模型“习惯”哪种格式，以便它能将问题、历史回复、系统指令等，正确地组装成模型能理解的提示词（Prompt）。Qwen 系列模型：通常使用 chatml 格式（注意：暂不支持）
- --reasoning-budget 0 可选参数，关闭 Thinking 模式

带图像的调用示例：

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://parking.png"}},
            {"type": "text", "text": "请描述这张图片。"}
        ]}],
        "max_tokens": 256
    }'
```

> `libmtmd` 体系仍在快速迭代，不同 commit 之间参数和行为可能变化（参考 mtmd README 中「**very heavy development, breaking changes are expected**」的官方提醒）。生产环境建议固定到一个验证过的 llama.cpp 版本。

---

## vLLM

### 适用场景
- OpenAI 兼容 API、并发请求、批量评测。
- 双卡或多卡张量并行部署（本机 2 × 3090 场景）。
- 持续运行的推理服务（而非一次性脚本）。

注意⚠️ vLLM 自身不做量化：量化由 `llm-compressor` 完成，vLLM 负责加载量化产物（`compressed-tensors` 格式）并提供推理服务。

### 1. 环境准备

```bash
uv venv vllm
source vllm/bin/activate
uv pip install vllm

uv vent llm
source llm/bin/activate
uv pip install llmcompressor
```

`vllm` 会自动安装匹配版本的 `torch` / CUDA 运行时，无需手动指定 CUDA 版本。
官方文档明确指出，vLLM 和 LLM Compressor 应在不同的 Python 环境中安装使用，因为它们可能存在依赖冲突。

### 2. 量化（本地权重 → W4A16）

将下方脚本保存为 `quantize_w4a16.py`，对本地的 `Qwen2.5-7B-Instruct` 进行 W4A16 量化：

量化脚本：[llm_quantize_w4a16](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/tools/vlm/llm_quantize_w4a16.py)

```bash
python quantize_w4a16.py
```

7B 模型量化耗时约 15-30 分钟，输出权重约 5 GB。

> **关于多 GPU**：GPTQ 算法逐层串行执行，多 GPU 不会缩短量化时间。它的作用是「装下模型」：7B-FP16（~14 GB）单张 3090 即可装下，量化时第二张卡空闲；当原模型放不下单卡时（如 32B-FP16 ~64 GB），在 `from_pretrained(..., device_map="auto")` 中加上该参数，权重会自动切分到两张 3090 上完成量化。
>
> **校准数据**：示例使用 `ultrachat_200k`，是通用 chat 模型最常用的校准集，首次运行会下载到 `~/.cache/huggingface/`。如果环境无法访问外网，把数据集预先离线下载到本地，再用 `load_dataset("path/to/local_dataset")` 替换即可。

### 3. 部署（双卡张量并行）

```bash
vllm serve /path/to/Qwen2.5-7B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --host 0.0.0.0 --port 8000
```

参数说明：
- `--quantization compressed-tensors`：W4A16 走这个 backend（不是 `awq` 或 `gptq`）。模型目录的 `config.json` 中已记录量化配置时也可省略，由 vLLM 自动识别。
- `--tensor-parallel-size 2`：双卡 3090 张量并行。
- `--max-model-len 8192`：根据显存余量调整；本机配置 7B-W4A16 可开到 32K 以上。

### 4. 调用验证

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/Qwen2.5-7B-Instruct-W4A16",
        "messages": [{"role": "user", "content": "你好，请用一句话介绍一下自己。"}],
        "max_tokens": 128
    }'
```

返回 `choices[0].message.content` 不为空即部署成功。

---

## VLM 量化注意事项

VLM 由「视觉编码器（ViT）+ 跨模态投影 + 语言模型主干（LLM）」三部分构成。**视觉编码器对低比特量化非常敏感**：将 ViT 一并量化为 W4A16 后，OCR、计数、小目标识别等任务往往出现明显掉点。**实践中通常保留视觉编码器为原精度（FP16/BF16），只对 LLM 主干量化。**

### 1. 量化脚本（Qwen3-VL-8B-Instruct）

相比纯 LLM 脚本，关键差异有三处：
1. 模型类换为 `Qwen3VLForConditionalGeneration`。
2. `ignore` 中通过正则排除整个视觉子模块。
3. 校准数据改用图文混合数据集（如 `lmms-lab/flickr30k`），并提供 `data_collator` 与 `sequential_targets`。

其他可选校准数据集：
- HuggingFaceM4/COCO：更大（~120K 图），物体检测标注丰富，偏重目标识别类 VLM；
- liuhaotian/LLaVA-Instruct-150K：多轮对话 + 图像，指令更复杂，强调指令跟随能力的 VLM；

此外，Qwen3-VL 官方推荐的量化配方不再使用单独的 `GPTQModifier`，而是 `AWQModifier`（权重重排）+ `QuantizationModifier`（W4A16 量化）两阶段组合。

量化脚本：[vlm_quantize_w4a16](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/tools/vlm/vlm_quantize_w4a16.py)

> `ignore` 与 `sequential_targets` 仅适用于 Qwen3-VL；其他 VLM（InternVL、LLaVA、Pixtral 等）需根据其模块命名调整，可通过 `print(model)` 查看实际模块树，或参考 [llm-compressor 官方多模态示例](https://github.com/vllm-project/llm-compressor/tree/main/examples/multimodal_vision)。

### 2. 部署差异

```bash
vllm serve /home/data/Qwen/Qwen3-VL-8B-Instruct-W4A16 \
    --quantization compressed-tensors \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 4}' \
    --host 0.0.0.0 --port 8000
```

相比 LLM 多出来的关键参数：
- `--limit-mm-per-prompt`：单条请求允许的多模态输入数量上限，按业务调整。评测脚本一次喂 N 张图就把 `image` 设为 N。

### 3. 调用验证（带图像输入）

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/data/Qwen/Qwen3-VL-8B-Instruct-W4A16",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///path/to/test.jpg"}},
            {"type": "text", "text": "请描述这张图片。"}
        ]}],
        "max_tokens": 256
    }'
```

---

## 常见问题

- **`Q4_K_M` 与 `W4A16` 是同一个东西吗？**
  心智模型上接近（都是 4-bit 权重量化），但**不是同一格式**：`Q4_K_M` 是 llama.cpp 的 GGUF 内部格式，只能被 llama.cpp 系工具加载；`W4A16` 是 compressed-tensors 描述的算法 + 比特配置，只能被 vLLM / `transformers` 加载。两者不能互换，按部署工具决定走哪条路径。

- **3090 能用 FP8 吗？**
  不能。FP8 需要 Hopper（H100/H800）或 Ada Lovelace（L40S 等）的硬件支持，3090 是 Ampere 架构，无 FP8 执行单元，启用后只会回退到模拟实现，反而更慢。

- **2× 3090 (48 GB) 能跑多大的模型？**
  W4A16 量化下，部署时实际可用约 42-44 GB 显存（需为 KV cache 与激活预留），可稳定承载 32B 量级模型（如 `Qwen2.5-32B-Instruct-W4A16`，权重约 18-20 GB）；Q4_K_M GGUF 同尺寸约 18 GB，也能跑。70B 即使 W4A16（~35 GB 权重）也会因 KV cache 紧张而难以稳定运行，建议换成更小模型。

- **量化后精度会掉多少？**
  本文不展开，请用项目内的评测流程实测。经验上 `Q4_K_M` / `W4A16` 在指令跟随、常识问答类任务上掉点 1-3%；**VLM 视觉相关任务在视觉编码器未量化时基本与原模型一致**，掉点主要来自语言端。

- **`vllm serve` 启动后报 KV cache 不够？**
  调小 `--max-model-len`，或显式设置 `--gpu-memory-utilization 0.85`（默认 0.9）。

- **`llama-quantize` 报「too many tensors are quantized」或 imatrix 相关警告？**
  说明使用了较新的量化方案。直接坚持 `Q4_K_M` 即可；若要更高精度可改用 `Q5_K_M`，避免在不熟悉的情况下使用 `IQ*` 系列。
