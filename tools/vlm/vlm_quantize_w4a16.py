# SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang
# SPDX-License-Identifier: Apache-2.0

"""Quantize VLM

Author: wanghaiyang
Date: 2025-01-01
"""

import base64
from io import BytesIO

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info  # 需 `pip install qwen-vl-utils`
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_PATH = "/home/data/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "/home/data/Qwen/Qwen3-VL-8B-Instruct-W4A16"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# 1. 加载本地原始模型与多模态 processor
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 2. 校准数据：图文样本
ds = load_dataset("/home/workspace/data/llms-lab/flickr30k", split=f"test[:{NUM_CALIBRATION_SAMPLES}]").shuffle(seed=42)


def preprocess_and_tokenize(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"data:image;base64,{encoded}"},
            {"type": "text", "text": "What does the image show?"},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {k: torch.tensor(v) for k, v in batch[0].items()}


# 3. 关键差异：AWQ 重排 + W4A16 量化，ignore 排除视觉编码器
recipe = [
    AWQModifier(duo_scaling=False),
    QuantizationModifier(
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:.*visual.*"],
    ),
]

# 4. 执行量化（sequential_targets 指向 LLM 主干的 decoder layer）
oneshot(
    model=model,
    tokenizer=MODEL_PATH,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=["Qwen3VLTextDecoderLayer"],
)

model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)