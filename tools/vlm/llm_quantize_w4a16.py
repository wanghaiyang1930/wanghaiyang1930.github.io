# SPDX-FileCopyrightText: Copyright (c) 2025 wanghaiyang
# SPDX-License-Identifier: Apache-2.0

"""Quantize LLM

Author: wanghaiyang
Date: 2026-06-18
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/path/to/Qwen2.5-7B-Instruct-W4A16"

# 1. 加载本地原始模型
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. 校准数据：512 条 ultrachat 通用指令样本
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
).shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess).map(tokenize, remove_columns=ds.column_names)

# 3. 配置 W4A16 量化（GPTQ 算法，跳过输出层）
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# 4. 执行量化并保存
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)