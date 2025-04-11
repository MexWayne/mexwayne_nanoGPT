import torch
import torch.onnx
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken

import sys
from model import GPTConfig, GPT
from types import SimpleNamespace

if __name__ == '__main__':
    # === Step 1: 加载 checkpoint ===
    ckpt_path = "./out/ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt["model_args"]
    
    # === Step 2: 构建 GPTConfig（用 SimpleNamespace 模拟）===
    # 如果你有 GPTConfig 类，建议替换 SimpleNamespace 为 GPTConfig
    config = SimpleNamespace(**model_args)
    model = GPT(config)
    
    # === Step 3: 加载 state_dict（清理 _orig_mod 前缀） ===
    raw_state_dict = ckpt["model"]
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    
    # === Step 4: 构造 dummy 输入（batch_size=1, seq_len=128） ===
    vocab_size = config.vocab_size
    dummy_input = torch.randint(0, vocab_size, (1, 128), dtype=torch.long)
    
    # === Step 5: 导出为 ONNX ===
    onnx_path = "./out/ckpt.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    print(f"Exported to: {onnx_path} finished!")