+++
title = 'Debugging Transformers Upgrade with torch DebugMode: v4.57.3 → v5.0'
date = 2026-01-20T10:00:00+00:00
draft = false
tags = ['transformers', 'debugging', 'pytorch', 'deep-learning', 'vibe']
categories = ['Technical']
comments = true
+++

This document tracks the debugging process for accuracy issues encountered when upgrading transformers from v4.57.3 to v5.0, using PyTorch debug mode to compare layer-by-layer outputs.

<!--more-->

## Overview

This document details the debugging process for accuracy issues encountered when upgrading Transformers from v4.57.3 to v5.0. We use PyTorch debug mode to compare layer-by-layer outputs.

## Summary

When upgrading the Transformers library from version 4.57.3 to 5.0, we encountered severe accuracy degradation in the DeepSeek-V3 model. While v4.57.3 generated coherent outputs, v5.0 produced repetitive and garbled text for the same inputs.

**Debugging Approach:**  
We employed PyTorch's `DebugMode` to perform layer-by-layer comparison of tensor operations between the two versions. This tool intercepts torch operations and records input/output hashes, allowing precise identification of where numerical divergence begins.

**Root Cause:**  
The investigation revealed a critical issue at the `unsqueeze` operation in the Rotary Position Embedding (RoPE) layer. In v5.0, the tensor hash value was abnormally large (7.372678e+31 vs. expected ~3.95), indicating uninitialized data. The root cause was:

- Models in v5.0 initialize on the `meta` device for memory efficiency
- The `_init_weights` method was commented out to reduce initialization overhead  
- This left the `self.inv_freq` buffer in `DeepseekV3RotaryEmbedding` uninitialized with garbage values

**Solution:**  
Re-enabling the `_init_weights` implementation properly initializes the RoPE frequency buffer, resolving the accuracy issues.

**Key Takeaway:**  
This debugging journey demonstrates how PyTorch's DebugMode can trace subtle numerical bugs through complex model architectures. A single uninitialized tensor can cascade into completely incorrect outputs, and systematic layer-by-layer comparison is essential for diagnosing such issues during library upgrades.

## Full Model Output Comparison

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
  <tr>
    <th style="width: 50%; text-align: center; padding: 10px; font-weight: bold; border-bottom: 2px solid #ddd;">v4.57.3</th>
    <th style="width: 50%; text-align: center; padding: 10px; font-weight: bold; border-bottom: 2px solid #ddd;">v5.0.0.dev0</th>
  </tr>
  <tr>
    <td style="width: 50%; padding: 15px; background: #f6f8fa; border: 1px solid #d0d7de; vertical-align: top;">
      <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; font-size: 0.9em;">The capital of France is Paris. Paris is the most populous city in France, with a population of over 12 million people in its metropolitan area. The city is located in the north</pre>
    </td>
    <td style="width: 50%; padding: 15px; background: #fff5f5; border: 1px solid #ffd7d5; vertical-align: top;">
      <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; font-size: 0.9em;">The capital of France is Paris.

Answer 1. Paris is the capital city.

The capital city is ParisParis.

The capital of France is Paris. The capital of France is Paris</pre>
    </td>
  </tr>
</table>

## Introduce torch `DebugMode`

`DebugMode` inherits from `TorchDispatchMode` and intercepts torch operation calls (`__torch_function__` or `__torch_dispatch__`) to record the input and output of each operation. More details can be found in the [PyTorch documentation](https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html).

Below is an example from the PyTorch documentation:
```python
from torch._inductor.decomposition import decomps_to_exclude
import torch
from torch.utils._debug_mode import DebugMode

def run_model(model, data, *, compile_with=None):
    if compile_with is not None:
        model = torch.compile(model, backend=compile_with)
    with DebugMode(record_output=True) as dm, DebugMode.log_tensor_hashes(
        hash_inputs=True,
    ):
        dm_out = model(*data)
    return dm, dm_out

class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x).mm(x.T)

inputs = (torch.randn(4, 4),)
dm_eager, _ = run_model(Toy(), inputs)
print("Eager mode:")
print(dm_eager.debug_string())

# Output:
# Eager mode:
#     aten::relu(t: f32[4, 4])  ->  t: f32[4, 4]  # {'input_hash': ((14.893587063997984,), {}), 'hash': 7.259045481681824}
#     aten::permute(t: f32[4, 4], [1, 0])  ->  t: f32[4, 4]  # {'input_hash': ((14.893587063997984, [None, None]), {}), 'hash': 14.893587063997984}
#     aten::mm(t: f32[4, 4], t: f32[4, 4])  ->  t: f32[4, 4]  # {'input_hash': ((7.259045481681824, 14.893587063997984), {}), 'hash': 26.860059764236212}
```

## One Layer Debug String Comparison


```python
# ds_in_v5.py
import psutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache


# clear cache to reload modified code
clear_import_cache()
model_name = "/mnt/disk5/unsloth/DeepSeek-R1-BF16"
# model_name = "/mnt/disk8/deepseek-ai/DeepSeek-V2-Lite-Chat"
device = "cpu"
from loguru import logger


# Memory monitor implementation


def dump_cur_ram(msg: str = ""):
    process = psutil.Process()
    current_ram = process.memory_info().rss / 1024**2  # MB
    logger.warning(f"[Memory] {msg} Current RAM usage: {round(current_ram, 2)}MB")


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def disable_concat_experts():
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)


from torch.utils._debug_mode import DebugMode

def main(args):
    model_name = args.model_name
    fixed_seed(42)
    disable_concat_experts()
    from v5_patch import apply_transformer_patches

    apply_transformer_patches()
    with torch.no_grad():
        trust_remote_code = False
        dump_cur_ram("before model load")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            device_map="cpu",  # device_map="auto",
        )
        msg = "The capital of France is"
        model.eval()
        print(model)
        inputs = tokenizer(msg, return_tensors="pt").to(device)
        if args.debug:
            with (
                DebugMode(
                    record_stack_trace=args.record_stack_trace,
                    record_ids=True,
                ) as dm,
                DebugMode.log_tensor_hashes(
                    hash_inputs=True,
                ),
            ):
                # outputs = model.generate(**inputs, max_new_tokens=32)
                print(f"Inputs: {inputs['input_ids']}")
                res = model(inputs["input_ids"])

            print(dm.debug_string(show_stack_trace=True))
            print(res)
            exit(0)
        inputs = tokenizer(msg, return_tensors="pt").to("cpu")

        outputs = model.generate(**inputs, max_new_tokens=32)
        decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_output)
        exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # input model path
    parser.add_argument("--model_name", type=str, default=model_name, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the quantized model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--record_stack_trace", "--stack", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)

```

```bash
python ds_in_v5.py --debug
```

<details open>
<summary><strong>Click to collapse detailed layer-by-layer comparison</strong></summary>

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
  <tr>
    <th style="width: 50%; text-align: center; padding: 10px; font-weight: bold; border-bottom: 2px solid #ddd;">v4.57.3</th>
    <th style="width: 50%; text-align: center; padding: 10px; font-weight: bold; border-bottom: 2px solid #ddd;">v5.0.0.dev0</th>
  </tr>
  <tr>
    <td style="width: 50%; padding: 0; background: #f6f8fa; border: 1px solid #d0d7de; vertical-align: top;">
      <div style="max-height: 600px; overflow-y: auto; padding: 15px;">
        <pre style="margin: 0; font-size: 11px; line-height: 1.4; font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">
   1     aten::embedding(t$0: bf16[129280, 7168], t$1: i64[1, 6], 128815)  ->  t$2: bf16[1, 6, 7168]  # {'input_hash': ((27375526.173154227, 16171.0, None), {}), 'hash': 788.8026814290788}
   2     aten::arange.start(0, 6, device=cpu, pin_memory=False)  ->  t$3: i64[6]  # {'hash': 15.0}
   3     aten::unsqueeze(t$3: i64[6], 0)  ->  t$4: i64[1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}
   4     aten::unsqueeze(t$5: f32[32], 0)  ->  t$6: f32[1, 32]  # {'input_hash': ((3.9489362656597677, None), {}), 'hash': 3.9489362656597677}
   5     aten::unsqueeze(t$6: f32[1, 32], 2)  ->  t$7: f32[1, 32, 1]  # {'input_hash': ((3.9489362656597677, None), {}), 'hash': 3.9489362656597677}
   6     aten::expand(t$7: f32[1, 32, 1], [1, -1, 1])  ->  t$8: f32[1, 32, 1]  # {'input_hash': ((3.9489362656597677, [None, None, None]), {}), 'hash': 3.9489362656597677}
   7     aten::unsqueeze(t$4: i64[1, 6], 1)  ->  t$9: i64[1, 1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}
   8     aten::_to_copy(t$9: i64[1, 1, 6], dtype=torch.float32)  ->  t$10: f32[1, 1, 6]  # {'input_hash': ((15.0,), {'dtype': None}), 'hash': 15.0}
   9     aten::expand(t$8: f32[1, 32, 1], [1, 32, 1])  ->  t$11: f32[1, 32, 1]  # {'input_hash': ((3.9489362656597677, [None, None, None]), {}), 'hash': 3.9489362656597677}
  10     aten::view(t$11: f32[1, 32, 1], [1, 32, 1])  ->  t$12: f32[1, 32, 1]  # {'input_hash': ((3.9489362656597677, [None, None, None]), {}), 'hash': 3.9489362656597677}
  11     aten::expand(t$10: f32[1, 1, 6], [1, 1, 6])  ->  t$13: f32[1, 1, 6]  # {'input_hash': ((15.0, [None, None, None]), {}), 'hash': 15.0}
  12     aten::view(t$13: f32[1, 1, 6], [1, 1, 6])  ->  t$14: f32[1, 1, 6]  # {'input_hash': ((15.0, [None, None, None]), {}), 'hash': 15.0}
  13     aten::bmm(t$12: f32[1, 32, 1], t$14: f32[1, 1, 6])  ->  t$15: f32[1, 32, 6]  # {'input_hash': ((3.9489362656597677, 15.0), {}), 'hash': 59.23404385846038}
  14     aten::_unsafe_view(t$15: f32[1, 32, 6], [1, 32, 6])  ->  t$16: f32[1, 32, 6]  # {'input_hash': ((59.23404385846038, [None, None, None]), {}), 'hash': 59.23404385846038}
  15     aten::transpose.int(t$16: f32[1, 32, 6], 1, 2)  ->  t$17: f32[1, 6, 32]  # {'input_hash': ((59.23404385846038, None, None), {}), 'hash': 59.23404385846038}
  16     aten::cat(['t$17: f32[1, 6, 32]', 't$17: f32[1, 6, 32]'], -1)  ->  t$18: f32[1, 6, 64]  # {'input_hash': (([59.23404385846038, 59.23404385846038], None), {}), 'hash': 118.46808771692076}
  17     aten::cos(t$18: f32[1, 6, 64])  ->  t$19: f32[1, 6, 64]  # {'input_hash': ((118.46808771692076,), {}), 'hash': 355.8641229439527}
  18     aten::mul.Tensor(t$19: f32[1, 6, 64], 1.0)  ->  t$20: f32[1, 6, 64]  # {'input_hash': ((355.8641229439527, None), {}), 'hash': 355.8641229439527}
  19     aten::sin(t$18: f32[1, 6, 64])  ->  t$21: f32[1, 6, 64]  # {'input_hash': ((118.46808771692076,), {}), 'hash': 61.185383417837784}
  20     aten::mul.Tensor(t$21: f32[1, 6, 64], 1.0)  ->  t$22: f32[1, 6, 64]  # {'input_hash': ((61.185383417837784, None), {}), 'hash': 61.185383417837784}
  21     aten::_to_copy(t$20: f32[1, 6, 64], dtype=torch.bfloat16)  ->  t$23: bf16[1, 6, 64]  # {'input_hash': ((355.8641229439527,), {'dtype': None}), 'hash': 355.8790283203125}
  22     aten::_to_copy(t$22: f32[1, 6, 64], dtype=torch.bfloat16)  ->  t$24: bf16[1, 6, 64]  # {'input_hash': ((61.185383417837784,), {'dtype': None}), 'hash': 61.14249849319458}
  23     aten::_to_copy(t$2: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$25: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788,), {'dtype': None}), 'hash': 788.8026814290788}
  24     aten::pow.Tensor_Scalar(t$25: f32[1, 6, 7168], 2)  ->  t$26: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788, None), {}), 'hash': 47.63002008403455}
  25     aten::mean.dim(t$26: f32[1, 6, 7168], [-1], True)  ->  t$27: f32[1, 6, 1]  # {'input_hash': ((47.63002008403455, [None], None), {}), 'hash': 0.006644813169259578}
  26     aten::add.Tensor(t$27: f32[1, 6, 1], 1e-06)  ->  t$28: f32[1, 6, 1]  # {'input_hash': ((0.006644813169259578, None), {}), 'hash': 0.006650813214946538}
  27     aten::rsqrt(t$28: f32[1, 6, 1])  ->  t$29: f32[1, 6, 1]  # {'input_hash': ((0.006650813214946538,), {}), 'hash': 185.91046714782715}
  28     aten::mul.Tensor(t$25: f32[1, 6, 7168], t$29: f32[1, 6, 1])  ->  t$30: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788, 185.91046714782715), {}), 'hash': 23589.8728415073}
  29     aten::_to_copy(t$30: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$31: bf16[1, 6, 7168]  # {'input_hash': ((23589.8728415073,), {'dtype': None}), 'hash': 23589.859999522567}
  30     aten::mul.Tensor(t$32: bf16[7168], t$31: bf16[1, 6, 7168])  ->  t$33: bf16[1, 6, 7168]  # {'input_hash': ((297.11181640625, 23589.859999522567), {}), 'hash': 1156.63240952231}
  31     aten::t(t$34: bf16[1536, 7168])  ->  t$35: bf16[7168, 1536]  # {'input_hash': ((141606.8712687064,), {}), 'hash': 141606.8712687064}
  32     aten::view(t$33: bf16[1, 6, 7168], [6, 7168])  ->  t$36: bf16[6, 7168]  # {'input_hash': ((1156.63240952231, [None, None]), {}), 'hash': 1156.63240952231}
  33     aten::mm(t$36: bf16[6, 7168], t$35: bf16[7168, 1536])  ->  t$37: bf16[6, 1536]  # {'input_hash': ((1156.63240952231, 141606.8712687064), {}), 'hash': 920.8860853910446}
  34     aten::_unsafe_view(t$37: bf16[6, 1536], [1, 6, 1536])  ->  t$38: bf16[1, 6, 1536]  # {'input_hash': ((920.8860853910446, [None, None, None]), {}), 'hash': 920.8860853910446}
  35     aten::_to_copy(t$38: bf16[1, 6, 1536], dtype=torch.float32)  ->  t$39: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446,), {'dtype': None}), 'hash': 920.8860853910446}
  36     aten::pow.Tensor_Scalar(t$39: f32[1, 6, 1536], 2)  ->  t$40: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446, None), {}), 'hash': 288.5757960550672}
  37     aten::mean.dim(t$40: f32[1, 6, 1536], [-1], True)  ->  t$41: f32[1, 6, 1]  # {'input_hash': ((288.5757960550672, [None], None), {}), 'hash': 0.1878748619928956}
  38     aten::add.Tensor(t$41: f32[1, 6, 1], 1e-06)  ->  t$42: f32[1, 6, 1]  # {'input_hash': ((0.1878748619928956, None), {}), 'hash': 0.18788085784763098}
  39     aten::rsqrt(t$42: f32[1, 6, 1])  ->  t$43: f32[1, 6, 1]  # {'input_hash': ((0.18788085784763098,), {}), 'hash': 39.29507780075073}
  40     aten::mul.Tensor(t$39: f32[1, 6, 1536], t$43: f32[1, 6, 1])  ->  t$44: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446, 39.29507780075073), {}), 'hash': 5729.73890671269}
  41     aten::_to_copy(t$44: f32[1, 6, 1536], dtype=torch.bfloat16)  ->  t$45: bf16[1, 6, 1536]  # {'input_hash': ((5729.73890671269,), {'dtype': None}), 'hash': 5729.775722026825}
  42     aten::mul.Tensor(t$46: bf16[1536], t$45: bf16[1, 6, 1536])  ->  t$47: bf16[1, 6, 1536]  # {'input_hash': ((681.26416015625, 5729.775722026825), {}), 'hash': 2617.230959892273}
  43     aten::t(t$48: bf16[24576, 1536])  ->  t$49: bf16[1536, 24576]  # {'input_hash': ((98730.36440096781,), {}), 'hash': 98730.36440096781}
  44     aten::view(t$47: bf16[1, 6, 1536], [6, 1536])  ->  t$50: bf16[6, 1536]  # {'input_hash': ((2617.230959892273, [None, None]), {}), 'hash': 2617.230959892273}
  45     aten::mm(t$50: bf16[6, 1536], t$49: bf16[1536, 24576])  ->  t$51: bf16[6, 24576]  # {'input_hash': ((2617.230959892273, 98730.36440096781), {}), 'hash': 58005.382189182565}
  46     aten::_unsafe_view(t$51: bf16[6, 24576], [1, 6, 24576])  ->  t$52: bf16[1, 6, 24576]  # {'input_hash': ((58005.382189182565, [None, None, None]), {}), 'hash': 58005.382189182565}
  47     aten::view(t$52: bf16[1, 6, 24576], [1, 6, -1, 192])  ->  t$53: bf16[1, 6, 128, 192]  # {'input_hash': ((58005.382189182565, [None, None, None, None]), {}), 'hash': 58005.382189182565}
  48     aten::transpose.int(t$53: bf16[1, 6, 128, 192], 1, 2)  ->  t$54: bf16[1, 128, 6, 192]  # {'input_hash': ((58005.382189182565, None, None), {}), 'hash': 58005.382189182565}
  49     aten::split_with_sizes(t$54: bf16[1, 128, 6, 192], [128, 64], -1)  ->  ['t$55: bf16[1, 128, 6, 128]', 't$56: bf16[1, 128, 6, 64]']  # {'input_hash': ((58005.382189182565, [None, None], None), {}), 'hash': [3620.7217003386468, 54384.66048884392]}
  50     aten::t(t$57: bf16[576, 7168])  ->  t$58: bf16[7168, 576]  # {'input_hash': ((60000.702554143965,), {}), 'hash': 60000.702554143965}
  51     aten::view(t$33: bf16[1, 6, 7168], [6, 7168])  ->  t$59: bf16[6, 7168]  # {'input_hash': ((1156.63240952231, [None, None]), {}), 'hash': 1156.63240952231}
  52     aten::mm(t$59: bf16[6, 7168], t$58: bf16[7168, 576])  ->  t$60: bf16[6, 576]  # {'input_hash': ((1156.63240952231, 60000.702554143965), {}), 'hash': 925.7674539089203}
  53     aten::_unsafe_view(t$60: bf16[6, 576], [1, 6, 576])  ->  t$61: bf16[1, 6, 576]  # {'input_hash': ((925.7674539089203, [None, None, None]), {}), 'hash': 925.7674539089203}
  54     aten::split_with_sizes(t$61: bf16[1, 6, 576], [512, 64], -1)  ->  ['t$62: bf16[1, 6, 512]', 't$63: bf16[1, 6, 64]']  # {'input_hash': ((925.7674539089203, [None, None], None), {}), 'hash': [447.7636544704437, 478.00379943847656]}
  55     aten::_to_copy(t$62: bf16[1, 6, 512], dtype=torch.float32)  ->  t$64: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437,), {'dtype': None}), 'hash': 447.7636544704437}
  56     aten::pow.Tensor_Scalar(t$64: f32[1, 6, 512], 2)  ->  t$65: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437, None), {}), 'hash': 2416.9021114688285}
  57     aten::mean.dim(t$65: f32[1, 6, 512], [-1], True)  ->  t$66: f32[1, 6, 1]  # {'input_hash': ((2416.9021114688285, [None], None), {}), 'hash': 4.7205121368169785}
  58     aten::add.Tensor(t$66: f32[1, 6, 1], 1e-06)  ->  t$67: f32[1, 6, 1]  # {'input_hash': ((4.7205121368169785, None), {}), 'hash': 4.720518007874489}
  59     aten::rsqrt(t$67: f32[1, 6, 1])  ->  t$68: f32[1, 6, 1]  # {'input_hash': ((4.720518007874489,), {}), 'hash': 9.387228786945343}
  60     aten::mul.Tensor(t$64: f32[1, 6, 512], t$68: f32[1, 6, 1])  ->  t$69: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437, 9.387228786945343), {}), 'hash': 681.568125214475}
  61     aten::_to_copy(t$69: f32[1, 6, 512], dtype=torch.bfloat16)  ->  t$70: bf16[1, 6, 512]  # {'input_hash': ((681.568125214475,), {'dtype': None}), 'hash': 681.4855184555054}
  62     aten::mul.Tensor(t$71: bf16[512], t$70: bf16[1, 6, 512])  ->  t$72: bf16[1, 6, 512]  # {'input_hash': ((3.804391235113144, 681.4855184555054), {}), 'hash': 5.108524536015466}
  63     aten::t(t$73: bf16[32768, 512])  ->  t$74: bf16[512, 32768]  # {'input_hash': ((54175.308287066175,), {}), 'hash': 54175.308287066175}
  64     aten::view(t$72: bf16[1, 6, 512], [6, 512])  ->  t$75: bf16[6, 512]  # {'input_hash': ((5.108524536015466, [None, None]), {}), 'hash': 5.108524536015466}
  65     aten::mm(t$75: bf16[6, 512], t$74: bf16[512, 32768])  ->  t$76: bf16[6, 32768]  # {'input_hash': ((5.108524536015466, 54175.308287066175), {}), 'hash': 111.4665225475328}
  66     aten::_unsafe_view(t$76: bf16[6, 32768], [1, 6, 32768])  ->  t$77: bf16[1, 6, 32768]  # {'input_hash': ((111.4665225475328, [None, None, None]), {}), 'hash': 111.4665225475328}
  67     aten::view(t$77: bf16[1, 6, 32768], [1, 6, -1, 256])  ->  t$78: bf16[1, 6, 128, 256]  # {'input_hash': ((111.4665225475328, [None, None, None, None]), {}), 'hash': 111.4665225475328}
  68     aten::transpose.int(t$78: bf16[1, 6, 128, 256], 1, 2)  ->  t$79: bf16[1, 128, 6, 256]  # {'input_hash': ((111.4665225475328, None, None), {}), 'hash': 111.4665225475328}
  69     aten::split_with_sizes(t$79: bf16[1, 128, 6, 256], [128, 128], -1)  ->  ['t$80: bf16[1, 128, 6, 128]', 't$81: bf16[1, 128, 6, 128]']  # {'input_hash': ((111.4665225475328, [None, None], None), {}), 'hash': [26.415770137362415, 85.05075241017039]}
  70     aten::view(t$63: bf16[1, 6, 64], [1, 1, 6, 64])  ->  t$82: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, [None, None, None, None]), {}), 'hash': 478.00379943847656}
  71     aten::unsqueeze(t$23: bf16[1, 6, 64], 1)  ->  t$83: bf16[1, 1, 6, 64]  # {'input_hash': ((355.8790283203125, None), {}), 'hash': 355.8790283203125}
  72     aten::unsqueeze(t$24: bf16[1, 6, 64], 1)  ->  t$84: bf16[1, 1, 6, 64]  # {'input_hash': ((61.14249849319458, None), {}), 'hash': 61.14249849319458}
  73     aten::view(t$56: bf16[1, 128, 6, 64], [1, 128, 6, 32, 2])  ->  t$85: bf16[1, 128, 6, 32, 2]  # {'input_hash': ((54384.66048884392, [None, None, None, None, None]), {}), 'hash': 54384.66048884392}
  74     aten::transpose.int(t$85: bf16[1, 128, 6, 32, 2], 4, 3)  ->  t$86: bf16[1, 128, 6, 2, 32]  # {'input_hash': ((54384.66048884392, None, None), {}), 'hash': 54384.66048884392}
  75     aten::clone(t$86: bf16[1, 128, 6, 2, 32], memory_format=torch.contiguous_format)  ->  t$87: bf16[1, 128, 6, 2, 32]  # {'input_hash': ((54384.66048884392,), {'memory_format': None}), 'hash': 54384.66048884392}
  76     aten::_unsafe_view(t$87: bf16[1, 128, 6, 2, 32], [1, 128, 6, 64])  ->  t$88: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, [None, None, None, None]), {}), 'hash': 54384.66048884392}
  77     aten::view(t$82: bf16[1, 1, 6, 64], [1, 1, 6, 32, 2])  ->  t$89: bf16[1, 1, 6, 32, 2]  # {'input_hash': ((478.00379943847656, [None, None, None, None, None]), {}), 'hash': 478.00379943847656}
  78     aten::transpose.int(t$89: bf16[1, 1, 6, 32, 2], 4, 3)  ->  t$90: bf16[1, 1, 6, 2, 32]  # {'input_hash': ((478.00379943847656, None, None), {}), 'hash': 478.00379943847656}
  79     aten::clone(t$90: bf16[1, 1, 6, 2, 32], memory_format=torch.contiguous_format)  ->  t$91: bf16[1, 1, 6, 2, 32]  # {'input_hash': ((478.00379943847656,), {'memory_format': None}), 'hash': 478.00379943847656}
  80     aten::_unsafe_view(t$91: bf16[1, 1, 6, 2, 32], [1, 1, 6, 64])  ->  t$92: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, [None, None, None, None]), {}), 'hash': 478.00379943847656}
  81     aten::mul.Tensor(t$88: bf16[1, 128, 6, 64], t$83: bf16[1, 1, 6, 64])  ->  t$93: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, 355.8790283203125), {}), 'hash': 51562.806232601404}
  82     aten::slice.Tensor(t$88: bf16[1, 128, 6, 64], 3, 0, 32)  ->  t$94: bf16[1, 128, 6, 32]  # {'input_hash': ((54384.66048884392, None, None, None), {}), 'hash': 28109.381856679916}
  83     aten::slice.Tensor(t$88: bf16[1, 128, 6, 64], 3, 32, 9223372036854775807)  ->  t$95: bf16[1, 128, 6, 32]  # {'input_hash': ((54384.66048884392, None, None, None), {}), 'hash': 26275.278632164}
  84     aten::neg(t$95: bf16[1, 128, 6, 32])  ->  t$96: bf16[1, 128, 6, 32]  # {'input_hash': ((26275.278632164,), {}), 'hash': 26275.278632164}
  85     aten::cat(['t$96: bf16[1, 128, 6, 32]', 't$94: bf16[1, 128, 6, 32]'], -1)  ->  t$97: bf16[1, 128, 6, 64]  # {'input_hash': (([26275.278632164, 28109.381856679916], None), {}), 'hash': 54384.66048884392}
  86     aten::mul.Tensor(t$97: bf16[1, 128, 6, 64], t$84: bf16[1, 1, 6, 64])  ->  t$98: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, 61.14249849319458), {}), 'hash': 6145.598498088773}
  87     aten::add.Tensor(t$93: bf16[1, 128, 6, 64], t$98: bf16[1, 128, 6, 64])  ->  t$99: bf16[1, 128, 6, 64]  # {'input_hash': ((51562.806232601404, 6145.598498088773), {}), 'hash': 54342.5701687336}
  88     aten::mul.Tensor(t$92: bf16[1, 1, 6, 64], t$83: bf16[1, 1, 6, 64])  ->  t$100: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, 355.8790283203125), {}), 'hash': 469.1174964904785}
  89     aten::slice.Tensor(t$92: bf16[1, 1, 6, 64], 3, 0, 32)  ->  t$101: bf16[1, 1, 6, 32]  # {'input_hash': ((478.00379943847656, None, None, None), {}), 'hash': 171.16246032714844}
  90     aten::slice.Tensor(t$92: bf16[1, 1, 6, 64], 3, 32, 9223372036854775807)  ->  t$102: bf16[1, 1, 6, 32]  # {'input_hash': ((478.00379943847656, None, None, None), {}), 'hash': 306.8413391113281}
  91     aten::neg(t$102: bf16[1, 1, 6, 32])  ->  t$103: bf16[1, 1, 6, 32]  # {'input_hash': ((306.8413391113281,), {}), 'hash': 306.8413391113281}
  92     aten::cat(['t$103: bf16[1, 1, 6, 32]', 't$101: bf16[1, 1, 6, 32]'], -1)  ->  t$104: bf16[1, 1, 6, 64]  # {'input_hash': (([306.8413391113281, 171.16246032714844], None), {}), 'hash': 478.00379943847656}
  93     aten::mul.Tensor(t$104: bf16[1, 1, 6, 64], t$84: bf16[1, 1, 6, 64])  ->  t$105: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, 61.14249849319458), {}), 'hash': 20.15657148777973}
  94     aten::add.Tensor(t$100: bf16[1, 1, 6, 64], t$105: bf16[1, 1, 6, 64])  ->  t$106: bf16[1, 1, 6, 64]  # {'input_hash': ((469.1174964904785, 20.15657148777973), {}), 'hash': 478.46177673339844}
  95     aten::expand(t$106: bf16[1, 1, 6, 64], [1, 128, 6, -1])  ->  t$107: bf16[1, 128, 6, 64]  # {'input_hash': ((478.46177673339844, [None, None, None, None]), {}), 'hash': 61243.107421875}
  96     aten::cat(['t$55: bf16[1, 128, 6, 128]', 't$99: bf16[1, 128, 6, 64]'], -1)  ->  t$108: bf16[1, 128, 6, 192]  # {'input_hash': (([3620.7217003386468, 54342.5701687336], None), {}), 'hash': 57963.29186907224}
  97     aten::cat(['t$80: bf16[1, 128, 6, 128]', 't$107: bf16[1, 128, 6, 64]'], -1)  ->  t$109: bf16[1, 128, 6, 192]  # {'input_hash': (([26.415770137362415, 61243.107421875], None), {}), 'hash': 61269.52319201236}
  98     aten::lift_fresh(t$110: bf16[0])  ->  t$110: bf16[0]  # {'input_hash': ((0.0,), {}), 'hash': 0.0}
  99     aten::lift_fresh(t$111: bf16[0])  ->  t$111: bf16[0]  # {'input_hash': ((0.0,), {}), 'hash': 0.0}
 100     aten::cat(['t$110: bf16[0]', 't$109: bf16[1, 128, 6, 192]'], -2)  ->  t$112: bf16[1, 128, 6, 192]  # {'input_hash': (([0.0, 61269.52319201236], None), {}), 'hash': 61269.52319201236}
 101     aten::cat(['t$111: bf16[0]', 't$81: bf16[1, 128, 6, 128]'], -2)  ->  t$113: bf16[1, 128, 6, 128]  # {'input_hash': (([0.0, 85.05075241017039], None), {}), 'hash': 85.05075241017039}
 102     aten::_to_copy(t$108: bf16[1, 128, 6, 192], dtype=torch.float32)  ->  t$114: f32[1, 128, 6, 192]  # {'input_hash': ((57963.29186907224,), {'dtype': None}), 'hash': 57963.29186907224}
 103     aten::_to_copy(t$112: bf16[1, 128, 6, 192], dtype=torch.float32)  ->  t$115: f32[1, 128, 6, 192]  # {'input_hash': ((61269.52319201236,), {'dtype': None}), 'hash': 61269.52319201236}
 104     aten::_to_copy(t$113: bf16[1, 128, 6, 128], dtype=torch.float32)  ->  t$116: f32[1, 128, 6, 128]  # {'input_hash': ((85.05075241017039,), {'dtype': None}), 'hash': 85.05075241017039}
 105     aten::mul.Scalar(t$114: f32[1, 128, 6, 192], 0.3677414565436974)  ->  t$117: f32[1, 128, 6, 192]  # {'input_hash': ((57963.29186907224, None), {}), 'hash': 21315.505901487195}
 106     aten::ones([6, 6], dtype=torch.bool, layout=torch.strided, device=cpu)  ->  t$118: b8[6, 6]  # {'hash': 36.0}
 107     aten::tril(t$118: b8[6, 6])  ->  t$119: b8[6, 6]  # {'input_hash': ((36.0,), {}), 'hash': 21.0}
 108     aten::scalar_tensor(-inf, dtype=torch.float32, device=cpu)  ->  t$120: f32[]  # {'hash': inf}
 109     aten::scalar_tensor(0.0, dtype=torch.float32, layout=torch.strided, device=cpu)  ->  t$121: f32[]  # {'hash': 0.0}
 110     aten::where.self(t$119: b8[6, 6], t$121: f32[], t$120: f32[])  ->  t$122: f32[6, 6]  # {'input_hash': ((21.0, 0.0, inf), {}), 'hash': inf}
 111     aten::transpose.int(t$115: f32[1, 128, 6, 192], -2, -1)  ->  t$123: f32[1, 128, 192, 6]  # {'input_hash': ((61269.52319201236, None, None), {}), 'hash': 61269.52319201236}
 112     aten::mul.Scalar(t$123: f32[1, 128, 192, 6], 0.3677414565436974)  ->  t$124: f32[1, 128, 192, 6]  # {'input_hash': ((61269.52319201236, None), {}), 'hash': 22531.34425791007}
 113     aten::expand(t$117: f32[1, 128, 6, 192], [1, 128, 6, 192])  ->  t$125: f32[1, 128, 6, 192]  # {'input_hash': ((21315.505901487195, [None, None, None, None]), {}), 'hash': 21315.505901487195}
 114     aten::view(t$125: f32[1, 128, 6, 192], [128, 6, 192])  ->  t$126: f32[128, 6, 192]  # {'input_hash': ((21315.505901487195, [None, None, None]), {}), 'hash': 21315.505901487195}
 115     aten::expand(t$124: f32[1, 128, 192, 6], [1, 128, 192, 6])  ->  t$127: f32[1, 128, 192, 6]  # {'input_hash': ((22531.34425791007, [None, None, None, None]), {}), 'hash': 22531.34425791007}
 116     aten::view(t$127: f32[1, 128, 192, 6], [128, 192, 6])  ->  t$128: f32[128, 192, 6]  # {'input_hash': ((22531.34425791007, [None, None, None]), {}), 'hash': 22531.34425791007}
 117     aten::bmm(t$126: f32[128, 6, 192], t$128: f32[128, 192, 6])  ->  t$129: f32[128, 6, 6]  # {'input_hash': ((21315.505901487195, 22531.34425791007), {}), 'hash': 22071.485942557454}
 118     aten::_unsafe_view(t$129: f32[128, 6, 6], [1, 128, 6, 6])  ->  t$130: f32[1, 128, 6, 6]  # {'input_hash': ((22071.485942557454, [None, None, None, None]), {}), 'hash': 22071.485942557454}
 119     aten::add.Tensor(t$130: f32[1, 128, 6, 6], t$122: f32[6, 6])  ->  t$131: f32[1, 128, 6, 6]  # {'input_hash': ((22071.485942557454, inf), {}), 'hash': inf}
 120     aten::_safe_softmax(t$131: f32[1, 128, 6, 6], -1)  ->  t$132: f32[1, 128, 6, 6]  # {'input_hash': ((inf, None), {}), 'hash': 768.0000004165083}
 121     aten::_to_copy(t$132: f32[1, 128, 6, 6], dtype=torch.bfloat16)  ->  t$133: bf16[1, 128, 6, 6]  # {'input_hash': ((768.0000004165083,), {'dtype': None}), 'hash': 768.0411031043295}
 122     aten::expand(t$132: f32[1, 128, 6, 6], [1, 128, 6, 6])  ->  t$134: f32[1, 128, 6, 6]  # {'input_hash': ((768.0000004165083, [None, None, None, None]), {}), 'hash': 768.0000004165083}
 123     aten::view(t$134: f32[1, 128, 6, 6], [128, 6, 6])  ->  t$135: f32[128, 6, 6]  # {'input_hash': ((768.0000004165083, [None, None, None]), {}), 'hash': 768.0000004165083}
 124     aten::expand(t$116: f32[1, 128, 6, 128], [1, 128, 6, 128])  ->  t$136: f32[1, 128, 6, 128]  # {'input_hash': ((85.05075241017039, [None, None, None, None]), {}), 'hash': 85.05075241017039}
 125     aten::view(t$136: f32[1, 128, 6, 128], [128, 6, 128])  ->  t$137: f32[128, 6, 128]  # {'input_hash': ((85.05075241017039, [None, None, None]), {}), 'hash': 85.05075241017039}
 126     aten::bmm(t$135: f32[128, 6, 6], t$137: f32[128, 6, 128])  ->  t$138: f32[128, 6, 128]  # {'input_hash': ((768.0000004165083, 85.05075241017039), {}), 'hash': 64.43761555664555}
 127     aten::_unsafe_view(t$138: f32[128, 6, 128], [1, 128, 6, 128])  ->  t$139: f32[1, 128, 6, 128]  # {'input_hash': ((64.43761555664555, [None, None, None, None]), {}), 'hash': 64.43761555664555}
 128     aten::_to_copy(t$139: f32[1, 128, 6, 128], dtype=torch.bfloat16)  ->  t$140: bf16[1, 128, 6, 128]  # {'input_hash': ((64.43761555664555,), {'dtype': None}), 'hash': 64.43932099102312}
 129     aten::transpose.int(t$140: bf16[1, 128, 6, 128], 1, 2)  ->  t$141: bf16[1, 6, 128, 128]  # {'input_hash': ((64.43932099102312, None, None), {}), 'hash': 64.43932099102312}
 130     aten::clone(t$141: bf16[1, 6, 128, 128], memory_format=torch.contiguous_format)  ->  t$142: bf16[1, 6, 128, 128]  # {'input_hash': ((64.43932099102312,), {'memory_format': None}), 'hash': 64.43932099102312}
 131     aten::view(t$142: bf16[1, 6, 128, 128], [1, 6, -1])  ->  t$143: bf16[1, 6, 16384]  # {'input_hash': ((64.43932099102312, [None, None, None]), {}), 'hash': 64.43932099102312}
 132     aten::t(t$144: bf16[7168, 16384])  ->  t$145: bf16[16384, 7168]  # {'input_hash': ((402437.33954404993,), {}), 'hash': 402437.33954404993}
 133     aten::view(t$143: bf16[1, 6, 16384], [6, 16384])  ->  t$146: bf16[6, 16384]  # {'input_hash': ((64.43932099102312, [None, None]), {}), 'hash': 64.43932099102312}
 134     aten::mm(t$146: bf16[6, 16384], t$145: bf16[16384, 7168])  ->  t$147: bf16[6, 7168]  # {'input_hash': ((64.43932099102312, 402437.33954404993), {}), 'hash': 240.59403831884265}
 135     aten::_unsafe_view(t$147: bf16[6, 7168], [1, 6, 7168])  ->  t$148: bf16[1, 6, 7168]  # {'input_hash': ((240.59403831884265, [None, None, None]), {}), 'hash': 240.59403831884265}
 136     aten::add.Tensor(t$2: bf16[1, 6, 7168], t$148: bf16[1, 6, 7168])  ->  t$149: bf16[1, 6, 7168]  # {'input_hash': ((788.8026814290788, 240.59403831884265), {}), 'hash': 832.5981237888336}
 137     aten::_to_copy(t$149: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$150: f32[1, 6, 7168]  # {'input_hash': ((832.5981237888336,), {'dtype': None}), 'hash': 832.5981237888336}
 138     aten::pow.Tensor_Scalar(t$150: f32[1, 6, 7168], 2)  ->  t$151: f32[1, 6, 7168]  # {'input_hash': ((832.5981237888336, None), {}), 'hash': 58.39140548157849}
 139     aten::mean.dim(t$151: f32[1, 6, 7168], [-1], True)  ->  t$152: f32[1, 6, 1]  # {'input_hash': ((58.39140548157849, [None], None), {}), 'hash': 0.008146122156176716}
 140     aten::add.Tensor(t$152: f32[1, 6, 1], 1e-06)  ->  t$153: f32[1, 6, 1]  # {'input_hash': ((0.008146122156176716, None), {}), 'hash': 0.008152122201863676}
 141     aten::rsqrt(t$153: f32[1, 6, 1])  ->  t$154: f32[1, 6, 1]  # {'input_hash': ((0.008152122201863676,), {}), 'hash': 169.3051872253418}
 142     aten::mul.Tensor(t$150: f32[1, 6, 7168], t$154: f32[1, 6, 1])  ->  t$155: f32[1, 6, 7168]  # {'input_hash': ((832.5981237888336, 169.3051872253418), {}), 'hash': 22537.589959858786}
 143     aten::_to_copy(t$155: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$156: bf16[1, 6, 7168]  # {'input_hash': ((22537.589959858786,), {'dtype': None}), 'hash': 22537.0666513443}
 144     aten::mul.Tensor(t$157: bf16[7168], t$156: bf16[1, 6, 7168])  ->  t$158: bf16[1, 6, 7168]  # {'input_hash': ((122.01806432008743, 22537.0666513443), {}), 'hash': 605.1124907890335}
 145     aten::t(t$159: bf16[18432, 7168])  ->  t$160: bf16[7168, 18432]  # {'input_hash': ((101450.95126219967,), {}), 'hash': 101450.95126219967}
 146     aten::view(t$158: bf16[1, 6, 7168], [6, 7168])  ->  t$161: bf16[6, 7168]  # {'input_hash': ((605.1124907890335, [None, None]), {}), 'hash': 605.1124907890335}
 147     aten::mm(t$161: bf16[6, 7168], t$160: bf16[7168, 18432])  ->  t$162: bf16[6, 18432]  # {'input_hash': ((605.1124907890335, 101450.95126219967), {}), 'hash': 788120.3197385073}
 148     aten::_unsafe_view(t$162: bf16[6, 18432], [1, 6, 18432])  ->  t$163: bf16[1, 6, 18432]  # {'input_hash': ((788120.3197385073, [None, None, None]), {}), 'hash': 788120.3197385073}
 149     aten::silu(t$163: bf16[1, 6, 18432])  ->  t$164: bf16[1, 6, 18432]  # {'input_hash': ((788120.3197385073,), {}), 'hash': 691.2022647857666}
 150     aten::t(t$165: bf16[18432, 7168])  ->  t$166: bf16[7168, 18432]  # {'input_hash': ((301350.00236657704,), {}), 'hash': 301350.00236657704}
 151     aten::view(t$158: bf16[1, 6, 7168], [6, 7168])  ->  t$167: bf16[6, 7168]  # {'input_hash': ((605.1124907890335, [None, None]), {}), 'hash': 605.1124907890335}
 152     aten::mm(t$167: bf16[6, 7168], t$166: bf16[7168, 18432])  ->  t$168: bf16[6, 18432]  # {'input_hash': ((605.1124907890335, 301350.00236657704), {}), 'hash': 2273.497127377428}
 153     aten::_unsafe_view(t$168: bf16[6, 18432], [1, 6, 18432])  ->  t$169: bf16[1, 6, 18432]  # {'input_hash': ((2273.497127377428, [None, None, None]), {}), 'hash': 2273.497127377428}
 154     aten::mul.Tensor(t$164: bf16[1, 6, 18432], t$169: bf16[1, 6, 18432])  ->  t$170: bf16[1, 6, 18432]  # {'input_hash': ((691.2022647857666, 2273.497127377428), {}), 'hash': 19.982094686583878}
 155     aten::t(t$171: bf16[7168, 18432])  ->  t$172: bf16[18432, 7168]  # {'input_hash': ((284068.9295961037,), {}), 'hash': 284068.9295961037}
 156     aten::view(t$170: bf16[1, 6, 18432], [6, 18432])  ->  t$173: bf16[6, 18432]  # {'input_hash': ((19.982094686583878, [None, None]), {}), 'hash': 19.982094686583878}
 157     aten::mm(t$173: bf16[6, 18432], t$172: bf16[18432, 7168])  ->  t$174: bf16[6, 7168]  # {'input_hash': ((19.982094686583878, 284068.9295961037), {}), 'hash': 265.6633223230019}
 158     aten::_unsafe_view(t$174: bf16[6, 7168], [1, 6, 7168])  ->  t$175: bf16[1, 6, 7168]  # {'input_hash': ((265.6633223230019, [None, None, None]), {}), 'hash': 265.6633223230019}
 159     aten::add.Tensor(t$149: bf16[1, 6, 7168], t$175: bf16[1, 6, 7168])  ->  t$176: bf16[1, 6, 7168]  # {'input_hash': ((832.5981237888336, 265.6633223230019), {}), 'hash': 873.137708440423}
 160     aten::_to_copy(t$176: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$177: f32[1, 6, 7168]  # {'input_hash': ((873.137708440423,), {'dtype': None}), 'hash': 873.137708440423}
 161     aten::pow.Tensor_Scalar(t$177: f32[1, 6, 7168], 2)  ->  t$178: f32[1, 6, 7168]  # {'input_hash': ((873.137708440423, None), {}), 'hash': 62.648662651667884}
 162     aten::mean.dim(t$178: f32[1, 6, 7168], [-1], True)  ->  t$179: f32[1, 6, 1]  # {'input_hash': ((62.648662651667884, [None], None), {}), 'hash': 0.008740047574974597}
 163     aten::add.Tensor(t$179: f32[1, 6, 1], 1e-06)  ->  t$180: f32[1, 6, 1]  # {'input_hash': ((0.008740047574974597, None), {}), 'hash': 0.008746047620661557}
 164     aten::rsqrt(t$180: f32[1, 6, 1])  ->  t$181: f32[1, 6, 1]  # {'input_hash': ((0.008746047620661557,), {}), 'hash': 162.54175567626953}
 165     aten::mul.Tensor(t$177: f32[1, 6, 7168], t$181: f32[1, 6, 1])  ->  t$182: f32[1, 6, 7168]  # {'input_hash': ((873.137708440423, 162.54175567626953), {}), 'hash': 22257.334139684648}
 166     aten::_to_copy(t$182: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$183: bf16[1, 6, 7168]  # {'input_hash': ((22257.334139684648,), {'dtype': None}), 'hash': 22258.00124192238}
 167     aten::mul.Tensor(t$184: bf16[7168], t$183: bf16[1, 6, 7168])  ->  t$185: bf16[1, 6, 7168]  # {'input_hash': ((2747.7391052246094, 22258.00124192238), {}), 'hash': 8401.77631354332}
 168     aten::alias(t$185: bf16[1, 6, 7168])  ->  t$186: bf16[1, 6, 7168]  # {'input_hash': ((8401.77631354332,), {}), 'hash': 8401.77631354332}
 169     aten::t(t$187: bf16[129280, 7168])  ->  t$188: bf16[7168, 129280]  # {'input_hash': ((58439660.51988735,), {}), 'hash': 58439660.51988734}
 170     aten::view(t$186: bf16[1, 6, 7168], [6, 7168])  ->  t$189: bf16[6, 7168]  # {'input_hash': ((8401.77631354332, [None, None]), {}), 'hash': 8401.77631354332}
 171     aten::mm(t$189: bf16[6, 7168], t$188: bf16[7168, 129280])  ->  t$190: bf16[6, 129280]  # {'input_hash': ((8401.77631354332, 58439660.51988734), {}), 'hash': 1418407.944872737}
 172     aten::_unsafe_view(t$190: bf16[6, 129280], [1, 6, 129280])  ->  t$191: bf16[1, 6, 129280]  # {'input_hash': ((1418407.944872737, [None, None, None]), {}), 'hash': 1418407.944872737}
 173 CausalLMOutputWithPast(loss=None, logits=tensor([[[ 9.6875e+00, -4.0938e+00,  8.0469e-01,  ...,  1.0312e+00,
 174            8.4766e-01,  7.7734e-01],
 175          [ 8.9355e-02, -1.8750e+00, -1.6113e-01,  ...,  1.0742e-01,
 176           -5.4932e-03,  1.8433e-02],
 177          [ 1.2031e+00, -2.3438e+00,  4.1406e-01,  ...,  3.3203e-01,
 178            3.0859e-01,  4.8828e-01],
 179          [ 5.1875e+00, -6.0938e+00,  2.0020e-01,  ...,  2.7148e-01,
 180            3.0859e-01,  1.6895e-01],
 181          [-1.2109e+00, -2.3750e+00,  5.0000e-01,  ...,  9.6680e-02,
 182            1.2695e-01,  6.4062e-01],
 183          [-1.5820e-01,  1.0010e-01,  2.4023e-01,  ...,  1.3281e-01,
 184            1.3086e-01,  3.8281e-01]]], dtype=torch.bfloat16), past_key_values=DynamicCache(layers=[DynamicLayer]), hidden_states=None, attentions=None)</pre>
      </div>
    </td>
    <td style="width: 50%; padding: 0; background: #fff5f5; border: 1px solid #ffd7d5; vertical-align: top;">
      <div style="max-height: 600px; overflow-y: auto; padding: 15px;">
        <pre style="margin: 0; font-size: 11px; line-height: 1.4; font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">
   1     aten::embedding(t$0: bf16[129280, 7168], t$1: i64[1, 6], 128815)  ->  t$2: bf16[1, 6, 7168]  # {'input_hash': ((27375526.173154227, 16171.0, None), {}), 'hash': 788.8026814290788}
   2     aten::arange(6, device=cpu, pin_memory=False)  ->  t$3: i64[6]  # {'hash': 15.0}
   3     aten::add.Tensor(t$3: i64[6], 0)  ->  t$4: i64[6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}
   4     aten::unsqueeze(t$4: i64[6], 0)  ->  t$5: i64[1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}
   5     aten::unsqueeze(t$6: f32[32], 0)  ->  t$7: f32[1, 32]  # {'input_hash': ((7.372678196482765e+31, None), {}), 'hash': 7.372678196482765e+31}
   6     aten::unsqueeze(t$7: f32[1, 32], 2)  ->  t$8: f32[1, 32, 1]  # {'input_hash': ((7.372678196482765e+31, None), {}), 'hash': 7.372678196482765e+31}
   7     aten::expand(t$8: f32[1, 32, 1], [1, -1, 1])  ->  t$9: f32[1, 32, 1]  # {'input_hash': ((7.372678196482765e+31, [None, None, None]), {}), 'hash': 7.372678196482765e+31}
   8     aten::unsqueeze(t$5: i64[1, 6], 1)  ->  t$10: i64[1, 1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}
   9     aten::_to_copy(t$10: i64[1, 1, 6], dtype=torch.float32)  ->  t$11: f32[1, 1, 6]  # {'input_hash': ((15.0,), {'dtype': None}), 'hash': 15.0}
  10     aten::expand(t$9: f32[1, 32, 1], [1, 32, 1])  ->  t$12: f32[1, 32, 1]  # {'input_hash': ((7.372678196482765e+31, [None, None, None]), {}), 'hash': 7.372678196482765e+31}
  11     aten::view(t$12: f32[1, 32, 1], [1, 32, 1])  ->  t$13: f32[1, 32, 1]  # {'input_hash': ((7.372678196482765e+31, [None, None, None]), {}), 'hash': 7.372678196482765e+31}
  12     aten::expand(t$11: f32[1, 1, 6], [1, 1, 6])  ->  t$14: f32[1, 1, 6]  # {'input_hash': ((15.0, [None, None, None]), {}), 'hash': 15.0}
  13     aten::view(t$14: f32[1, 1, 6], [1, 1, 6])  ->  t$15: f32[1, 1, 6]  # {'input_hash': ((15.0, [None, None, None]), {}), 'hash': 15.0}
  14     aten::bmm(t$13: f32[1, 32, 1], t$15: f32[1, 1, 6])  ->  t$16: f32[1, 32, 6]  # {'input_hash': ((7.372678196482765e+31, 15.0), {}), 'hash': 1.1059017487963386e+33}
  15     aten::_unsafe_view(t$16: f32[1, 32, 6], [1, 32, 6])  ->  t$17: f32[1, 32, 6]  # {'input_hash': ((1.1059017487963386e+33, [None, None, None]), {}), 'hash': 1.1059017487963386e+33}
  16     aten::transpose.int(t$17: f32[1, 32, 6], 1, 2)  ->  t$18: f32[1, 6, 32]  # {'input_hash': ((1.1059017487963386e+33, None, None), {}), 'hash': 1.1059017487963388e+33}
  17     aten::cat(['t$18: f32[1, 6, 32]', 't$18: f32[1, 6, 32]'], -1)  ->  t$19: f32[1, 6, 64]  # {'input_hash': (([1.1059017487963388e+33, 1.1059017487963388e+33], None), {}), 'hash': 2.2118034975926775e+33}
  18     aten::cos(t$19: f32[1, 6, 64])  ->  t$20: f32[1, 6, 64]  # {'input_hash': ((2.2118034975926775e+33,), {}), 'hash': 339.7364740315825}
  19     aten::mul.Tensor(t$20: f32[1, 6, 64], 1.0)  ->  t$21: f32[1, 6, 64]  # {'input_hash': ((339.7364740315825, None), {}), 'hash': 339.7364740315825}
  20     aten::sin(t$19: f32[1, 6, 64])  ->  t$22: f32[1, 6, 64]  # {'input_hash': ((2.2118034975926775e+33,), {}), 'hash': 76.23392300636053}
  21     aten::mul.Tensor(t$22: f32[1, 6, 64], 1.0)  ->  t$23: f32[1, 6, 64]  # {'input_hash': ((76.23392300636053, None), {}), 'hash': 76.23392300636053}
  22     aten::_to_copy(t$21: f32[1, 6, 64], dtype=torch.bfloat16)  ->  t$24: bf16[1, 6, 64]  # {'input_hash': ((339.7364740315825,), {'dtype': None}), 'hash': 339.74072265625}
  23     aten::_to_copy(t$23: f32[1, 6, 64], dtype=torch.bfloat16)  ->  t$25: bf16[1, 6, 64]  # {'input_hash': ((76.23392300636053,), {'dtype': None}), 'hash': 76.23876954196976}
  24     aten::_to_copy(t$2: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$26: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788,), {'dtype': None}), 'hash': 788.8026814290788}
  25     aten::pow.Tensor_Scalar(t$26: f32[1, 6, 7168], 2)  ->  t$27: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788, None), {}), 'hash': 47.63002008403455}
  26     aten::mean.dim(t$27: f32[1, 6, 7168], [-1], True)  ->  t$28: f32[1, 6, 1]  # {'input_hash': ((47.63002008403455, [None], None), {}), 'hash': 0.006644813169259578}
  27     aten::add.Tensor(t$28: f32[1, 6, 1], 1e-06)  ->  t$29: f32[1, 6, 1]  # {'input_hash': ((0.006644813169259578, None), {}), 'hash': 0.006650813214946538}
  28     aten::rsqrt(t$29: f32[1, 6, 1])  ->  t$30: f32[1, 6, 1]  # {'input_hash': ((0.006650813214946538,), {}), 'hash': 185.91046714782715}
  29     aten::mul.Tensor(t$26: f32[1, 6, 7168], t$30: f32[1, 6, 1])  ->  t$31: f32[1, 6, 7168]  # {'input_hash': ((788.8026814290788, 185.91046714782715), {}), 'hash': 23589.8728415073}
  30     aten::_to_copy(t$31: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$32: bf16[1, 6, 7168]  # {'input_hash': ((23589.8728415073,), {'dtype': None}), 'hash': 23589.859999522567}
  31     aten::mul.Tensor(t$33: bf16[7168], t$32: bf16[1, 6, 7168])  ->  t$34: bf16[1, 6, 7168]  # {'input_hash': ((297.11181640625, 23589.859999522567), {}), 'hash': 1156.63240952231}
  32     aten::t(t$35: bf16[1536, 7168])  ->  t$36: bf16[7168, 1536]  # {'input_hash': ((141606.8712687064,), {}), 'hash': 141606.8712687064}
  33     aten::view(t$34: bf16[1, 6, 7168], [6, 7168])  ->  t$37: bf16[6, 7168]  # {'input_hash': ((1156.63240952231, [None, None]), {}), 'hash': 1156.63240952231}
  34     aten::mm(t$37: bf16[6, 7168], t$36: bf16[7168, 1536])  ->  t$38: bf16[6, 1536]  # {'input_hash': ((1156.63240952231, 141606.8712687064), {}), 'hash': 920.8860853910446}
  35     aten::_unsafe_view(t$38: bf16[6, 1536], [1, 6, 1536])  ->  t$39: bf16[1, 6, 1536]  # {'input_hash': ((920.8860853910446, [None, None, None]), {}), 'hash': 920.8860853910446}
  36     aten::_to_copy(t$39: bf16[1, 6, 1536], dtype=torch.float32)  ->  t$40: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446,), {'dtype': None}), 'hash': 920.8860853910446}
  37     aten::pow.Tensor_Scalar(t$40: f32[1, 6, 1536], 2)  ->  t$41: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446, None), {}), 'hash': 288.5757960550672}
  38     aten::mean.dim(t$41: f32[1, 6, 1536], [-1], True)  ->  t$42: f32[1, 6, 1]  # {'input_hash': ((288.5757960550672, [None], None), {}), 'hash': 0.1878748619928956}
  39     aten::add.Tensor(t$42: f32[1, 6, 1], 1e-06)  ->  t$43: f32[1, 6, 1]  # {'input_hash': ((0.1878748619928956, None), {}), 'hash': 0.18788085784763098}
  40     aten::rsqrt(t$43: f32[1, 6, 1])  ->  t$44: f32[1, 6, 1]  # {'input_hash': ((0.18788085784763098,), {}), 'hash': 39.29507780075073}
  41     aten::mul.Tensor(t$40: f32[1, 6, 1536], t$44: f32[1, 6, 1])  ->  t$45: f32[1, 6, 1536]  # {'input_hash': ((920.8860853910446, 39.29507780075073), {}), 'hash': 5729.73890671269}
  42     aten::_to_copy(t$45: f32[1, 6, 1536], dtype=torch.bfloat16)  ->  t$46: bf16[1, 6, 1536]  # {'input_hash': ((5729.73890671269,), {'dtype': None}), 'hash': 5729.775722026825}
  43     aten::mul.Tensor(t$47: bf16[1536], t$46: bf16[1, 6, 1536])  ->  t$48: bf16[1, 6, 1536]  # {'input_hash': ((681.26416015625, 5729.775722026825), {}), 'hash': 2617.230959892273}
  44     aten::t(t$49: bf16[24576, 1536])  ->  t$50: bf16[1536, 24576]  # {'input_hash': ((98730.36440096781,), {}), 'hash': 98730.36440096781}
  45     aten::view(t$48: bf16[1, 6, 1536], [6, 1536])  ->  t$51: bf16[6, 1536]  # {'input_hash': ((2617.230959892273, [None, None]), {}), 'hash': 2617.230959892273}
  46     aten::mm(t$51: bf16[6, 1536], t$50: bf16[1536, 24576])  ->  t$52: bf16[6, 24576]  # {'input_hash': ((2617.230959892273, 98730.36440096781), {}), 'hash': 58005.382189182565}
  47     aten::_unsafe_view(t$52: bf16[6, 24576], [1, 6, 24576])  ->  t$53: bf16[1, 6, 24576]  # {'input_hash': ((58005.382189182565, [None, None, None]), {}), 'hash': 58005.382189182565}
  48     aten::view(t$53: bf16[1, 6, 24576], [1, 6, -1, 192])  ->  t$54: bf16[1, 6, 128, 192]  # {'input_hash': ((58005.382189182565, [None, None, None, None]), {}), 'hash': 58005.382189182565}
  49     aten::transpose.int(t$54: bf16[1, 6, 128, 192], 1, 2)  ->  t$55: bf16[1, 128, 6, 192]  # {'input_hash': ((58005.382189182565, None, None), {}), 'hash': 58005.382189182565}
  50     aten::split_with_sizes(t$55: bf16[1, 128, 6, 192], [128, 64], -1)  ->  ['t$56: bf16[1, 128, 6, 128]', 't$57: bf16[1, 128, 6, 64]']  # {'input_hash': ((58005.382189182565, [None, None], None), {}), 'hash': [3620.7217003386468, 54384.66048884392]}
  51     aten::t(t$58: bf16[576, 7168])  ->  t$59: bf16[7168, 576]  # {'input_hash': ((60000.702554143965,), {}), 'hash': 60000.702554143965}
  52     aten::view(t$34: bf16[1, 6, 7168], [6, 7168])  ->  t$60: bf16[6, 7168]  # {'input_hash': ((1156.63240952231, [None, None]), {}), 'hash': 1156.63240952231}
  53     aten::mm(t$60: bf16[6, 7168], t$59: bf16[7168, 576])  ->  t$61: bf16[6, 576]  # {'input_hash': ((1156.63240952231, 60000.702554143965), {}), 'hash': 925.7674539089203}
  54     aten::_unsafe_view(t$61: bf16[6, 576], [1, 6, 576])  ->  t$62: bf16[1, 6, 576]  # {'input_hash': ((925.7674539089203, [None, None, None]), {}), 'hash': 925.7674539089203}
  55     aten::split_with_sizes(t$62: bf16[1, 6, 576], [512, 64], -1)  ->  ['t$63: bf16[1, 6, 512]', 't$64: bf16[1, 6, 64]']  # {'input_hash': ((925.7674539089203, [None, None], None), {}), 'hash': [447.7636544704437, 478.00379943847656]}
  56     aten::_to_copy(t$63: bf16[1, 6, 512], dtype=torch.float32)  ->  t$65: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437,), {'dtype': None}), 'hash': 447.7636544704437}
  57     aten::pow.Tensor_Scalar(t$65: f32[1, 6, 512], 2)  ->  t$66: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437, None), {}), 'hash': 2416.9021114688285}
  58     aten::mean.dim(t$66: f32[1, 6, 512], [-1], True)  ->  t$67: f32[1, 6, 1]  # {'input_hash': ((2416.9021114688285, [None], None), {}), 'hash': 4.7205121368169785}
  59     aten::add.Tensor(t$67: f32[1, 6, 1], 1e-06)  ->  t$68: f32[1, 6, 1]  # {'input_hash': ((4.7205121368169785, None), {}), 'hash': 4.720518007874489}
  60     aten::rsqrt(t$68: f32[1, 6, 1])  ->  t$69: f32[1, 6, 1]  # {'input_hash': ((4.720518007874489,), {}), 'hash': 9.387228786945343}
  61     aten::mul.Tensor(t$65: f32[1, 6, 512], t$69: f32[1, 6, 1])  ->  t$70: f32[1, 6, 512]  # {'input_hash': ((447.7636544704437, 9.387228786945343), {}), 'hash': 681.568125214475}
  62     aten::_to_copy(t$70: f32[1, 6, 512], dtype=torch.bfloat16)  ->  t$71: bf16[1, 6, 512]  # {'input_hash': ((681.568125214475,), {'dtype': None}), 'hash': 681.4855184555054}
  63     aten::mul.Tensor(t$72: bf16[512], t$71: bf16[1, 6, 512])  ->  t$73: bf16[1, 6, 512]  # {'input_hash': ((3.804391235113144, 681.4855184555054), {}), 'hash': 5.108524536015466}
  64     aten::t(t$74: bf16[32768, 512])  ->  t$75: bf16[512, 32768]  # {'input_hash': ((54175.308287066175,), {}), 'hash': 54175.308287066175}
  65     aten::view(t$73: bf16[1, 6, 512], [6, 512])  ->  t$76: bf16[6, 512]  # {'input_hash': ((5.108524536015466, [None, None]), {}), 'hash': 5.108524536015466}
  66     aten::mm(t$76: bf16[6, 512], t$75: bf16[512, 32768])  ->  t$77: bf16[6, 32768]  # {'input_hash': ((5.108524536015466, 54175.308287066175), {}), 'hash': 111.4665225475328}
  67     aten::_unsafe_view(t$77: bf16[6, 32768], [1, 6, 32768])  ->  t$78: bf16[1, 6, 32768]  # {'input_hash': ((111.4665225475328, [None, None, None]), {}), 'hash': 111.4665225475328}
  68     aten::view(t$78: bf16[1, 6, 32768], [1, 6, -1, 256])  ->  t$79: bf16[1, 6, 128, 256]  # {'input_hash': ((111.4665225475328, [None, None, None, None]), {}), 'hash': 111.4665225475328}
  69     aten::transpose.int(t$79: bf16[1, 6, 128, 256], 1, 2)  ->  t$80: bf16[1, 128, 6, 256]  # {'input_hash': ((111.4665225475328, None, None), {}), 'hash': 111.4665225475328}
  70     aten::split_with_sizes(t$80: bf16[1, 128, 6, 256], [128, 128], -1)  ->  ['t$81: bf16[1, 128, 6, 128]', 't$82: bf16[1, 128, 6, 128]']  # {'input_hash': ((111.4665225475328, [None, None], None), {}), 'hash': [26.415770137362415, 85.05075241017039]}
  71     aten::view(t$64: bf16[1, 6, 64], [1, 1, 6, 64])  ->  t$83: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, [None, None, None, None]), {}), 'hash': 478.00379943847656}
  72     aten::unsqueeze(t$24: bf16[1, 6, 64], 1)  ->  t$84: bf16[1, 1, 6, 64]  # {'input_hash': ((339.74072265625, None), {}), 'hash': 339.74072265625}
  73     aten::unsqueeze(t$25: bf16[1, 6, 64], 1)  ->  t$85: bf16[1, 1, 6, 64]  # {'input_hash': ((76.23876954196976, None), {}), 'hash': 76.23876954196976}
  74     aten::view(t$57: bf16[1, 128, 6, 64], [1, 128, 6, 32, 2])  ->  t$86: bf16[1, 128, 6, 32, 2]  # {'input_hash': ((54384.66048884392, [None, None, None, None, None]), {}), 'hash': 54384.66048884392}
  75     aten::transpose.int(t$86: bf16[1, 128, 6, 32, 2], 4, 3)  ->  t$87: bf16[1, 128, 6, 2, 32]  # {'input_hash': ((54384.66048884392, None, None), {}), 'hash': 54384.66048884392}
  76     aten::clone(t$87: bf16[1, 128, 6, 2, 32], memory_format=torch.contiguous_format)  ->  t$88: bf16[1, 128, 6, 2, 32]  # {'input_hash': ((54384.66048884392,), {'memory_format': None}), 'hash': 54384.66048884392}
  77     aten::_unsafe_view(t$88: bf16[1, 128, 6, 2, 32], [1, 128, 6, 64])  ->  t$89: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, [None, None, None, None]), {}), 'hash': 54384.66048884392}
  78     aten::view(t$83: bf16[1, 1, 6, 64], [1, 1, 6, 32, 2])  ->  t$90: bf16[1, 1, 6, 32, 2]  # {'input_hash': ((478.00379943847656, [None, None, None, None, None]), {}), 'hash': 478.00379943847656}
  79     aten::transpose.int(t$90: bf16[1, 1, 6, 32, 2], 4, 3)  ->  t$91: bf16[1, 1, 6, 2, 32]  # {'input_hash': ((478.00379943847656, None, None), {}), 'hash': 478.00379943847656}
  80     aten::clone(t$91: bf16[1, 1, 6, 2, 32], memory_format=torch.contiguous_format)  ->  t$92: bf16[1, 1, 6, 2, 32]  # {'input_hash': ((478.00379943847656,), {'memory_format': None}), 'hash': 478.00379943847656}
  81     aten::_unsafe_view(t$92: bf16[1, 1, 6, 2, 32], [1, 1, 6, 64])  ->  t$93: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, [None, None, None, None]), {}), 'hash': 478.00379943847656}
  82     aten::mul.Tensor(t$89: bf16[1, 128, 6, 64], t$84: bf16[1, 1, 6, 64])  ->  t$94: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, 339.74072265625), {}), 'hash': 49226.70544719696}
  83     aten::slice.Tensor(t$89: bf16[1, 128, 6, 64], 3, 0, 32)  ->  t$95: bf16[1, 128, 6, 32]  # {'input_hash': ((54384.66048884392, None, None, None), {}), 'hash': 28109.381856679916}
  84     aten::slice.Tensor(t$89: bf16[1, 128, 6, 64], 3, 32, 9223372036854775807)  ->  t$96: bf16[1, 128, 6, 32]  # {'input_hash': ((54384.66048884392, None, None, None), {}), 'hash': 26275.278632164}
  85     aten::neg(t$96: bf16[1, 128, 6, 32])  ->  t$97: bf16[1, 128, 6, 32]  # {'input_hash': ((26275.278632164,), {}), 'hash': 26275.278632164}
  86     aten::cat(['t$97: bf16[1, 128, 6, 32]', 't$95: bf16[1, 128, 6, 32]'], -1)  ->  t$98: bf16[1, 128, 6, 64]  # {'input_hash': (([26275.278632164, 28109.381856679916], None), {}), 'hash': 54384.66048884392}
  87     aten::mul.Tensor(t$98: bf16[1, 128, 6, 64], t$85: bf16[1, 1, 6, 64])  ->  t$99: bf16[1, 128, 6, 64]  # {'input_hash': ((54384.66048884392, 76.23876954196976), {}), 'hash': 9243.289089380418}
  88     aten::add.Tensor(t$94: bf16[1, 128, 6, 64], t$99: bf16[1, 128, 6, 64])  ->  t$100: bf16[1, 128, 6, 64]  # {'input_hash': ((49226.70544719696, 9243.289089380418), {}), 'hash': 54235.34918093681}
  89     aten::mul.Tensor(t$93: bf16[1, 1, 6, 64], t$84: bf16[1, 1, 6, 64])  ->  t$101: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, 339.74072265625), {}), 'hash': 431.57656478881836}
  90     aten::slice.Tensor(t$93: bf16[1, 1, 6, 64], 3, 0, 32)  ->  t$102: bf16[1, 1, 6, 32]  # {'input_hash': ((478.00379943847656, None, None, None), {}), 'hash': 171.16246032714844}
  91     aten::slice.Tensor(t$93: bf16[1, 1, 6, 64], 3, 32, 9223372036854775807)  ->  t$103: bf16[1, 1, 6, 32]  # {'input_hash': ((478.00379943847656, None, None, None), {}), 'hash': 306.8413391113281}
  92     aten::neg(t$103: bf16[1, 1, 6, 32])  ->  t$104: bf16[1, 1, 6, 32]  # {'input_hash': ((306.8413391113281,), {}), 'hash': 306.8413391113281}
  93     aten::cat(['t$104: bf16[1, 1, 6, 32]', 't$102: bf16[1, 1, 6, 32]'], -1)  ->  t$105: bf16[1, 1, 6, 64]  # {'input_hash': (([306.8413391113281, 171.16246032714844], None), {}), 'hash': 478.00379943847656}
  94     aten::mul.Tensor(t$105: bf16[1, 1, 6, 64], t$85: bf16[1, 1, 6, 64])  ->  t$106: bf16[1, 1, 6, 64]  # {'input_hash': ((478.00379943847656, 76.23876954196976), {}), 'hash': 89.2572899106276}
  95     aten::add.Tensor(t$101: bf16[1, 1, 6, 64], t$106: bf16[1, 1, 6, 64])  ->  t$107: bf16[1, 1, 6, 64]  # {'input_hash': ((431.57656478881836, 89.2572899106276), {}), 'hash': 473.74784088134766}
  96     aten::expand(t$107: bf16[1, 1, 6, 64], [1, 128, 6, -1])  ->  t$108: bf16[1, 128, 6, 64]  # {'input_hash': ((473.74784088134766, [None, None, None, None]), {}), 'hash': 60639.7236328125}
  97     aten::cat(['t$56: bf16[1, 128, 6, 128]', 't$100: bf16[1, 128, 6, 64]'], -1)  ->  t$109: bf16[1, 128, 6, 192]  # {'input_hash': (([3620.7217003386468, 54235.34918093681], None), {}), 'hash': 57856.07088127546}
  98     aten::cat(['t$81: bf16[1, 128, 6, 128]', 't$108: bf16[1, 128, 6, 64]'], -1)  ->  t$110: bf16[1, 128, 6, 192]  # {'input_hash': (([26.415770137362415, 60639.7236328125], None), {}), 'hash': 60666.13940294986}
  99     aten::lift_fresh(t$111: bf16[0])  ->  t$111: bf16[0]  # {'input_hash': ((0.0,), {}), 'hash': 0.0}
 100     aten::lift_fresh(t$112: bf16[0])  ->  t$112: bf16[0]  # {'input_hash': ((0.0,), {}), 'hash': 0.0}
 101     aten::cat(['t$111: bf16[0]', 't$110: bf16[1, 128, 6, 192]'], -2)  ->  t$113: bf16[1, 128, 6, 192]  # {'input_hash': (([0.0, 60666.13940294986], None), {}), 'hash': 60666.13940294986}
 102     aten::cat(['t$112: bf16[0]', 't$82: bf16[1, 128, 6, 128]'], -2)  ->  t$114: bf16[1, 128, 6, 128]  # {'input_hash': (([0.0, 85.05075241017039], None), {}), 'hash': 85.05075241017039}
 103     aten::_to_copy(t$109: bf16[1, 128, 6, 192], dtype=torch.float32)  ->  t$115: f32[1, 128, 6, 192]  # {'input_hash': ((57856.07088127546,), {'dtype': None}), 'hash': 57856.07088127546}
 104     aten::_to_copy(t$113: bf16[1, 128, 6, 192], dtype=torch.float32)  ->  t$116: f32[1, 128, 6, 192]  # {'input_hash': ((60666.13940294986,), {'dtype': None}), 'hash': 60666.13940294986}
 105     aten::_to_copy(t$114: bf16[1, 128, 6, 128], dtype=torch.float32)  ->  t$117: f32[1, 128, 6, 128]  # {'input_hash': ((85.05075241017039,), {'dtype': None}), 'hash': 85.05075241017039}
 106     aten::mul.Scalar(t$115: f32[1, 128, 6, 192], 0.3677414565436974)  ->  t$118: f32[1, 128, 6, 192]  # {'input_hash': ((57856.07088127546, None), {}), 'hash': 21276.07629972101}
 107     aten::ones([6, 6], dtype=torch.bool, layout=torch.strided, device=cpu)  ->  t$119: b8[6, 6]  # {'hash': 36.0}
 108     aten::tril(t$119: b8[6, 6])  ->  t$120: b8[6, 6]  # {'input_hash': ((36.0,), {}), 'hash': 21.0}
 109     aten::scalar_tensor(-inf, dtype=torch.float32, device=cpu)  ->  t$121: f32[]  # {'hash': inf}
 110     aten::scalar_tensor(0.0, dtype=torch.float32, layout=torch.strided, device=cpu)  ->  t$122: f32[]  # {'hash': 0.0}
 111     aten::where.self(t$120: b8[6, 6], t$122: f32[], t$121: f32[])  ->  t$123: f32[6, 6]  # {'input_hash': ((21.0, 0.0, inf), {}), 'hash': inf}
 112     aten::transpose.int(t$116: f32[1, 128, 6, 192], -2, -1)  ->  t$124: f32[1, 128, 192, 6]  # {'input_hash': ((60666.13940294986, None, None), {}), 'hash': 60666.13940294986}
 113     aten::mul.Scalar(t$124: f32[1, 128, 192, 6], 0.3677414565436974)  ->  t$125: f32[1, 128, 192, 6]  # {'input_hash': ((60666.13940294986, None), {}), 'hash': 22309.455075962735}
 114     aten::expand(t$118: f32[1, 128, 6, 192], [1, 128, 6, 192])  ->  t$126: f32[1, 128, 6, 192]  # {'input_hash': ((21276.07629972101, [None, None, None, None]), {}), 'hash': 21276.07629972101}
 115     aten::view(t$126: f32[1, 128, 6, 192], [128, 6, 192])  ->  t$127: f32[128, 6, 192]  # {'input_hash': ((21276.07629972101, [None, None, None]), {}), 'hash': 21276.07629972101}
 116     aten::expand(t$125: f32[1, 128, 192, 6], [1, 128, 192, 6])  ->  t$128: f32[1, 128, 192, 6]  # {'input_hash': ((22309.455075962735, [None, None, None, None]), {}), 'hash': 22309.455075962735}
 117     aten::view(t$128: f32[1, 128, 192, 6], [128, 192, 6])  ->  t$129: f32[128, 192, 6]  # {'input_hash': ((22309.455075962735, [None, None, None]), {}), 'hash': 22309.455075962735}
 118     aten::bmm(t$127: f32[128, 6, 192], t$129: f32[128, 192, 6])  ->  t$130: f32[128, 6, 6]  # {'input_hash': ((21276.07629972101, 22309.455075962735), {}), 'hash': 20362.785286933184}
 119     aten::_unsafe_view(t$130: f32[128, 6, 6], [1, 128, 6, 6])  ->  t$131: f32[1, 128, 6, 6]  # {'input_hash': ((20362.785286933184, [None, None, None, None]), {}), 'hash': 20362.785286933184}
 120     aten::add.Tensor(t$131: f32[1, 128, 6, 6], t$123: f32[6, 6])  ->  t$132: f32[1, 128, 6, 6]  # {'input_hash': ((20362.785286933184, inf), {}), 'hash': inf}
 121     aten::_safe_softmax(t$132: f32[1, 128, 6, 6], -1)  ->  t$133: f32[1, 128, 6, 6]  # {'input_hash': ((inf, None), {}), 'hash': 767.9999987812288}
 122     aten::_to_copy(t$133: f32[1, 128, 6, 6], dtype=torch.bfloat16)  ->  t$134: bf16[1, 128, 6, 6]  # {'input_hash': ((767.9999987812288,), {'dtype': None}), 'hash': 767.9952215677288}
 123     aten::expand(t$133: f32[1, 128, 6, 6], [1, 128, 6, 6])  ->  t$135: f32[1, 128, 6, 6]  # {'input_hash': ((767.9999987812288, [None, None, None, None]), {}), 'hash': 767.9999987812288}
 124     aten::view(t$135: f32[1, 128, 6, 6], [128, 6, 6])  ->  t$136: f32[128, 6, 6]  # {'input_hash': ((767.9999987812288, [None, None, None]), {}), 'hash': 767.9999987812288}
 125     aten::expand(t$117: f32[1, 128, 6, 128], [1, 128, 6, 128])  ->  t$137: f32[1, 128, 6, 128]  # {'input_hash': ((85.05075241017039, [None, None, None, None]), {}), 'hash': 85.05075241017039}
 126     aten::view(t$137: f32[1, 128, 6, 128], [128, 6, 128])  ->  t$138: f32[128, 6, 128]  # {'input_hash': ((85.05075241017039, [None, None, None]), {}), 'hash': 85.05075241017039}
 127     aten::bmm(t$136: f32[128, 6, 6], t$138: f32[128, 6, 128])  ->  t$139: f32[128, 6, 128]  # {'input_hash': ((767.9999987812288, 85.05075241017039), {}), 'hash': 64.34864976890512}
 128     aten::_unsafe_view(t$139: f32[128, 6, 128], [1, 128, 6, 128])  ->  t$140: f32[1, 128, 6, 128]  # {'input_hash': ((64.34864976890512, [None, None, None, None]), {}), 'hash': 64.34864976890512}
 129     aten::_to_copy(t$140: f32[1, 128, 6, 128], dtype=torch.bfloat16)  ->  t$141: bf16[1, 128, 6, 128]  # {'input_hash': ((64.34864976890512,), {'dtype': None}), 'hash': 64.35423506208463}
 130     aten::transpose.int(t$141: bf16[1, 128, 6, 128], 1, 2)  ->  t$142: bf16[1, 6, 128, 128]  # {'input_hash': ((64.35423506208463, None, None), {}), 'hash': 64.35423506208463}
 131     aten::clone(t$142: bf16[1, 6, 128, 128], memory_format=torch.contiguous_format)  ->  t$143: bf16[1, 6, 128, 128]  # {'input_hash': ((64.35423506208463,), {'memory_format': None}), 'hash': 64.35423506208463}
 132     aten::view(t$143: bf16[1, 6, 128, 128], [1, 6, -1])  ->  t$144: bf16[1, 6, 16384]  # {'input_hash': ((64.35423506208463, [None, None, None]), {}), 'hash': 64.35423506208463}
 133     aten::t(t$145: bf16[7168, 16384])  ->  t$146: bf16[16384, 7168]  # {'input_hash': ((402437.33954404993,), {}), 'hash': 402437.33954404993}
 134     aten::view(t$144: bf16[1, 6, 16384], [6, 16384])  ->  t$147: bf16[6, 16384]  # {'input_hash': ((64.35423506208463, [None, None]), {}), 'hash': 64.35423506208463}
 135     aten::mm(t$147: bf16[6, 16384], t$146: bf16[16384, 7168])  ->  t$148: bf16[6, 7168]  # {'input_hash': ((64.35423506208463, 402437.33954404993), {}), 'hash': 236.73609862662852}
 136     aten::_unsafe_view(t$148: bf16[6, 7168], [1, 6, 7168])  ->  t$149: bf16[1, 6, 7168]  # {'input_hash': ((236.73609862662852, [None, None, None]), {}), 'hash': 236.73609862662852}
 137     aten::add.Tensor(t$2: bf16[1, 6, 7168], t$149: bf16[1, 6, 7168])  ->  t$150: bf16[1, 6, 7168]  # {'input_hash': ((788.8026814290788, 236.73609862662852), {}), 'hash': 831.3630962371826}
 138     aten::_to_copy(t$150: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$151: f32[1, 6, 7168]  # {'input_hash': ((831.3630962371826,), {'dtype': None}), 'hash': 831.3630962371826}
 139     aten::pow.Tensor_Scalar(t$151: f32[1, 6, 7168], 2)  ->  t$152: f32[1, 6, 7168]  # {'input_hash': ((831.3630962371826, None), {}), 'hash': 58.23028035571633}
 140     aten::mean.dim(t$152: f32[1, 6, 7168], [-1], True)  ->  t$153: f32[1, 6, 1]  # {'input_hash': ((58.23028035571633, [None], None), {}), 'hash': 0.008123643870931119}
 141     aten::add.Tensor(t$153: f32[1, 6, 1], 1e-06)  ->  t$154: f32[1, 6, 1]  # {'input_hash': ((0.008123643870931119, None), {}), 'hash': 0.008129643916618079}
 142     aten::rsqrt(t$154: f32[1, 6, 1])  ->  t$155: f32[1, 6, 1]  # {'input_hash': ((0.008129643916618079,), {}), 'hash': 169.5744113922119}
 143     aten::mul.Tensor(t$151: f32[1, 6, 7168], t$155: f32[1, 6, 1])  ->  t$156: f32[1, 6, 7168]  # {'input_hash': ((831.3630962371826, 169.5744113922119), {}), 'hash': 22519.253141999594}
 144     aten::_to_copy(t$156: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$157: bf16[1, 6, 7168]  # {'input_hash': ((22519.253141999594,), {'dtype': None}), 'hash': 22520.068055152893}
 145     aten::mul.Tensor(t$158: bf16[7168], t$157: bf16[1, 6, 7168])  ->  t$159: bf16[1, 6, 7168]  # {'input_hash': ((122.01806432008743, 22520.068055152893), {}), 'hash': 602.6456905032974}
 146     aten::t(t$160: bf16[18432, 7168])  ->  t$161: bf16[7168, 18432]  # {'input_hash': ((101450.95126219967,), {}), 'hash': 101450.95126219967}
 147     aten::view(t$159: bf16[1, 6, 7168], [6, 7168])  ->  t$162: bf16[6, 7168]  # {'input_hash': ((602.6456905032974, [None, None]), {}), 'hash': 602.6456905032974}
 148     aten::mm(t$162: bf16[6, 7168], t$161: bf16[7168, 18432])  ->  t$163: bf16[6, 18432]  # {'input_hash': ((602.6456905032974, 101450.95126219967), {}), 'hash': 780578.6303224564}
 149     aten::_unsafe_view(t$163: bf16[6, 18432], [1, 6, 18432])  ->  t$164: bf16[1, 6, 18432]  # {'input_hash': ((780578.6303224564, [None, None, None]), {}), 'hash': 780578.6303224564}
 150     aten::silu(t$164: bf16[1, 6, 18432])  ->  t$165: bf16[1, 6, 18432]  # {'input_hash': ((780578.6303224564,), {}), 'hash': 706.0014340877533}
 151     aten::t(t$166: bf16[18432, 7168])  ->  t$167: bf16[7168, 18432]  # {'input_hash': ((301350.00236657704,), {}), 'hash': 301350.00236657704}
 152     aten::view(t$159: bf16[1, 6, 7168], [6, 7168])  ->  t$168: bf16[6, 7168]  # {'input_hash': ((602.6456905032974, [None, None]), {}), 'hash': 602.6456905032974}
 153     aten::mm(t$168: bf16[6, 7168], t$167: bf16[7168, 18432])  ->  t$169: bf16[6, 18432]  # {'input_hash': ((602.6456905032974, 301350.00236657704), {}), 'hash': 2254.3726884126663}
 154     aten::_unsafe_view(t$169: bf16[6, 18432], [1, 6, 18432])  ->  t$170: bf16[1, 6, 18432]  # {'input_hash': ((2254.3726884126663, [None, None, None]), {}), 'hash': 2254.3726884126663}
 155     aten::mul.Tensor(t$165: bf16[1, 6, 18432], t$170: bf16[1, 6, 18432])  ->  t$171: bf16[1, 6, 18432]  # {'input_hash': ((706.0014340877533, 2254.3726884126663), {}), 'hash': 20.34076667185036}
 156     aten::t(t$172: bf16[7168, 18432])  ->  t$173: bf16[18432, 7168]  # {'input_hash': ((284068.9295961037,), {}), 'hash': 284068.9295961037}
 157     aten::view(t$171: bf16[1, 6, 18432], [6, 18432])  ->  t$174: bf16[6, 18432]  # {'input_hash': ((20.34076667185036, [None, None]), {}), 'hash': 20.34076667185036}
 158     aten::mm(t$174: bf16[6, 18432], t$173: bf16[18432, 7168])  ->  t$175: bf16[6, 7168]  # {'input_hash': ((20.34076667185036, 284068.9295961037), {}), 'hash': 268.8421404538676}
 159     aten::_unsafe_view(t$175: bf16[6, 7168], [1, 6, 7168])  ->  t$176: bf16[1, 6, 7168]  # {'input_hash': ((268.8421404538676, [None, None, None]), {}), 'hash': 268.8421404538676}
 160     aten::add.Tensor(t$150: bf16[1, 6, 7168], t$176: bf16[1, 6, 7168])  ->  t$177: bf16[1, 6, 7168]  # {'input_hash': ((831.3630962371826, 268.8421404538676), {}), 'hash': 872.0488355457783}
 161     aten::_to_copy(t$177: bf16[1, 6, 7168], dtype=torch.float32)  ->  t$178: f32[1, 6, 7168]  # {'input_hash': ((872.0488355457783,), {'dtype': None}), 'hash': 872.0488355457783}
 162     aten::pow.Tensor_Scalar(t$178: f32[1, 6, 7168], 2)  ->  t$179: f32[1, 6, 7168]  # {'input_hash': ((872.0488355457783, None), {}), 'hash': 62.40043227744725}
 163     aten::mean.dim(t$179: f32[1, 6, 7168], [-1], True)  ->  t$180: f32[1, 6, 1]  # {'input_hash': ((62.40043227744725, [None], None), {}), 'hash': 0.008705417509190738}
 164     aten::add.Tensor(t$180: f32[1, 6, 1], 1e-06)  ->  t$181: f32[1, 6, 1]  # {'input_hash': ((0.008705417509190738, None), {}), 'hash': 0.008711417554877698}
 165     aten::rsqrt(t$181: f32[1, 6, 1])  ->  t$182: f32[1, 6, 1]  # {'input_hash': ((0.008711417554877698,), {}), 'hash': 163.1077537536621}
 166     aten::mul.Tensor(t$178: f32[1, 6, 7168], t$182: f32[1, 6, 1])  ->  t$183: f32[1, 6, 7168]  # {'input_hash': ((872.0488355457783, 163.1077537536621), {}), 'hash': 22274.217669785372}
 167     aten::_to_copy(t$183: f32[1, 6, 7168], dtype=torch.bfloat16)  ->  t$184: bf16[1, 6, 7168]  # {'input_hash': ((22274.217669785372,), {'dtype': None}), 'hash': 22274.649938583374}
 168     aten::mul.Tensor(t$185: bf16[7168], t$184: bf16[1, 6, 7168])  ->  t$186: bf16[1, 6, 7168]  # {'input_hash': ((2747.7391052246094, 22274.649938583374), {}), 'hash': 8408.873062133789}
 169     aten::alias(t$186: bf16[1, 6, 7168])  ->  t$187: bf16[1, 6, 7168]  # {'input_hash': ((8408.873062133789,), {}), 'hash': 8408.873062133789}
 170     aten::t(t$188: bf16[129280, 7168])  ->  t$189: bf16[7168, 129280]  # {'input_hash': ((58439660.51988735,), {}), 'hash': 58439660.51988734}
 171     aten::view(t$187: bf16[1, 6, 7168], [6, 7168])  ->  t$190: bf16[6, 7168]  # {'input_hash': ((8408.873062133789, [None, None]), {}), 'hash': 8408.873062133789}
 172     aten::mm(t$190: bf16[6, 7168], t$189: bf16[7168, 129280])  ->  t$191: bf16[6, 129280]  # {'input_hash': ((8408.873062133789, 58439660.51988734), {}), 'hash': 1421614.0652765036}
 173     aten::_unsafe_view(t$191: bf16[6, 129280], [1, 6, 129280])  ->  t$192: bf16[1, 6, 129280]  # {'input_hash': ((1421614.0652765036, [None, None, None]), {}), 'hash': 1421614.0652765036}
 174 CausalLMOutputWithPast(loss=None, logits=tensor([[[ 9.6875e+00, -4.0938e+00,  8.0469e-01,  ...,  1.0312e+00,
 175            8.4766e-01,  7.7734e-01],
 176          [ 2.4707e-01, -1.8125e+00, -1.4551e-01,  ...,  1.2695e-01,
 177            4.2419e-03,  2.6733e-02],
 178          [ 1.2734e+00, -2.2969e+00,  4.1016e-01,  ...,  3.2422e-01,
 179            3.0664e-01,  4.7852e-01],
 180          [ 5.1250e+00, -6.3438e+00,  2.1973e-01,  ...,  2.7734e-01,
 181            3.2227e-01,  1.6895e-01],
 182          [-1.0469e+00, -2.1250e+00,  4.7266e-01,  ...,  8.1543e-02,
 183            9.8145e-02,  6.3281e-01],
 184          [ 2.3438e-01, -8.7402e-02,  3.1055e-01,  ...,  2.0117e-01,
 185             1.8848e-01,  4.2383e-01]]], dtype=torch.bfloat16), past_key_values=DynamicCache(layers=[DynamicLayer]), hidden_states=None, attentions=None)</pre>
      </div>
    </td>
  </tr>
</table>
</details>

Line 4 shows a difference. Let's check the stack trace for more details.
```bash
    ...

    # File: /mnt/disk3/yiliu4/t47/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:571 in forward, code: position_ids = cache_position.unsqueeze(0)
    aten::unsqueeze(t$3: i64[6], 0)  ->  t$4: i64[1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}

    # File: /mnt/disk3/yiliu4/t47/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:79 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    aten::unsqueeze(t$5: f32[32], 0)  ->  t$6: f32[1, 32]  # {'input_hash': ((3.9489362656597677, None), {}), 'hash': 3.9489362656597677}
```

```bash
    ...

    # File: /mnt/disk3/yiliu4/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py:623 in forward, code: position_ids = cache_position.unsqueeze(0)
    aten::unsqueeze(t$4: i64[6], 0)  ->  t$5: i64[1, 6]  # {'input_hash': ((15.0, None), {}), 'hash': 15.0}

    # File: /mnt/disk3/yiliu4/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py:109 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    aten::unsqueeze(t$6: f32[32], 0)  ->  t$7: f32[1, 32]  # {'input_hash': ((9.351147240759962e+31, None), {}), 'hash': 9.351147240759962e+31}
```
The input to `aten::unsqueeze` differs between versions. Let's check the source code.
```python
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) #<----------
        position_ids_expanded = position_ids[:, None, :].float()
    ...

```
We found that `self.inv_freq` differs between versions. After checking the code, we discovered that `self.inv_freq` is initialized in the `DeepseekV3RotaryEmbedding.__init__` stage. However, in v5, the model is initialized with the `meta` device, which requires post-processing in the `_init_weights` stage. These lines were commented out because they caused significant initialization overhead.
The problem was resolved by uncommenting the `_init_weights` implementation.
 
```python
@auto_docstring
class DeepseekV3PreTrainedModel(PreTrainedModel):
    config: DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DeepseekV3DecoderLayer,
        "attentions": DeepseekV3Attention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        pass
        # super()._init_weights(module)
        # if isinstance(module, DeepseekV3TopkRouter):
        #     init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        #     init.zeros_(module.e_score_correction_bias)
        # elif isinstance(module, DeepseekV3NaiveMoe):
        #     init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
        #     init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        # if "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
        #     rope_fn = (
        #         ROPE_INIT_FUNCTIONS[module.rope_type]
        #         if module.rope_type != "default"
        #         else module.compute_default_rope_parameters
        #     )
        #     buffer_value, _ = rope_fn(module.config)
        #     init.copy_(module.inv_freq, buffer_value)
        #     init.copy_(module.original_inv_freq, buffer_value)
```

