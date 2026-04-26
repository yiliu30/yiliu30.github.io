+++
title = 'DeepSeek V4 KV Cache Design: How 1M Tokens Fit in 10 GiB'
date = 2026-04-26T10:00:00+00:00
draft = false
tags = ['deepseek', 'kv-cache', 'attention', 'vllm', 'inference']
categories = ['Technical']
comments = true
+++

DeepSeek V4 supports 1M-token context, yet its KV cache for a 61-layer model
fits in ~9.6 GiB (BF16) — a **6.3× reduction** over naive full attention. This
post breaks down how three orthogonal techniques combine to make that possible.

<!--more-->

## The Problem

Standard transformer attention stores one KV vector per token per layer. At 1M
tokens, 61 layers, and 512-dim MLA vectors (BF16), that's: `61 layers × 1,048,576 tokens × 512 × 2 bytes = ~61 GiB`.


That's just the KV cache — before model weights, activations, or anything else.
DeepSeek V4 reduces this to **~9.6 GiB** by combining three techniques.

## Three Primitives

### 1. Sliding Window Attention (SWA)

Only attend to the most recent **128 tokens**. A circular buffer overwrites the
oldest entry each step.

- **Result**: `[128, 512-dim]` per layer — fixed cost regardless of sequence
  length.
- **Trade-off**: Loses all context beyond 128 positions. Good for local patterns
  (syntax, short phrases).

### 2. Token Compression

Learn to compress `ratio` consecutive tokens into 1 entry via **gated pooling**:
softmax over learned scores × weighted sum of KV states.

Two variants:
- **C4A (ratio=4, overlap=True)**: Each compressed entry sees an 8-token
  overlapping window (4 current + 4 from the prior block) → `[S/4, 512-dim]`
- **C128A (ratio=128, overlap=False)**: Non-overlapping 128-token blocks →
  `[S/128, 512-dim]`

**Why overlap for C4A?** Without it, token 3 and token 4 end up in separate
compressed entries with zero shared context. The overlapping window lets the
compressor see across boundaries, preserving cross-block dependencies.

### 3. Indexer (Learned Sparse Selection)

Even after 4× compression, 1M tokens still yield 262K compressed entries.
Attending to all of them is expensive. The **indexer** learns to select only the
**top-512 most relevant** entries per query — like a retrieval system for KV
cache entries.

Key detail: the indexer uses its **own separate query** (64 heads × 128-dim)
and its **own compressor** (128-dim, not 512-dim), both smaller than the main
attention's. This keeps the scoring GEMM cheap enough to run every decode step.

- **Result**: Indexer KV cache `[S/4, 128-dim]` — used only for scoring, not
  attention.

## Layer Configurations

DeepSeek V4 Pro (61 layers) combines these primitives into two layer types:

### Compressed Sparse Attention (CSA) — 30 layers

**SWA + C4A Compression + Indexer**

Three KV caches per layer:

| Cache | Shape | Purpose |
|---|---|---|
| SWA | `[128, 512-dim]` | Sliding window of raw tokens |
| C4A main KV | `[S/4, 512-dim]` | Compressed entries for attention |
| C4A indexer KV | `[S/4, 128-dim]` | Smaller cache for top-k scoring |

### Heavily Compressed Attention (HCA) — 31 layers

**SWA + C128A Compression** (no indexer — too few entries to be selective about)

Two KV caches per layer:

| Cache | Shape | Purpose |
|---|---|---|
| SWA | `[128, 512-dim]` | Sliding window of raw tokens |
| C128A main KV | `[S/128, 512-dim]` | Heavily compressed, attend to all |

At 128:1 compression, 1M tokens produce only 8,192 entries. Attending to all
8K entries is cheaper than running a scoring + top-k pass, so no indexer is
needed.

## 1M Token Budget (BF16)

All entries stored as BF16: main KV = 512 × 2 = **1,024 bytes/slot**, indexer
KV = 128 × 2 = **256 bytes/slot**.

| Layer Type | KV Cache | Count | Slots/Layer | Bytes/Slot | Subtotal |
|---|---|---|---|---|---|
| CSA | SWA | 30 | 128 | 512 × 2 = 1,024 | 0.004 GiB |
| CSA | C4A main KV | 30 | 1M / 4 = 262,144 | 512 × 2 = 1,024 | 7.50 GiB |
| CSA | C4A indexer KV | 30 | 1M / 4 = 262,144 | 128 × 2 = 256 | 1.88 GiB |
| HCA | SWA | 31 | 128 | 512 × 2 = 1,024 | 0.004 GiB |
| HCA | C128A main KV | 31 | 1M / 128 = 8,192 | 512 × 2 = 1,024 | 0.25 GiB |
| **Total** | | **61** | | | **~9.62 GiB** |



**C4A dominates** (~97% of the budget). The indexer adds ~20% overhead on top
of the main KV — the cost of learned sparse selection. C128A and SWA are nearly
free.

> **Note on vLLM's FP8 implementation**: In practice, vLLM stores the KV cache
> in FP8 with a mixed layout: 448B FP8 NoPE + 128B BF16 RoPE + 7B UE8M0
> scales + 1B pad = **584 bytes/slot** for main KV, and 128B FP8 + 4B FP32
> scale = **132 bytes/slot** for indexer KV. This reduces the total from
> ~9.62 GiB (BF16) to **~5.39 GiB** at 1M tokens.

## CSA Pipeline: How a Decode Step Works

Here's the data flow for a single CSA layer during decode, with a 4096-token
context:

<img src="/images/ds_v4_csa_pipeline.svg" alt="CSA Pipeline Diagram" style="max-width: 100%; margin: 20px 0;" />

### Step 0: SWA Cache

Store the most recent 128 raw tokens in a circular buffer.

→ `[128, 512-dim]`

### Step 1: Main Compressor

Compress 4096 tokens into 1024 entries via overlapping gated pooling (8-token
window with stride 4).

→ `[4096 tokens] → [4096/4 = 1024, 512-dim]` in main KV cache

### Step 2: Indexer Compressor

A separate, smaller compressor produces 128-dim representations of the same
compressed positions.

→ `[4096 tokens] → [4096/4 = 1024, 128-dim]` in indexer KV cache

### Step 3: Indexer Scoring (DeepGEMM)

The indexer's own query (64 heads × 128-dim — separate from the main
attention's 128 heads × 512-dim) scores each compressed position via
`fp8_fp4_mqa_logits`:

→ `Indexer_Q [1, 64h, 128] @ K [1024, 128]ᵀ → logits [1, 1024]`

Low-bit FP8/FP4 operands go directly to tensor cores here. This is fine because
only the **ranking order** matters — we're picking top-k, not computing precise
attention values.

### Step 4: Top-k Selection

Select the 512 highest-scoring compressed positions.

→ `logits [1, 1024] → indices [1, 512]`

### Step 5: Sparse Attention (FlashMLA)

Gather the selected 512 compressed entries from the main KV cache, combine with
128 SWA entries, and run fused attention via FlashMLA:

→ `Main_Q [1, 128h, 512] × gathered_K [512+128 = 640, 512] → output [1, 512]`

The FP8 KV cache is dequantized to BF16 before reaching the tensor cores.
FlashMLA uses warpgroup specialization to overlap the dequant with MMA
operations.

## Design Insights

**Why two Q projections?** The main attention Q (128 heads × 512-dim = 65K
parameters per token) needs full precision to compute accurate attention values.
The indexer Q (64 heads × 128-dim = 8K parameters per token) only needs to
rank positions. This 8× reduction in scoring dimensionality keeps the indexer
fast enough to run every decode step.

**Why C4A gets an indexer but C128A doesn't?** At 4:1 compression, 1M tokens
produce 262K entries — far too many to attend to all. At 128:1, there are only
8K entries, which is cheaper to attend to than to score-and-select from.

**Why the 128-token window is so small?** The sliding window primarily exists
to solve a **causal gap** with C128A: a compressed entry summarizing positions
0–127 would leak future information to queries at position 50. The 128-slot raw
window ensures every query can access its local context without violating
causality.

**Why alternating C4A / C128A layers?** V4 Pro alternates between fine-grained
(4:1) and coarse (128:1) compressed attention every other layer. This gives the
model both detailed and broad views of the context at every pair of layers —
analogous to multi-scale feature pyramids in computer vision.

## References

- [vLLM blog: DeepSeek V4 support](https://vllm.ai/blog/deepseek-v4)
- [DeepSeek V4 Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)
- [DeepSeek V4 Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
- vLLM source: `vllm/model_executor/layers/deepseek_v4_attention.py`,
  `vllm/model_executor/layers/deepseek_compressor.py`,
  `vllm/model_executor/layers/sparse_attn_indexer.py`
