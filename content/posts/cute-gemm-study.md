---
title: "Understanding CuTe GEMM: A Visual Study Guide"
date: 2025-05-23
tags: ["cuda", "cute", "gpu-programming", "cutlass"]
summary: "A series of interactive visual notes exploring how NVIDIA CuTe builds a GEMM kernel — from layout algebra to thread-level execution traces."
showtoc: true
---

## Introduction

When I started learning NVIDIA's [CuTe](https://github.com/NVIDIA/cutlass) (the layout algebra engine inside CUTLASS 3.x), the biggest challenge wasn't the math — it was building visual intuition for how abstract layouts map to real threads and real data.

These interactive pages are the study notes I created along the way. They trace how a simple Ampere GEMM kernel is designed using CuTe DSL, starting from the high-level architecture and drilling down to individual thread behavior. Each page is self-contained and interactive — you can modify parameters and see results update live.

## Reading Order

I recommend going through these in order. Each page builds on concepts introduced in the previous one.

### 1. [Kernel Design Overview](/cute-gemm-study/kernel_design_overview.html)

The big picture: how a CuTe GEMM kernel decomposes a matrix multiplication into tiled, pipelined work. Covers the three-level hierarchy (device → block → thread) and how CuTe layouts encode the tiling strategy.

### 2. [TV Layout & MMA Atom Explained](/cute-gemm-study/tv_layout_mma_atom_explained.html)

The core abstraction: what a TV (Thread-Value) layout is and how an MMA atom wraps a hardware matrix-multiply instruction with layout information. This is the fundamental building block that everything else composes on top of.

### 3. [TV Layout Visualizer](/cute-gemm-study/tv_layout_visualizer.html)

An interactive tool to play with TV layout formulas. Plug in different shapes and strides and see which threads own which elements. Useful for building intuition about how `TiledMma` works.

### 4. [Partition C Permutation Explained](/cute-gemm-study/partition_c_permutation_explained.html)

The bridge between a single MMA atom and a full tile: how CuTe's permutation mechanism replicates and distributes atom layouts across a larger output tile. This is where `make_tiled_mma` clicks.

### 5. [Step 1 Thread Trace](/cute-gemm-study/step1_thread_trace.html)

A concrete execution trace through the first step of a CuTe DSL GEMM kernel. Follow along as specific threads load data, execute MMA instructions, and write results — with exact register and shared memory contents shown.

### 6. [CuTe DSL vs Pure CUDA Comparison](/cute-gemm-study/cute_vs_cuda_comparison.html)

A side-by-side comparison of the same GEMM kernel written in CuTe DSL vs raw CUDA. Helps connect CuTe abstractions back to familiar CUDA concepts like `threadIdx`, shared memory indexing, and `__syncthreads()`.

### 7. [Step 1: CuTe vs CUDA Deep Dive](/cute-gemm-study/step1_cute_vs_cuda.html)

A detailed walkthrough comparing the Step 1 naive GEMM implementation in CuTe DSL against its CUDA equivalent — line by line, showing how each CuTe construct maps to concrete CUDA operations.

### 8. [Step 1: Thread Ownership Visualization](/cute-gemm-study/step1_thread_ownership.html)

Interactive visualization of which threads own which output elements in the Step 1 kernel — showing how CuTe's partition maps thread IDs to matrix coordinates.

### 9. [Partition C Derivation](/cute-gemm-study/partition_c_derivation.html)

Mathematical derivation of how `partition_C` computes thread-to-element ownership from TiledMma layout composition — the algebra behind the partition.

### 10. [Step 2–5: CuTe vs CUDA Progression](/cute-gemm-study/step2_to_step5_cute_vs_cuda.html)

Side-by-side comparison of Steps 2 through 5 (shared memory tiling, async copy, pipelining) in CuTe DSL vs CUDA, showing how each optimization maps between the two worlds.

## Prerequisites

To get the most out of these notes, you should be comfortable with:

- Basic CUDA programming (kernels, threads, blocks, shared memory)
- Matrix multiplication and tiling concepts
- What a GEMM is and why it matters for GPU performance

No prior CuTe or CUTLASS experience needed — that's what these pages teach.
