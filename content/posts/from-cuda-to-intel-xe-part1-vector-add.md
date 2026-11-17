+++
title = "From CUDA to Intel Xe — Part 1: Vector Add"
date = 2026-11-17T00:00:00+00:00
draft = false
tags = ["Intel Xe", "SYCL", "CUDA", "GPU programming", "oneAPI"]
categories = ["Technical"]
summary = "A hands-on, interactive introduction to Intel GPU kernel programming, comparing CUDA and SYCL vector-add code, execution hierarchy, and indexing."
showtoc = false
layout = "intel-xe-vector-add"
+++

If you're familiar with CUDA, **vector addition** is a practical way to get started with kernel development on Intel Xe GPUs.

In this first article, I compare CUDA and SYCL implementations of a 256-element vector add, with color-aligned kernel stages and interactive diagrams for work-groups, sub-groups, and indexing. It also introduces the Intel Arc Pro B60/B70 and Crescent Island GPU families.

This post contains the complete interactive article, including side-by-side CUDA and SYCL kernel code, the visualization controls, and a comparison of the execution models.

The article is a personal technical blog and does not represent Intel's views.
