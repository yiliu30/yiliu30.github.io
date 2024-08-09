+++
title = 'Understanding CUDA Memory Usage with Example'
date = 2024-08-08T21:03:18-04:00
draft = false
+++



A tiny example for understanding the CUDA memory snapshot.

> https://pytorch.org/docs/stable/torch_cuda_memory.html

[full_code](images/vis.py)
```python
torch.cuda.memory._record_memory_history()


device = torch.device("cuda")

see_memory_usage("Before run")
one_gb_tensor = torch.randn(1024**3 // 4, dtype=torch.float32).to(device)
x1 = one_gb_tensor.clone()
x2 = one_gb_tensor.clone()
see_memory_usage("After allocating three 1GB tensors")

x_lst = torch.cat([one_gb_tensor, x1, x2], dim=0)
see_memory_usage("After concatenating three 1GB tensors")

del one_gb_tensor
see_memory_usage("After deleting three 1GB tensors")
torch.cuda.empty_cache()
see_memory_usage("After emptying the cache")
del x1, x2
see_memory_usage("After deleting the x1, x2")

del x_lst
see_memory_usage("After deleting the concatenated tensor")

torch.cuda.memory._dump_snapshot("my_snapshot_empty2.pickle")

```

### The visualization result with notes
> Open the [my_snapshot_empty2.pickle](images/my_snapshot_empty2.pickle) at [pytorch.org/memory_viz](https://pytorch.org/memory_viz)

![cuda_mem](images/cuda_mem.png)