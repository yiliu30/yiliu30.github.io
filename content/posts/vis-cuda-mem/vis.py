import torch
import logging

logging.basicConfig(level=logging.INFO)


def see_memory_usage(message, force=True):
    # Modified from DeepSpeed
    import gc

    import torch.distributed as dist

    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logging.info(message)
    logging.info(
        f"AllocatedMem {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        MaxAllocatedMem {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        ReservedMem {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        MaxReservedMem {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB "
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()


# https://pytorch.org/docs/stable/torch_cuda_memory.html
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
