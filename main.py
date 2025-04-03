import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.cpp_extension import load

# ✅ 编译 CUDA 内核
custom_cuda = load(name="custom_cuda", sources=["my_kernel.cu"], verbose=True)

def worker(rank, world_size):
    """
    多进程启动 CUDA Kernel 并使用 NCCL 进行通信
    """
    # 配置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # 选择 GPU
    torch.cuda.set_device(rank)

    # 初始化 NCCL
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 创建张量
    tensor = torch.zeros(1024, device=f'cuda:{rank}')
    print(f"Rank {rank}: Before Kernel -> {tensor[:5]}")

    # ✅ 运行 CUDA Kernel
    custom_cuda.launch_kernel(tensor)

    print(f"Rank {rank}: After Kernel -> {tensor[:5]}")

    # ✅ NCCL 同步（all_reduce 计算总和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank}: After All-Reduce -> {tensor[:5]}")

    # 进程同步
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
