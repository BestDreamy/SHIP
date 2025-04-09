import os
import time
import torch
import torch.distributed as dist
import deep_reduce

def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    # print("rank: {}, num_ranks: {}, group: {}".format(dist.get_rank(), dist.get_world_size(), list(range(num_local_ranks * num_nodes))))

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert(rank == local_rank)
    assert(num_ranks == num_local_ranks)

    # test_ll_compatibility, num_rdma_bytes = False, 0

    deep_reduce.torchReduce(c, a, b, n)
    # buffer = ep.Buffer(group, int(1e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,
    #                         num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1))

if __name__ == '__main__':
    num_processes = 8

    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
