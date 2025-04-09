import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../psrc')))

from utils import MoEConfig, ProcessGroupInfo
from utils import init_dist

moe = MoEConfig(
    num_experts=8,
    experts_per_token=3,
    hidden_dim=512,
    max_num_tokens=10,
)

def work(
    rank: int,
    world_size: int,
    dp_size: int,
) -> None:
    # rank = pgi.rank
    # local_rank = pgi.local_rank
    # world_size = pgi.world_size
    # dp_rank = rank // dp_size
    # num_dp = world_size // dp_size
    # assert torch.cuda.current_device() == local_rank
    # device = pgi.device

    ata = AllToAll(
        max_num_tokens=moe.max_num_tokens,
        num_experts=moe.num_experts,
        experts_per_token=moe.experts_per_token,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=moe.hidden_dim,
        hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
        hidden_dim_scale_bytes=(
            0
            if moe.in_dtype.itemsize != 1
            else (
                (moe.hidden_dim + moe.block_size - 1)
                // moe.block_size
                * torch.float32.itemsize
            )
        ),
    )

    # Generate the same test data on all ranks
    rng = torch.Generator()
    rng.manual_seed(123)
    all_rank_data = [
        RankTestData(moe, rng, use_max_tokens=False) for _ in range(num_dp)
    ]
    rank_data = all_rank_data[dp_rank]

    # Collect info by expert
    expert_token_from: list[list[tuple[int, int]]] = [
        [] for _ in range(moe.num_experts)
    ]
    for i_rank, rd in enumerate(all_rank_data):
        for token_idx in range(rd.num_tokens):
            for expert_idx in rd.indices[token_idx]:
                expert_token_from[expert_idx].append((i_rank, token_idx))

if __name__ == '__main__':
    dp_size = 2
    world_size = 8
    torch.multiprocessing.spawn(work, args=(world_size, dp_size), nprocs=world_size)
